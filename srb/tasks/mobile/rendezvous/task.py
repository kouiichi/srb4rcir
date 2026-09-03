from typing import Sequence, Tuple

import torch

from srb import assets
from srb._typing import StepReturn
from srb.core.action import ThrustAction
from srb.core.asset import RigidObject, RigidObjectCfg
from srb.core.env import (
    OrbitalEnv,
    OrbitalEnvCfg,
    OrbitalEventCfg,
    OrbitalSceneCfg,
    ViewerCfg,
)
from srb.core.manager import EventTermCfg, SceneEntityCfg
from srb.core.marker import VisualizationMarkers, VisualizationMarkersCfg
from srb.core.mdp import reset_root_state_uniform
from srb.core.sim import ArrowCfg, PreviewSurfaceCfg, SimforgeAssetCfg
from srb.utils.cfg import configclass
from srb.utils.math import (
    combine_frame_transforms,
    deg_to_rad,
    matrix_from_quat,
    quat_error_magnitude,
    rotmat_to_rot6d,
    rpy_to_quat,
    subtract_frame_transforms,
)

##############
### Config ###
##############


@configclass
class SceneCfg(OrbitalSceneCfg):
    ## Assets
    target: RigidObjectCfg = assets.Cubesat().asset_cfg
    target.prim_path = "{ENV_REGEX_NS}/target"


@configclass
class EventCfg(OrbitalEventCfg):
    randomize_target_state: EventTermCfg = EventTermCfg(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {
                "x": (20.0, 30.0),
                "y": (-2.0, 2.0),
                "z": (-2.0, 2.0),
                "roll": (-torch.pi, torch.pi),
                "pitch": (-torch.pi, torch.pi),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-deg_to_rad(10.0), deg_to_rad(10.0)),
                "pitch": (-deg_to_rad(10.0), deg_to_rad(10.0)),
                "yaw": (-deg_to_rad(10.0), deg_to_rad(10.0)),
            },
        },
    )


@configclass
class TaskCfg(OrbitalEnvCfg):
    ## Scene
    scene: SceneCfg = SceneCfg()

    ## Events
    events: EventCfg = EventCfg()

    ## Time
    episode_length_s: float = 60.0
    is_finite_horizon: bool = True

    ## Viewer
    viewer: ViewerCfg = ViewerCfg(
        eye=(15.0, -24.0, 12.0),
        lookat=(15.0, 0.0, 0.25),
        origin_type="env",
    )

    ## Target offset
    target_offset_pos: Tuple[float, float, float] = (0.0, 0.0, 0.5)
    target_offset_quat: Tuple[float, float, float, float] = rpy_to_quat(0.0, 90.0, 0.0)

    ## Success criteria
    success_distance_threshold: float = 1
    success_attitude_error_threshold: float = deg_to_rad(20.0)
    success_relative_speed_threshold: float = 0.3
    success_hold_steps: int = 10
    target_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/target",
        markers={
            "target": ArrowCfg(
                tail_radius=0.01,
                tail_length=0.05,
                head_radius=0.02,
                head_length=0.025,
                visual_material=PreviewSurfaceCfg(emissive_color=(0.2, 0.2, 0.8)),
            )
        },
    )

    def __post_init__(self):
        super().__post_init__()

        # Scene: Target
        if isinstance(self.scene.target.spawn, SimforgeAssetCfg):
            self.scene.target.spawn.seed = self.seed + self.scene.num_envs


############
### Task ###
############


class Task(OrbitalEnv):
    cfg: TaskCfg

    def __init__(self, cfg: TaskCfg, **kwargs):
        super().__init__(cfg, **kwargs)

        ## Get scene assets
        self._target: RigidObject = self.scene["target"]
        self._target_marker: VisualizationMarkers = VisualizationMarkers(
            self.cfg.target_marker_cfg
        )

        ## Initialize buffers
        self._target_offset_pos = torch.tensor(
            self.cfg.target_offset_pos, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._target_offset_quat = torch.tensor(
            self.cfg.target_offset_quat, dtype=torch.float32, device=self.device
        ).repeat(self.num_envs, 1)
        self._success_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._success_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._prev_distance_robot_to_target = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._success_streak[env_ids] = 0
        self._success_once[env_ids] = False
        self._prev_distance_robot_to_target[env_ids] = 0.0

    def extract_step_return(self) -> StepReturn:
        ## Compute the target pose with offset
        tf_pos_target, tf_quat_target = combine_frame_transforms(
            self._target.data.root_pos_w,
            self._target.data.root_quat_w,
            self._target_offset_pos,
            self._target_offset_quat,
        )

        ## Visualize target
        self._target_marker.visualize(tf_pos_target, tf_quat_target)

        ## Get remaining fuel (if applicable)
        if self._thrust_action_term_key:
            thrust_action_term: ThrustAction = self.action_manager._terms[  # type: ignore
                self._thrust_action_term_key
            ]
            remaining_fuel = (
                thrust_action_term.remaining_fuel / thrust_action_term.cfg.fuel_capacity
            ).unsqueeze(-1)
        else:
            remaining_fuel = None

        step_return = _compute_step_return(
            ## Time
            episode_length=self.episode_length_buf,
            max_episode_length=self.max_episode_length,
            truncate_episodes=self.cfg.truncate_episodes,
            success_hold_steps=self.cfg.success_hold_steps,
            ## Actions
            act_current=self.action_manager.action,
            act_previous=self.action_manager.prev_action,
            ## States
            # Root
            tf_pos_robot=self._robot.data.root_pos_w,
            tf_quat_robot=self._robot.data.root_quat_w,
            vel_lin_robot=self._robot.data.root_lin_vel_b,
            vel_ang_robot=self._robot.data.root_ang_vel_b,
            vel_lin_target=self._target.data.root_lin_vel_b,
            vel_ang_target=self._target.data.root_ang_vel_b,
            # Transforms (world frame)
            tf_pos_target=tf_pos_target,
            tf_quat_target=tf_quat_target,
            vel_lin_robot_w=self._robot.data.root_lin_vel_w,
            vel_lin_target_w=self._target.data.root_lin_vel_w,
            # IMU
            imu_lin_acc=self._imu_robot.data.lin_acc_b,
            imu_ang_vel=self._imu_robot.data.ang_vel_b,
            # Fuel
            remaining_fuel=remaining_fuel,
            # Progress
            prev_distance_robot_to_target=self._prev_distance_robot_to_target,
            # Success
            prev_success_streak=self._success_streak,
            prev_success_once=self._success_once,
            success_distance_threshold=self.cfg.success_distance_threshold,
            success_attitude_error_threshold=self.cfg.success_attitude_error_threshold,
            success_relative_speed_threshold=self.cfg.success_relative_speed_threshold,
        )
        if step_return.info is not None:
            self._success_streak = step_return.info["success_streak"].to(
                dtype=torch.long
            )
            self._success_once = step_return.info["is_success"].to(dtype=torch.bool)
            self._prev_distance_robot_to_target = step_return.info[
                "distance_robot_to_target"
            ].to(dtype=torch.float32)
        return step_return


@torch.jit.script
def _compute_step_return(
    *,
    ## Time
    episode_length: torch.Tensor,
    max_episode_length: int,
    truncate_episodes: bool,
    success_hold_steps: int,
    ## Actions
    act_current: torch.Tensor,
    act_previous: torch.Tensor,
    ## States
    # Root
    tf_pos_robot: torch.Tensor,
    tf_quat_robot: torch.Tensor,
    vel_lin_robot: torch.Tensor,
    vel_ang_robot: torch.Tensor,
    vel_lin_target: torch.Tensor,
    vel_ang_target: torch.Tensor,
    # Transforms (world frame)
    tf_pos_target: torch.Tensor,
    tf_quat_target: torch.Tensor,
    vel_lin_robot_w: torch.Tensor,
    vel_lin_target_w: torch.Tensor,
    # IMU
    imu_lin_acc: torch.Tensor,
    imu_ang_vel: torch.Tensor,
    # Fuel
    remaining_fuel: torch.Tensor | None,
    # Progress
    prev_distance_robot_to_target: torch.Tensor,
    # Success
    prev_success_streak: torch.Tensor,
    prev_success_once: torch.Tensor,
    success_distance_threshold: float,
    success_attitude_error_threshold: float,
    success_relative_speed_threshold: float,
) -> StepReturn:
    num_envs = episode_length.size(0)
    dtype = tf_pos_robot.dtype
    device = episode_length.device

    ############
    ## States ##
    ############
    ## Root
    tf_rotmat_robot = matrix_from_quat(tf_quat_robot)
    tf_rot6d_robot = rotmat_to_rot6d(tf_rotmat_robot)

    ## Transforms (world frame)
    # Robot -> Target
    tf_pos_robot_to_target, tf_quat_robot_to_target = subtract_frame_transforms(
        t01=tf_pos_robot,
        q01=tf_quat_robot,
        t02=tf_pos_target,
        q02=tf_quat_target,
    )
    tf_rotmat_robot_to_target = matrix_from_quat(tf_quat_robot_to_target)
    tf_rot6d_robot_to_target = rotmat_to_rot6d(tf_rotmat_robot_to_target)

    distance_robot_to_target = torch.norm(tf_pos_robot_to_target, dim=-1)
    attitude_error_robot_to_target = quat_error_magnitude(
        tf_quat_robot,
        tf_quat_target,
    )
    relative_speed_robot_to_target = torch.norm(
        vel_lin_robot_w - vel_lin_target_w, dim=-1
    )

    ## Fuel
    remaining_fuel = (
        remaining_fuel
        if remaining_fuel is not None
        else torch.ones((num_envs, 1), dtype=dtype, device=device)
    )

    #############
    ## Rewards ##
    #############
    # Penalty: Action rate
    WEIGHT_ACTION_RATE = -0.05
    penalty_action_rate = WEIGHT_ACTION_RATE * torch.sum(
        torch.square(act_current - act_previous), dim=1
    )

    # Penalty: Fuel consumption
    WEIGHT_FUEL_CONSUMPTION = -8.0
    penalty_fuel_consumption = WEIGHT_FUEL_CONSUMPTION * torch.square(
        1.0 - remaining_fuel.squeeze(-1)
    )

    # Penalty: Angular velocity
    WEIGHT_ANGULAR_VELOCITY = -0.05
    MAX_ANGULAR_VELOCITY_PENALTY = -5.0
    penalty_angular_velocity = torch.clamp_min(
        WEIGHT_ANGULAR_VELOCITY * torch.sum(torch.square(vel_ang_robot), dim=1),
        min=MAX_ANGULAR_VELOCITY_PENALTY,
    )

    # Penalty: Distance | Robot <--> Target
    GOAL_DISTANCE = 1.5
    MAX_APPROACH_DISTANCE = 30.0
    WEIGHT_DISTANCE_ROBOT_TO_TARGET = -16.0
    distance_error = torch.clamp_min(distance_robot_to_target - GOAL_DISTANCE, 0.0)
    penalty_distance_robot_to_target = WEIGHT_DISTANCE_ROBOT_TO_TARGET * torch.square(
        distance_error / MAX_APPROACH_DISTANCE
    )

    # Reward: Per-step progress toward the target
    WEIGHT_PROGRESS = 30.0
    progress_robot_to_target = torch.clamp(
        prev_distance_robot_to_target - distance_robot_to_target,
        min=-1.0,
        max=1.0,
    )
    reward_progress_robot_to_target = torch.where(
        episode_length <= 1,
        torch.zeros_like(progress_robot_to_target),
        WEIGHT_PROGRESS * progress_robot_to_target,
    )

    # Penalty: Attitude error | Robot <--> Target
    WEIGHT_ATTITUDE_ERROR = -2.0
    penalty_attitude_error_robot_to_target = WEIGHT_ATTITUDE_ERROR * torch.square(
        attitude_error_robot_to_target
    )

    # Penalty: Relative speed, weighted more strongly near the target
    WEIGHT_RELATIVE_SPEED = -3.0
    relative_speed_near_weight = torch.exp(-distance_error / 7.0)
    penalty_relative_speed_robot_to_target = (
        WEIGHT_RELATIVE_SPEED
        * relative_speed_near_weight
        * torch.square(relative_speed_robot_to_target)
    )

    # Reward: Distance (linear and angular) | Robot <--> Target (precision rendezvous)
    WEIGHT_PRECISION_RENDEZVOUS = 64.0
    TANH_STD_PRECISION_RENDEZVOUS_POS = 3.0
    TANH_STD_PRECISION_RENDEZVOUS_QUAT = 0.5
    reward_precision_rendezvous = WEIGHT_PRECISION_RENDEZVOUS * (
        1.0
        - torch.tanh(
            (distance_error / TANH_STD_PRECISION_RENDEZVOUS_POS)
            + (
                torch.norm(
                    torch.norm(
                        tf_rotmat_robot_to_target
                        - torch.eye(3, device=device).unsqueeze(0),
                        dim=-1,
                    ),
                    dim=-1,
                )
                / TANH_STD_PRECISION_RENDEZVOUS_QUAT
            )
        )
    )

    #############
    ## Success ##
    #############
    success_instant = (
        (distance_robot_to_target < success_distance_threshold)
        & (attitude_error_robot_to_target < success_attitude_error_threshold)
        & (relative_speed_robot_to_target < success_relative_speed_threshold)
    )
    success_streak = torch.where(
        success_instant,
        prev_success_streak + 1,
        torch.zeros_like(prev_success_streak),
    )
    success_now = success_streak >= success_hold_steps
    success_new = success_now & ~prev_success_once
    is_success = prev_success_once | success_now

    # Reward: Success
    WEIGHT_SUCCESS_INSTANT = 5.0
    WEIGHT_SUCCESS = 100.0
    reward_success_instant = WEIGHT_SUCCESS_INSTANT * success_instant.to(dtype)
    reward_success = WEIGHT_SUCCESS * success_new.to(dtype)

    ##################
    ## Terminations ##
    ##################
    # No termination condition
    termination = torch.zeros(num_envs, dtype=torch.bool, device=device)
    # Truncation
    truncation = (
        episode_length >= max_episode_length
        if truncate_episodes
        else torch.zeros(num_envs, dtype=torch.bool, device=device)
    )

    return StepReturn(
        {
            "state": {
                "tf_rot6d_robot": tf_rot6d_robot,
                "vel_lin_robot": vel_lin_robot,
                "vel_ang_robot": vel_ang_robot,
                "vel_lin_target": vel_lin_target,
                "vel_ang_target": vel_ang_target,
                "tf_pos_robot_to_target": tf_pos_robot_to_target,
                "tf_rot6d_robot_to_target": tf_rot6d_robot_to_target,
            },
            "proprio": {
                "imu_lin_acc": imu_lin_acc,
                "imu_ang_vel": imu_ang_vel,
                "remaining_fuel": remaining_fuel,
            },
        },
        {
            "penalty_action_rate": penalty_action_rate,
            "penalty_fuel_consumption": penalty_fuel_consumption,
            "penalty_angular_velocity": penalty_angular_velocity,
            "penalty_distance_robot_to_target": penalty_distance_robot_to_target,
            "reward_progress_robot_to_target": reward_progress_robot_to_target,
            "penalty_attitude_error_robot_to_target": penalty_attitude_error_robot_to_target,
            "penalty_relative_speed_robot_to_target": penalty_relative_speed_robot_to_target,
            "reward_precision_rendezvous": reward_precision_rendezvous,
            "reward_success_instant": reward_success_instant,
            "reward_success": reward_success,
        },
        termination,
        truncation,
        {
            "is_success": is_success,
            "success_instant": success_instant,
            "success_streak": success_streak,
            "distance_robot_to_target": distance_robot_to_target,
            "attitude_error_robot_to_target": attitude_error_robot_to_target,
            "relative_speed_robot_to_target": relative_speed_robot_to_target,
        },
    )
