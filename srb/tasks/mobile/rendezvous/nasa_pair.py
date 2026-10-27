"""Rendezvous configuration for original DESDynI and NEAR visuals.

Vehicle USD files and physical parameters are explicit inputs. An optional
RGB/depth camera adds a sensor without hardware mass or collision mesh;
the experiment runner can update its pose with a kinematic mount.
"""
from pathlib import Path

from srb.core.action import ThrustActionCfg, ThrustActionGroup, ThrusterCfg
from srb.core.asset import Frame, OrbitalRobot, RigidObjectCfg
from srb.core.sim import CollisionPropertiesCfg, MeshCollisionPropertiesCfg, RigidBodyPropertiesCfg, UsdFileCfg

from .task import SceneCfg, Task, TaskCfg


class OriginalDesdyni(OrbitalRobot):
    frame_base: Frame = Frame()
    frame_payload_mount: Frame = Frame()
    frame_manipulator_mount: Frame = Frame()
    frame_onboard_camera: Frame = Frame(prim_relpath='camera_onboard')


def make_task_cfg(*, robot_usd, target_usd, offsets, exhaust_directions, powers,
                  fuel_capacity, fuel_consumption_rate, start, target_position,
                  target_quaternion, goal_position, duration_s, seed, onboard_camera=None):
    """Construct the existing rendezvous task with explicit original-model assets."""
    import numpy as np
    from srb.nmpc.quaternion import quaternion_to_rotation

    def spawn(path):
        return UsdFileCfg(
            usd_path=str(Path(path).resolve()), activate_contact_sensors=True,
            collision_props=CollisionPropertiesCfg(),
            mesh_collision_props=MeshCollisionPropertiesCfg(mesh_approximation='convexDecomposition'),
            rigid_props=RigidBodyPropertiesCfg(linear_damping=0., angular_damping=.05,
                                                enable_gyroscopic_forces=True, max_depenetration_velocity=5.),
        )

    robot = OriginalDesdyni(
        asset_cfg=RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/robot', spawn=spawn(robot_usd),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(np.asarray(start)+target_position)),
        ),
        actions=ThrustActionGroup(ThrustActionCfg(
            asset_name='robot',
            thrusters=tuple(ThrusterCfg(offset=tuple(p), direction=tuple(d), power=float(f))
                           for p,d,f in zip(offsets,exhaust_directions,powers)),
            fuel_capacity=float(fuel_capacity), fuel_consumption_rate=float(fuel_consumption_rate),
        )),
    )
    target = RigidObjectCfg(
        prim_path='{ENV_REGEX_NS}/target', spawn=spawn(target_usd),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(target_position),rot=tuple(target_quaternion)),
    )
    rotation = quaternion_to_rotation(target_quaternion)
    cfg = TaskCfg(robot=robot, scene=SceneCfg(target=target), seed=int(seed),
                  target_offset_pos=tuple(rotation.T@np.asarray(goal_position)),
                  target_offset_quat=(1.,0.,0.,0.), episode_length_s=float(duration_s))
    cfg.scene.num_envs = 1
    cfg.mean_motion_rad_s = 0.
    cfg.orbital_tidal_acceleration = False
    if onboard_camera is not None:
        from srb.core.sensor import CameraCfg, PinholeCameraCfg
        spec = onboard_camera
        width,height = spec['resolution_wh']
        cfg.scene.cam_onboard = CameraCfg(
            prim_path='{ENV_REGEX_NS}/robot/camera_onboard',
            offset=CameraCfg.OffsetCfg(pos=tuple(spec['position_B_m']),
                rot=tuple(spec['quaternion_BC_wxyz']),convention=spec['orientation_convention']),
            spawn=PinholeCameraCfg(focal_length=spec['focal_length'],
                horizontal_aperture=spec['horizontal_aperture'],vertical_aperture=spec['vertical_aperture'],
                clipping_range=tuple(spec['clipping_range_m'])),
            width=width,height=height,update_period=0.,data_types=['rgb','distance_to_image_plane'],
        )
    for name in ('randomize_robot_state','randomize_target_state'):
        event = getattr(cfg.events,name)
        event.params['pose_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
        event.params['velocity_range'] = {k:(0.,0.) for k in ('x','y','z','roll','pitch','yaw')}
    return cfg


class NasaPairTask(Task):
    """Existing free-floating rendezvous behavior with original NASA assets."""

    def __init__(self,cfg,**kwargs):
        super().__init__(cfg,**kwargs)
        self._target_marker.set_visibility(False)
