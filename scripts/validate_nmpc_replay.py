#!/usr/bin/env python3
"""Offline NMPC planning, open-loop SRB replay, and measured-state closed-loop validation.

Use the SRB environment wrapper for both commands. ``plan`` never starts Kit;
``replay`` loads saved duties unless --closed-loop requests measured-state feedback. The replay uses
real ThrustAction/PhysX steps, not pose playback. Only the initial state is set.
The native SRB success flag and a separate 10 cm / 5 cm/s hover check are both
recorded. No controller, task source, or default configuration is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(val) for val in value]
    return value


def save_json(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def warm_after_prefix(result, consumed, previous_duty):
    """Pre-shift by K-1; AcadosExpert applies its existing one-step duty shift.

    K=1 preserves the historical state/duty warm-start convention exactly.
    The actual last executed duty, not a predicted future one, is rate memory.
    """
    def advance(array):
        indices = np.minimum(np.arange(len(array)) + consumed - 1, len(array) - 1)
        return array[indices].copy()
    if not 1 <= consumed <= len(result.predicted_duty):
        raise ValueError('consumed prefix must lie inside the prediction horizon')
    return {'x': advance(result.predicted_state), 'u': advance(result.predicted_duty),
            'previous_duty': np.asarray(previous_duty).copy()}


def metrics(states, duties, params, refs):
    q = states[:, 6:10] / np.linalg.norm(states[:, 6:10], axis=1)[:, None]
    qr = refs[:, 6:10] / np.linalg.norm(refs[:, 6:10], axis=1)[:, None]
    angles = 2 * np.arccos(np.clip(np.abs(np.sum(q * qr, axis=1)), 0, 1))
    errors = np.linalg.norm(states[:, :3] - refs[:, :3], axis=1)
    speeds = np.linalg.norm(states[:, 3:6] - refs[:, 3:6], axis=1)
    good = (errors < 0.1) & (speeds < 0.05) & (angles < np.deg2rad(5))
    streak, longest = 0, 0
    for value in good:
        streak = streak + 1 if value else 0
        longest = max(streak, longest)
    acceleration = np.diff(states[:, 3:6], axis=0) / params.control_period_s
    jerk = np.diff(acceleration, axis=0) / params.control_period_s
    rates = np.diff(np.vstack((np.zeros(16), duties)), axis=0) / params.control_period_s
    return {
        "steps": len(duties),
        "duration_s": len(duties) * params.control_period_s,
        "final_position_error_m": errors[-1],
        "min_position_error_m": np.min(errors),
        "final_relative_speed_m_s": speeds[-1],
        "max_speed_m_s": np.max(np.linalg.norm(states[:, 3:6], axis=1)),
        "final_attitude_error_deg": np.rad2deg(angles[-1]),
        "min_center_distance_m": np.min(np.linalg.norm(states[:, :3] - params.koz_center_L_m, axis=1)),
        "koz_radius_m": params.koz_radius_m,
        "min_koz_clearance_m": np.min(np.linalg.norm(states[:, :3] - params.koz_center_L_m, axis=1)) - params.koz_radius_m,
        "max_acceleration_m_s2": np.max(np.linalg.norm(acceleration, axis=1)) if len(acceleration) else 0,
        "max_jerk_m_s3": np.max(np.linalg.norm(jerk, axis=1)) if len(jerk) else 0,
        "rms_jerk_m_s3": np.sqrt(np.mean(jerk**2)) if len(jerk) else 0,
        "max_duty_rate_per_s": np.max(np.abs(rates)) if len(rates) else 0,
        "duty_rate_limit_per_s": params.duty_rate_per_s,
        "max_duty_jump": np.max(np.abs(rates)) * params.control_period_s if len(rates) else 0,
        "hover_hold_achieved": max(0, longest - 1) * params.control_period_s >= 1.0,
        "longest_hover_hold_s": max(0, longest - 1) * params.control_period_s,
        "final_hover_hold_s": max(0, streak - 1) * params.control_period_s,
        "hover_criteria": "position < 0.1 m, speed < 0.05 m/s, attitude < 5 deg for >= 1 s",
    }


def plot(path, states, duties, params, refs, simulated=None, label="NMPC plan"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(states)) * params.control_period_s
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes[0, 0].plot(states[:, 0], states[:, 1], label=label)
    circle = plt.Circle(params.koz_center_L_m[:2], params.koz_radius_m, color="red", alpha=0.2, label="KOZ (XY projection)")
    axes[0, 0].add_patch(circle)
    axes[0, 0].scatter(refs[-1, 0], refs[-1, 1], marker="x", color="black", label="Goal")
    axes[0, 0].set(xlabel="x [m]", ylabel="y [m]", aspect="equal")
    axes[0, 1].plot(t, np.linalg.norm(states[:, :3] - refs[:, :3], axis=1), label=label)
    axes[0, 1].set(ylabel="Position error [m]", xlabel="Time [s]")
    axes[0, 2].plot(t, np.linalg.norm(states[:, 3:6], axis=1), label=label)
    axes[0, 2].set(ylabel="Speed [m/s]", xlabel="Time [s]")
    axes[1, 0].plot(t, np.linalg.norm(states[:, :3], axis=1) - params.koz_radius_m, label=label)
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set(ylabel="Center-to-KOZ clearance [m]", xlabel="Time [s]")
    axes[1, 1].plot(t[:-1], duties)
    axes[1, 1].set(ylabel="Thruster duty", xlabel="Time [s]")
    acc = np.diff(states[:, 3:6], axis=0) / params.control_period_s
    axes[1, 2].plot(t[1:], np.linalg.norm(acc, axis=1), label=label)
    axes[1, 2].set(ylabel="Acceleration [m/s²]", xlabel="Time [s]")
    if simulated is not None:
        ts = t[:len(simulated)]
        axes[0, 0].plot(simulated[:, 0], simulated[:, 1], "--", label="PhysX replay")
        axes[0, 1].plot(ts, np.linalg.norm(simulated[:, :3] - refs[:len(simulated), :3], axis=1), "--", label="PhysX replay")
        axes[0, 2].plot(ts, np.linalg.norm(simulated[:, 3:6], axis=1), "--", label="PhysX replay")
        axes[1, 0].plot(ts, np.linalg.norm(simulated[:, :3], axis=1) - params.koz_radius_m, "--", label="PhysX replay")
        axes[1, 2].plot(ts[1:], np.linalg.norm(np.diff(simulated[:, 3:6], axis=0) / params.control_period_s, axis=1), "--", label="PhysX replay")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        if ax is not axes[1, 1]:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plan(args):
    from srb.nmpc import AcadosExpert, load_nmpc_config
    from srb.nmpc.dynamics import rk4_step, reference_trajectory
    from srb.nmpc.constraints import koz_terms

    cfg = load_nmpc_config(args.config)
    params = cfg.task_params
    if args.free_floating:
        params = replace(params, mean_motion_rad_s=0.0)
    if args.fov:
        params = replace(params, enable_fov=True)
    x = np.array([*args.start, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    expert = AcadosExpert(params, nlp_solver_type=cfg.solver.nlp_solver_type,
                          max_iterations=cfg.solver.production_max_iterations,
                          qp_solver_cond_N=cfg.solver.qp_solver_cond_N,
                          qp_solver_iter_max=cfg.solver.qp_solver_iter_max,
                          code_export_root=cfg.resolved_code_export_root(ROOT))
    states, duties, runtimes, margins = [x.copy()], [], [], []
    warm = None
    failure = None
    dense_clearance = float(np.linalg.norm(x[:3]) - params.koz_radius_m)
    dense_ecbf = None
    clipping = 0.0
    for k in range(args.steps):
        started = time.perf_counter()
        result = expert.solve(x, warm_start=warm)
        runtimes.append(time.perf_counter() - started)
        if not result.accepted:
            failure = {"step": k, "time_s": k * params.control_period_s,
                       "status": result.status, "reason": result.failure_reason,
                       "margins": result.constraint_margins, "state": x}
            break
        # The solver quality gate admits numerical bound errors <= 1e-6.
        # The adapter requires exact [0,1]. Record this roundoff projection.
        u = np.clip(result.actual_duty, 0.0, 1.0)
        clipping = max(clipping, float(np.max(np.abs(u - result.actual_duty))))
        if clipping > 1e-6:
            raise RuntimeError("Duty correction exceeds solver feasibility tolerance")
        dense_x = x.copy()
        for _ in range(5):
            dense_x = rk4_step(dense_x, u, params, dt_s=0.01)
            dense_clearance = min(dense_clearance, float(np.linalg.norm(dense_x[:3]) - params.koz_radius_m))
            ecbf = koz_terms(dense_x, u, params)[3]
            dense_ecbf = ecbf if dense_ecbf is None else min(dense_ecbf, ecbf)
        x = rk4_step(x, u, params)
        states.append(x.copy())
        duties.append(u.copy())
        margins.append(result.constraint_margins)
        warm = {"x": result.predicted_state, "u": result.predicted_duty, "previous_duty": u}
        if k % 100 == 0:
            print(f"plan step={k} position={x[:3]} speed={np.linalg.norm(x[3:6]):.4f}", flush=True)
    states = np.asarray(states)
    duties = np.asarray(duties).reshape(-1, 16)
    refs = reference_trajectory(params, np.arange(len(states)) * params.control_period_s)
    report = metrics(states, duties, params, refs)
    report.update({"failure": failure, "config": str(args.config), "params": asdict(params),
                   "requested_steps": args.steps, "all_requested_steps_planned": failure is None,
                   "dense_check_dt_s": 0.01, "dense_min_koz_clearance_m": dense_clearance,
                   "dense_min_ecbf": dense_ecbf, "max_roundoff_duty_correction": clipping,
                   "first_solve_wall_s_including_build": runtimes[0],
                   "hot_solve_p99_ms": float(np.percentile(runtimes[1:], 99) * 1000) if len(runtimes) > 1 else None})
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")
    save_json(args.output / "plan.json", report)
    np.savez_compressed(args.output / "plan.npz", states=states, duties=duties, refs=refs, solve_wall_s=runtimes)
    if len(duties):
        plot(args.output / "plan.png", states, duties, params, refs)
    print(json.dumps(jsonable({k: v for k, v in report.items() if k != "params"}), indent=2), flush=True)
    return 0 if failure is None else 1


def replay(args):
    from srb.nmpc import TaskParams
    report = json.loads((args.plan / "plan.json").read_text())
    values = report["params"]
    values["action_block_lengths"] = tuple(values["action_block_lengths"])
    params = TaskParams(**values)
    if args.matched_executor:
        params = replace(params, mean_motion_rad_s=0.)
    if args.duty_rate is not None:
        params = replace(params, duty_rate_per_s=args.duty_rate)
    if args.horizon_steps is not None:
        base_blocks = (3, 3, 3, 3, 2, 2, 2, 2)
        params = replace(params, horizon_steps=args.horizon_steps,
                         action_block_lengths=base_blocks * (args.horizon_steps // 20))
    if args.execute_steps > params.horizon_steps:
        raise ValueError('execute_steps cannot exceed horizon_steps')
    data = np.load(args.plan / "plan.npz", allow_pickle=False)
    states, duties, refs = data["states"], data["duties"], data["refs"]
    if not len(duties):
        raise RuntimeError("Plan contains no accepted executable commands")
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'invocation.json', {
        'cwd': str(Path.cwd()), 'argv': sys.argv, 'python': sys.executable,
        'command': shlex.join(sys.argv),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'CUDA_DEVICE_ORDER': os.environ.get('CUDA_DEVICE_ORDER'),
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS'),
        'kit_device': args.device, 'seed': args.seed,
    })
    if args.physics_only:
        if args.video:
            raise ValueError('--physics-only does not support video')
        from isaacsim import SimulationApp
        # Avoid waiting forever for an RTX viewport on a host without Kit CUDA interop.
        SimulationApp.DEFAULT_LAUNCHER_CONFIG['create_new_stage'] = False
    from srb.core.app import AppLauncher
    from srb.utils.path import SRB_APPS_DIR
    launcher = AppLauncher(headless=True, enable_cameras=args.video, multi_gpu=False, device=args.device,
                            experience=SRB_APPS_DIR / ("srb.headless.rendering.kit" if args.video else "srb.headless.kit"))
    env = None
    writer = None
    try:
        import torch
        from omni.physx import acquire_physx_interface
        from srb.tasks.mobile.rendezvous.task import Task16Rcs, Task16RcsCfg
        from srb.core.sensor import ContactSensorCfg
        from srb.nmpc.rcs import rcs_matrix_from_task
        from srb.nmpc.quaternion import quaternion_to_rotation
        if args.physics_only:
            import omni.usd
            omni.usd.get_context().new_stage()
        # Matched dynamics were validated with CPU PhysX. Camera rendering
        # still uses the Kit GPU and must not select a different physics backend.
        force_cpu_physx = args.physics_only or args.matched_executor
        acquire_physx_interface().overwrite_gpu_setting(0 if force_cpu_physx else 1)
        cfg = Task16RcsCfg()
        if args.matched_executor:
            cfg.orbital_tidal_acceleration = False
        cfg.scene.num_envs = 2 if args.check_mass_reset else 1
        cfg.episode_length_s = ((args.steps or len(duties)) + 100) * params.control_period_s
        cfg.viewer.eye = (-1.5, -6.0, 3.0)
        cfg.viewer.lookat = (-1.5, 0.0, 0.0)
        cfg.viewer.resolution = (960, 540)
        if args.surface_approach:
            cfg.viewer.eye = (-1.5, -3.2, 1.4)
            cfg.viewer.lookat = (-1.45, 0.0, 0.0)
        if args.stable_video_lighting:
            # Keep the Earth backdrop and illumination fixed for evaluation
            # recordings; the default interval events can hide the vehicles.
            cfg.events.randomize_sunlight_orientation = None
            cfg.events.randomize_sunlight_intensity = None
            cfg.events.randomize_skydome_orientation = None
        if args.video:
            save_json(args.output / 'video_config.json', {
                'fps': 1.0 / params.control_period_s,
                'resolution': cfg.viewer.resolution,
                'eye': cfg.viewer.eye, 'lookat': cfg.viewer.lookat,
                'stable_lighting': args.stable_video_lighting,
                'surface_tracking_camera': args.surface_approach,
                'video_fill_intensity': args.video_fill_intensity,
                'surface_display_materials': args.surface_display_materials,
                'frame_source': 'env.render() after each actual PhysX control step',
            })
        cfg.scene.replay_contacts = ContactSensorCfg(
            prim_path=cfg.scene.robot.prim_path, update_period=0.0,
            history_length=1,
        )
        if args.align_goal:
            cfg.target_offset_pos = tuple(refs[0, :3])
            cfg.target_offset_quat = tuple(refs[0, 6:10])
        class AuditedTask(Task16Rcs):
            def _setup_scene(self):
                super()._setup_scene()
                from pxr import Usd, UsdPhysics, PhysxSchema
                stage = self.sim.stage
                root = stage.GetPrimAtPath('/World/envs/env_0/robot')
                bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
                if len(bodies) != 1:
                    raise RuntimeError(f'Expected one robot rigid body; found {[str(b.GetPath()) for b in bodies]}')
                body = bodies[0]
                PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
                path = str(body.GetPath()).replace('/env_0/', '/env_.*/')
                self.scene.sensors['replay_contacts'].cfg.prim_path = path
                print('CONTACT BODY ' + path, flush=True)
                if args.physics_only:
                    self.sim.set_render_mode(self.sim.RenderMode.NO_GUI_OR_RENDERING)
                if force_cpu_physx:
                    acquire_physx_interface().overwrite_gpu_setting(0)
        env = AuditedTask(cfg, render_mode="rgb_array" if args.video else None)
        constructor_properties = None
        if args.check_mass_reset:
            constructor_properties = (
                env._robot.root_physx_view.get_masses().detach().cpu().numpy().copy(),
                env._robot.root_physx_view.get_inertias().detach().cpu().numpy().copy(),
            )
        env.reset(seed=args.seed)
        if args.video and args.video_fill_intensity:
            from pxr import Gf, UsdGeom, UsdLux
            # Evaluation-only illumination from both sides of the encounter.
            # These USD lights have no physics or controller attributes.
            for index, yaw in enumerate((-40., 40.)):
                light = UsdLux.DistantLight.Define(env.sim.stage, f'/World/replay_fill_{index}')
                light.CreateIntensityAttr(args.video_fill_intensity)
                light.CreateAngleAttr(15.)
                UsdGeom.Xformable(light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(65., yaw, 0.))
        if args.check_mass_reset:
            from check_thrust_mass_reset import check_mass_reset
            reset_check = check_mass_reset(env, constructor_properties)
            save_json(args.output / 'mass_reset_check.json', reset_check)
            print(json.dumps(reset_check, indent=2), flush=True)
            return 0
        def numpy(tensor):
            return tensor.detach().cpu().numpy().astype(float)
        def state():
            r, t = env._robot.data, env._target.data
            position = r.root_com_pos_w[0] if args.matched_executor else r.root_pos_w[0]
            return np.concatenate((numpy(position - t.root_pos_w[0]),
                                   numpy(r.root_lin_vel_w[0] - t.root_lin_vel_w[0]),
                                   numpy(r.root_quat_w[0]), numpy(r.root_ang_vel_b[0])))
        def set_state(asset, x, position_offset=None):
            root = asset.data.default_root_state.clone()
            offset = np.zeros(3) if position_offset is None else np.asarray(position_offset, dtype=float)
            root[:, :3] = torch.as_tensor(x[:3] + offset, dtype=root.dtype, device=root.device) + env.scene.env_origins
            root[:, 3:7] = torch.as_tensor(x[6:10], dtype=root.dtype, device=root.device)
            root[:, 7:10] = torch.as_tensor(x[3:6], dtype=root.dtype, device=root.device)
            root[:, 10:13] = torch.as_tensor(quaternion_to_rotation(x[6:10]) @ x[10:13], dtype=root.dtype, device=root.device)
            asset.write_root_state_to_sim(root)
            asset.reset()
        if args.start is not None:
            states[0, :3] = args.start
        target_position = np.asarray(args.target_position, dtype=float)
        target_state = np.array([*target_position, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
        # ``--start`` is the actor-root position relative to the target. The
        # measured matched state remains COM-relative, as in the existing
        # replay convention, after both actors are placed in world coordinates.
        set_state(env._robot, states[0], position_offset=target_position)
        set_state(env._target, target_state)
        env._success_streak.zero_()
        env._success_once.zero_()
        term = env.action_manager._terms[env._thrust_action_term_key]
        term.record_execution = args.executor_probe or args.matched_executor
        telemetry = []
        view = env._robot.root_physx_view
        com = numpy(view.get_coms())[0, :3]
        surface = None
        if args.surface_approach:
            from srb.nmpc.surface_approach import read_actor_mesh, largest_axis_face, face_goal, surface_sample
            actor_points, actor_faces, actor_paths = read_actor_mesh(env.sim.stage, '/World/envs/env_0/robot')
            target_points, target_faces, target_paths = read_actor_mesh(env.sim.stage, '/World/envs/env_0/target')
            top = largest_axis_face(actor_points, actor_faces, [0., 0., 1.])
            front = largest_axis_face(target_points, target_faces, [-1., 0., 0.])
            goal_position, goal_quat = face_goal([0.,0.,0.], [1.,0.,0.,0.], front, top, .30)
            cfg.target_offset_pos = tuple(goal_position)
            cfg.target_offset_quat = tuple(goal_quat)
            env._target_offset_pos[:] = torch.as_tensor(goal_position, dtype=torch.float32, device=env.device)
            env._target_offset_quat[:] = torch.as_tensor(goal_quat, dtype=torch.float32, device=env.device)
            params = replace(params, koz_radius_m=float(np.max(np.linalg.norm(target_points, axis=1))))
            surface = dict(actor_top=top, target_face=front, actor_points=actor_points, target_points=target_points)
            save_json(args.output / 'surface_geometry.json', {
                **surface, 'actor_mesh_paths': actor_paths, 'target_mesh_paths': target_paths,
                'actor_goal_position_m': goal_position, 'actor_goal_quaternion_wxyz': goal_quat,
                'target_position_m': target_position, 'target_quaternion_wxyz': [1.,0.,0.,0.],
                'initial_actor_relative_root_position_m': states[0, :3],
                'initial_actor_world_root_position_m': target_position + states[0, :3],
                'initial_actor_target_distance_m': float(np.linalg.norm(states[0, :3])),
                'front_definition': 'target -X bus face, panels extend along +/-Y; actor top is +Z',
                'desired_bus_face_gap_m': .30, 'minimum_mesh_plane_gap_m': .02,
                'koz_radius_m': params.koz_radius_m,
                'koz_definition': 'target full-mesh maximum vertex radius; additional full-mesh plane rejection guard',
                'guard_scope': 'all predicted nodes before accepting a prefix and all measured control samples; not continuous collision checking',
            })
            print('SURFACE GOAL ' + json.dumps(jsonable({'position': goal_position, 'quaternion': goal_quat,
                                                        'top': top, 'front': front, 'koz_radius': params.koz_radius_m})), flush=True)
            if args.surface_display_materials:
                from srb.nmpc.surface_approach import apply_display_materials
                apply_display_materials(env.sim.stage, actor_paths, target_paths, top, front)
        forces = numpy(-term.cfg.scale * term._thruster_power[:, None] * term._thruster_direction)
        offsets = numpy(term._thruster_offset)
        measured_B = np.vstack((forces.T, np.cross(offsets - com, forces).T))
        audit = {"initial_mass_kg": numpy(view.get_masses()), "initial_inertia": numpy(view.get_inertias()),
                 "com_m": com, "actual_rcs_matrix": measured_B,
                 "rcs_matrix_max_difference": np.max(np.abs(measured_B - rcs_matrix_from_task(params))),
                 "dry_mass_kg": numpy(term._dry_masses), "fuel_capacity_kg": term.cfg.fuel_capacity,
                 "fuel_consumption_rate": term.cfg.fuel_consumption_rate,
                 "mean_motion_plan_rad_s": params.mean_motion_rad_s,
                 "mean_motion_srb_rad_s": 0.0,
                 "physics_dt_s": env.physics_dt, "step_dt_s": env.step_dt,
                 "target_offset_pos": cfg.target_offset_pos, "target_offset_quat": cfg.target_offset_quat,
                 "target_world_position_m": target_position,
                 "initial_actor_relative_root_position_m": states[0, :3],
                 "initial_actor_world_root_position_m": target_position + states[0, :3],
                 "initial_actor_target_distance_m": float(np.linalg.norm(states[0, :3])),
                 "goal_aligned_for_this_run": args.align_goal, "physics_only": args.physics_only,
                 "env_device": str(env.device), "sim_device": str(env.sim.device),
                 "physx_mass_tensor_device": str(view.get_masses().device),
                 "physx_gpu_override": acquire_physx_interface().get_overwrite_gpu_setting()}
        save_json(args.output / "physics_audit.json", audit)
        if term.record_execution:
            from pxr import Usd, UsdPhysics, PhysxSchema
            body = next(p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/robot'))
                        if p.HasAPI(UsdPhysics.RigidBodyAPI))
            body_api = PhysxSchema.PhysxRigidBodyAPI(body)
            audit.update({'com_pose': numpy(view.get_coms()),
                          'gyroscopic': body_api.GetEnableGyroscopicForcesAttr().Get(),
                          'linear_damping': body_api.GetLinearDampingAttr().Get(),
                          'angular_damping': body_api.GetAngularDampingAttr().Get(),
                          'decimation': cfg.decimation, 'agent_rate': cfg.agent_rate,
                          'thruster_power': numpy(term._thruster_power)})
            save_json(args.output / 'physics_audit.json', audit)
        if not np.isclose(env.step_dt, params.control_period_s):
            raise RuntimeError("Simulation and command periods do not match")
        print("PHYSICS AUDIT " + json.dumps(jsonable(audit)), flush=True)
        if args.video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(args.output / "replay.mp4", fps=20, codec="libx264", quality=7, macro_block_size=2)
        expert = None
        executor_model = None
        matched_parameters = []
        predicted_mass_fuel = []
        acceleration_boundaries = []
        warm = None
        solve_times, executed, solver_records = [], [], []
        predictions_x, predictions_u, solve_steps, prediction_accepted = [], [], [], []
        solver_references = []
        video_camera_poses = []
        pending = None
        prefix_index = 0
        failure = None
        if args.closed_loop:
            from srb.nmpc import AcadosExpert
            from srb.nmpc.dynamics import reference_trajectory
            if args.native_goal:
                params = replace(params, target_point_T_m=np.asarray(cfg.target_offset_pos))
                # Native goal attitude is an explicit reference at every horizon node.
                params = replace(params, target_q_LT0=np.asarray(cfg.target_offset_quat))
                # target_q_LT0 also rotates the position; transform so the world goal stays correct.
                params = replace(params, target_point_T_m=quaternion_to_rotation(params.target_q_LT0).T @ np.asarray(cfg.target_offset_pos))
            if args.match_physics:
                params = replace(params, mass_kg=float(numpy(view.get_masses()).reshape(-1)[0]),
                                 inertia_B_kg_m2=numpy(view.get_inertias()).reshape(3, 3), com_B_m=com)
            if args.matched_executor:
                from srb.nmpc.matched_executor import MatchedExecutor, stationary_com_goal
                if cfg.decimation != 1 or audit['linear_damping'] != 0 or not audit['gyroscopic']:
                    raise RuntimeError('Matched model requires decimation=1, zero linear damping and enabled gyro')
                if term._num_thrusters_with_gimbal or term.external_acceleration_B is not None:
                    raise RuntimeError('Matched model requires fixed RCS and no external acceleration')
                executor_model = MatchedExecutor(float(numpy(term._dry_masses).ravel()[0]),
                    term.cfg.fuel_consumption_rate * cfg.agent_rate * numpy(term._thruster_power),
                    params.control_period_s, float(audit['angular_damping']))
                J = numpy(view.get_inertias()).reshape(3,3)
                matched_goal = stationary_com_goal(cfg.target_offset_pos,cfg.target_offset_quat,com)
                params = replace(params, mass_kg=float(numpy(view.get_masses()).ravel()[0]),
                    inertia_B_kg_m2=(J+J.T)/2, com_B_m=com, rcs_matrix_B=measured_B,
                    target_point_T_m=quaternion_to_rotation(matched_goal[6:10]).T@matched_goal[:3],
                    target_q_LT0=matched_goal[6:10],target_omega_TI_T_rad_s=np.zeros(3))
                audit['state_position'] = 'robot COM relative to stationary target actor origin'
                audit['force_accumulation_mass'] = 'before mass/inertia setters'
                audit['matched_discretization'] = 'PhysX semi-implicit COM translation, quaternion exponential, implicit gyro linearization'
                save_json(args.output / 'physics_audit.json', audit)
            expert = AcadosExpert(params, code_export_root=ROOT / 'third_party/acados/generated',
                                  executor_model=executor_model, jerk_weight=args.jerk_weight,
                                  jerk_scale_m_s3=args.jerk_scale)
            count = args.steps or len(duties)
            duties = np.zeros((count, 16))
            # Evaluate against the actual stationary SRB goal, independently of HCW reference rotation.
            goal = np.zeros(13)
            goal[:3] = cfg.target_offset_pos
            if args.matched_executor:
                goal = matched_goal.copy()
            goal[6:10] = cfg.target_offset_quat
            refs = np.repeat(goal[None, :], count + 1, axis=0)
            save_json(args.output / 'controller_params.json', asdict(params))
            save_json(args.output / 'smoothness_config.json', {
                'duty_rate_per_s': params.duty_rate_per_s, 'jerk_weight': args.jerk_weight,
                'jerk_scale_m_s3': args.jerk_scale, 'initial_acceleration_W': [0.,0.,0.],
                'jerk_definition': '(predicted interval acceleration - preceding interval acceleration)/dt',
                'boundary_source': 'last two actual executed COM velocities; zero before first ignition'})
        elif args.steps:
            duties = duties[:args.steps]
        measured = [state()]
        success_seen = False
        stopped = None
        max_contact_force = 0.0
        with torch.inference_mode():
            for k, duty in enumerate(duties):
                if args.executor_probe:
                    duty = np.zeros(16)
                    duty[(k // 3) % 16] = .2 if k % 3 != 2 else 0.
                if expert is not None:
                    if k % args.execute_steps == 0:
                        started = time.perf_counter()
                        if executor_model is not None:
                            J = numpy(view.get_inertias()).reshape(3,3)
                            params = replace(params, mass_kg=float(numpy(view.get_masses()).ravel()[0]),
                                             inertia_B_kg_m2=(J+J.T)/2)
                            executor_model.remaining_fuel = float(numpy(term.remaining_fuel).ravel()[0])
                            from srb.nmpc.physical import pack_physical_parameters
                            matched_parameters.append(pack_physical_parameters(params))
                        previous_acceleration = (measured[-1][3:6]-measured[-2][3:6])/params.control_period_s if k else np.zeros(3)
                        acceleration_boundaries.append(previous_acceleration.copy())
                        horizon_reference = np.repeat(goal[None,:],params.horizon_steps+1,axis=0) if args.matched_executor else None
                        if args.surface_stages:
                            from srb.nmpc.surface_approach import staged_reference
                            horizon_reference = staged_reference(
                                (k+np.arange(params.horizon_steps+1))*params.control_period_s,
                                states[0,:3], states[0,6:10], cfg.target_offset_pos, cfg.target_offset_quat,
                                com, *args.surface_stages)
                        if horizon_reference is not None:
                            solver_references.append(horizon_reference.copy())
                        pending = expert.solve(measured[-1], task_params=params, warm_start=warm,
                                               previous_acceleration_W=previous_acceleration,
                                               reference_states=horizon_reference,
                                               time_s=k*params.control_period_s if args.matched_executor else 0.)
                        if executor_model is not None and expert.predicted_mass_fuel is not None:
                            predicted_mass_fuel.append(expert.predicted_mass_fuel.copy())
                        solve_times.append(time.perf_counter() - started)
                        solve_steps.append(k)
                        predictions_x.append(pending.predicted_state.copy())
                        predictions_u.append(pending.predicted_duty.copy())
                        accepted, reason = pending.accepted, pending.failure_reason
                        margins = dict(pending.constraint_margins)
                        if surface is not None and accepted:
                            margins['surface_plane_gap_m'] = min(surface_sample(x, com, **surface)['mesh_plane_gap_m']
                                                                 for x in pending.predicted_state)
                            if margins['surface_plane_gap_m'] < .02:
                                accepted, reason = False, 'predicted full-mesh plane gap below 0.02 m'
                        prediction_accepted.append(accepted)
                        solver_records.append({'step': k, 'status': pending.status, 'accepted': accepted,
                                               'reason': reason, 'margins': margins})
                        if not accepted:
                            failure = solver_records[-1]
                            stopped = 'NMPC quality gate rejected command'
                            break
                        prefix_index = 0
                    proposed = pending.predicted_duty[prefix_index]
                    duty = np.clip(proposed, 0.0, 1.0)
                    if np.max(np.abs(duty - proposed)) > 1e-6:
                        raise RuntimeError('Duty correction exceeds numerical tolerance')
                    prefix_index += 1
                    warm = warm_after_prefix(pending, prefix_index, duty)
                executed.append(duty.copy())
                output = env.step(torch.as_tensor(duty[None, :], dtype=torch.float32, device=env.device))
                measured.append(state())
                if term.record_execution:
                    telemetry.append({key: numpy(value) for key, value in term.last_execution.items()})
                success_seen |= bool(env._success_once[0].item())
                contact_force = float(torch.linalg.vector_norm(env.scene["replay_contacts"].data.net_forces_w, dim=-1).max().item())
                max_contact_force = max(max_contact_force, contact_force)
                if writer is not None:
                    if surface is not None:
                        actor_pos = numpy(env._robot.data.root_pos_w[0])
                        target_pos = numpy(env._target.data.root_pos_w[0])
                        center = (actor_pos+target_pos)/2
                        distance = max(1.4, 1.2*np.linalg.norm(actor_pos-target_pos)+.6)
                        eye = center + np.array([-.2, -distance, .45*distance])
                        env.sim.set_camera_view(eye, center)
                        video_camera_poses.append(np.r_[eye, center])
                    writer.append_data(env.render())
                if k % 100 == 0:
                    print(f"replay step={k} position={measured[-1][:3]}", flush=True)
                if bool(output[2].any() or output[3].any()):
                    stopped = "episode ended or reset"
                    break
                if np.linalg.norm(measured[-1][:3]) <= params.koz_radius_m:
                    stopped = "measured center entered plan KOZ; replay stopped"
                    break
                if args.matched_executor and np.linalg.norm(numpy(env._robot.data.root_pos_w[0]-env._target.data.root_pos_w[0])) <= params.koz_radius_m:
                    stopped = 'measured actor origin entered KOZ'
                    break
                if contact_force > 1e-4:
                    stopped = "robot contact detected; replay stopped"
                    break
                if surface is not None and surface_sample(measured[-1], com, **surface)['mesh_plane_gap_m'] < .02:
                    stopped = 'measured full-mesh plane gap below 0.02 m'
                    break
        measured = np.asarray(measured)
        executed = np.asarray(executed).reshape(-1, 16)
        result = metrics(measured, executed, params, refs[:len(measured)])
        result.update({"native_srb_success_seen": success_seen, "stopped_reason": stopped,
                       "max_contact_force_N": max_contact_force,
                       "source_plan": str(args.plan), "open_loop": not args.closed_loop,
                       "failure": failure, "seed": args.seed, "match_physics": args.match_physics,
                       "matched_executor": args.matched_executor,
                       "state_position_reference": "COM" if args.matched_executor else "actor origin",
                       "initial_state": measured[0], "requested_steps": len(duties),
                       "horizon_steps": params.horizon_steps, "horizon_seconds": params.horizon_seconds,
                       "execute_steps": args.execute_steps, "solve_count": len(solve_times),
                       "jerk_weight": args.jerk_weight, "jerk_scale_m_s3": args.jerk_scale,
                       "action_block_lengths": params.action_block_lengths,
                       "solver_first_wall_s_including_build": solve_times[0] if solve_times else None,
                       "solver_hot_total_s": sum(solve_times[1:]),
                       "solver_hot_ms_per_executed_step": sum(solve_times[1:]) * 1000 / max(1, len(executed) - args.execute_steps),
                       "solver_replan_deadline_miss_fraction": np.mean(np.asarray(solve_times[1:]) > args.execute_steps * params.control_period_s) if len(solve_times)>1 else None,
                       "complete_success": len(executed) == len(duties) and stopped is None and failure is None and result['hover_hold_achieved'],
                       "native_goal": args.native_goal,
                       "solver_hot_p50_ms": np.percentile(solve_times[1:], 50)*1000 if len(solve_times)>1 else None,
                       "solver_hot_p99_ms": np.percentile(solve_times[1:], 99)*1000 if len(solve_times)>1 else None,
                       "solver_deadline_miss_fraction": np.mean(np.asarray(solve_times[1:]) > 0.05) if len(solve_times)>1 else None,
                       "all_planned_commands_executed": len(executed) == len(duties),
                       "max_position_deviation_from_plan_m": None if args.closed_loop else np.max(np.linalg.norm(measured[:, :3] - states[:len(measured), :3], axis=1)),
                       "final_mass_kg": numpy(view.get_masses()), "fuel_remaining_kg": numpy(term.remaining_fuel),
                       "goal_aligned_for_this_run": args.align_goal, "physics_only": args.physics_only})
        if surface is not None:
            samples = [surface_sample(x, com, **surface) for x in measured]
            series = {key: np.asarray([s[key] for s in samples]) for key in samples[0]}
            angles = 2*np.arccos(np.clip(np.abs((measured[:,6:10] / np.linalg.norm(measured[:,6:10],axis=1)[:,None]) @ goal[6:10]),0,1))
            good = ((np.abs(series['face_gap_m']-.30) < .02) & (series['lateral_error_m'] < .02)
                    & (angles < np.deg2rad(5)) & (np.linalg.norm(measured[:,3:6],axis=1) < .05)
                    & (np.linalg.norm(measured[:,10:13],axis=1) < .05))
            streak, longest = 0, 0
            for value in good:
                streak = streak + 1 if value else 0
                longest = max(longest, streak)
            surface_result = {
                'final': samples[-1], 'min_mesh_plane_gap_m': float(series['mesh_plane_gap_m'].min()),
                'final_hold_s': max(0,streak-1)*params.control_period_s,
                'longest_hold_s': max(0,longest-1)*params.control_period_s,
                'criteria': 'bus face gap 0.30 +/-0.02 m, lateral <0.02 m, full attitude <5 deg, speed <0.05 m/s, angular speed <0.05 rad/s for final >=1 s',
            }
            result['generic_hover_complete_success'] = result['complete_success']
            result['complete_success'] = (len(executed) == len(duties) and stopped is None and failure is None
                                           and surface_result['final_hold_s'] >= 1.
                                           and surface_result['min_mesh_plane_gap_m'] >= .02
                                           and max_contact_force <= 1e-4)
            result['surface_approach'] = surface_result
            result['surface_reference_stages_s'] = args.surface_stages
            np.savez_compressed(args.output / 'surface_metrics.npz', **series, success_sample=good)
        save_json(args.output / "replay.json", result)
        if telemetry:
            np.savez_compressed(args.output / 'executor.npz',
                                **{key: np.asarray([r[key] for r in telemetry]) for key in telemetry[0]})
        if matched_parameters:
            np.savez_compressed(args.output / 'matched_parameters.npz',
                physical=np.asarray(matched_parameters), predicted_mass_fuel=np.asarray(predicted_mass_fuel))
        np.savez_compressed(args.output / "replay.npz", states=measured, duties=executed, refs=refs[:len(measured)], solve_wall_s=solve_times)
        if args.closed_loop:
            np.savez_compressed(args.output / 'predictions.npz',
                                states=np.asarray(predictions_x), duties=np.asarray(predictions_u),
                                steps=np.asarray(solve_steps), accepted=np.asarray(prediction_accepted),
                                previous_acceleration_W=np.asarray(acceleration_boundaries))
            if solver_references:
                np.savez_compressed(args.output / 'solver_references.npz', states=np.asarray(solver_references), steps=np.asarray(solve_steps))
        save_json(args.output / 'solver_records.json', solver_records)
        if video_camera_poses:
            np.savez_compressed(args.output / 'video_camera_poses.npz', eye_lookat=np.asarray(video_camera_poses))
        if args.closed_loop:
            if len(executed):
                plot(args.output / 'comparison.png', measured, executed, params, refs[:len(measured)], label='SRB closed-loop')
        else:
            plot(args.output / "comparison.png", states[:len(duties)+1], duties, params, refs[:len(duties)+1], simulated=measured)
        print(json.dumps(jsonable(result), indent=2), flush=True)
        return 0
    except BaseException as exc:
        # Kit shutdown can terminate the interpreter before an exception is
        # displayed. Persist it before closing the app, including interrupts.
        import traceback
        save_json(args.output / 'runtime_error.json', {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", type=Path, default=ROOT / "configs/nmpc/default.yaml")
    p.add_argument("--start", nargs=3, type=float, default=[-3.0, 0.2, 0.15])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--free-floating", action="store_true", help="Explicitly set mean motion to zero for SRB integration")
    p.add_argument("--fov", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("replay")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--align-goal", action="store_true", help="Runtime-only SRB goal matching the saved NMPC hover goal")
    r.add_argument("--video", action="store_true")
    r.add_argument('--stable-video-lighting', action='store_true',
                   help='Disable sunlight/skydome randomization for readable evaluation video')
    r.add_argument('--video-fill-intensity', type=float, default=0.,
                   help='Optional pair of soft distant fill lights for readable video; rendering only')
    r.add_argument('--physics-only', action='store_true', help='Explicit CPU PhysX validation without waiting for an RTX viewport')
    r.add_argument('--closed-loop', action='store_true', help='Replan from measured PhysX state after each executed prefix')
    r.add_argument('--device', default='cuda:0', help='AppLauncher GPU; env tensor backend stays unchanged')
    r.add_argument('--steps', type=int)
    r.add_argument('--horizon-steps', type=int, choices=(20, 40),
                   help='Closed-loop prediction intervals (default: 40); open-loop preserves the saved plan')
    r.add_argument('--execute-steps', type=int,
                   help='Successive predicted duties before replanning (closed-loop default: 3)')
    r.add_argument('--seed', type=int, default=0)
    r.add_argument('--duty-rate', type=float, help='Override per-thruster duty slew limit [1/s]')
    r.add_argument('--jerk-weight', type=float, default=0., help='Optional normalized world COM jerk cost weight')
    r.add_argument('--jerk-scale', type=float, default=.1, help='Fixed jerk normalization [m/s^3]')
    r.add_argument('--start', nargs=3, type=float)
    r.add_argument('--target-position', nargs=3, type=float, default=[0., 0., 0.],
                   help='Target actor world position; controller state remains target-relative')
    r.add_argument('--match-physics', action='store_true', help='Use measured initial mass, inertia and COM in controller only')
    r.add_argument('--executor-probe', action='store_true', help='Small thruster pulses for executor identification; no NMPC')
    r.add_argument('--check-mass-reset', action='store_true', help='Run real-PhysX wet-mass reset assertions with two environments')
    r.add_argument('--matched-executor', action='store_true', help='Use audited free-floating executor model')
    r.add_argument('--surface-approach', action='store_true',
                   help='Derive actor +Z / target -X bus face goal from loaded geometry at 0.30 m gap')
    r.add_argument('--surface-stages', nargs=2, type=float, metavar=('TURN_S', 'APPROACH_S'),
                   help='Optional quintic turn-then-approach reference; default uses the final static goal directly')
    r.add_argument('--surface-display-materials', action='store_true',
                   help='Video-only readable colours and bus face highlights; does not alter geometry or physics')
    r.add_argument('--native-goal', action='store_true', help='Use native SRB goal in closed-loop controller')
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error('--steps must be positive')
    if args.mode == 'replay' and args.native_goal and (args.align_goal or not args.closed_loop):
        parser.error('--native-goal requires --closed-loop and excludes --align-goal')
    if args.mode == 'replay':
        if len(args.target_position) != 3 or not np.all(np.isfinite(args.target_position)):
            parser.error('--target-position must be three finite values')
        if args.surface_display_materials and not (args.surface_approach and args.video):
            parser.error('--surface-display-materials requires surface-approach and video')
        if not np.isfinite(args.video_fill_intensity) or args.video_fill_intensity < 0 or (args.video_fill_intensity and not args.video):
            parser.error('--video-fill-intensity must be finite/nonnegative and requires video')
        if args.surface_stages and (not args.surface_approach or any(not np.isfinite(s) or s <= 0 for s in args.surface_stages)):
            parser.error('--surface-stages requires surface-approach and two positive finite durations')
        if args.surface_approach and (not args.matched_executor or args.align_goal or args.native_goal or args.check_mass_reset):
            parser.error('--surface-approach requires matched-executor and excludes align-goal/native-goal/check-mass-reset')
        if args.stable_video_lighting and not args.video:
            parser.error('--stable-video-lighting requires --video')
        if args.duty_rate is not None and (not np.isfinite(args.duty_rate) or args.duty_rate <= 0 or not args.closed_loop):
            parser.error('--duty-rate must be positive/finite and requires closed-loop')
        if not np.isfinite(args.jerk_weight) or args.jerk_weight < 0:
            parser.error('--jerk-weight must be finite and nonnegative')
        if not np.isfinite(args.jerk_scale) or args.jerk_scale <= 0:
            parser.error('--jerk-scale must be finite and positive')
        if args.jerk_weight > 0 and not args.matched_executor:
            parser.error('--jerk-weight requires --matched-executor')
        if args.closed_loop and args.horizon_steps is None:
            args.horizon_steps = 40
        if args.execute_steps is None:
            args.execute_steps = 3 if args.closed_loop else 1
        if args.matched_executor and (not args.closed_loop or args.match_physics or args.native_goal):
            parser.error('--matched-executor requires closed-loop and excludes legacy match-physics/native-goal')
        if args.executor_probe and args.closed_loop:
            parser.error('--executor-probe excludes --closed-loop')
        if args.execute_steps < 1:
            parser.error('--execute-steps must be positive')
        if (args.horizon_steps is not None or args.execute_steps != 1) and not args.closed_loop:
            parser.error('horizon/prefix overrides require --closed-loop')
    return plan(args) if args.mode == "plan" else replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
