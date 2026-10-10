"""Opt-in synchronized observation capture for the orbital lighting trial.

All Kit imports are lazy. Rendering callers own the physics step; this module
never steps physics, submits forces, resets actors, or alters their materials.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from srb.nmpc.quaternion import quaternion_to_rotation
from .camera_quality import summarize_roi


def numpy(value):
    # CPU tensors and annotator arrays are reused in place. Recorded metadata
    # must own its memory, just as the recorded images do.
    return value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value).copy()


def refresh_camera(camera):
    """Isaac Lab 2.1 cache adapter; do not reset timestamps/frame counters.

    force_recompute alone only reads entries already flagged outdated. This
    adapter invalidates the image/pose cache after a render at unchanged time.
    Lab's frame counter counts buffer reads, not independent physical frames.
    """
    camera._is_outdated[:] = True
    camera.update(0.0, force_recompute=True)


def camera_mount_transform(stage, camera_schema):
    """Resolve the concrete schema and the moving rigid-body ancestor."""
    from pxr import UsdGeom, UsdPhysics
    prim = camera_schema.GetPrim() if hasattr(camera_schema, 'GetPrim') else camera_schema
    body = prim.GetParent()
    while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
        body = body.GetParent()
    if not body:
        raise RuntimeError(f'Camera {prim.GetPath()} has no rigid-body ancestor')
    matrix, _ = UsdGeom.XformCache().ComputeRelativeTransform(prim, body)
    return prim, body, matrix


def publish_initial_visual_poses(env):
    """Author the already-set physical poses without a physics/control step.

    Used only at initialization in the CPU/USD rendering path. Subsequent
    PhysX steps publish USD normally. No force, velocity or property is set.
    """
    from pxr import Usd, UsdGeom, UsdPhysics
    from .orbit_lighting import captured_world_transform
    cache = UsdGeom.XformCache()
    for root, asset in (('robot', env._robot), ('target', env._target)):
        body = next(p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath(f'/World/envs/env_0/{root}'))
                    if p.HasAPI(UsdPhysics.RigidBodyAPI))
        world = captured_world_transform(numpy(asset.data.root_pos_w[0]), numpy(asset.data.root_quat_w[0]))
        local = world*cache.GetLocalToWorldTransform(body.GetParent()).GetInverse()
        xf = UsdGeom.Xformable(body)
        # Keep the original op layout, which PhysX cached during initialization.
        from pxr import Gf
        ops = xf.GetOrderedXformOps()
        translation = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
        orientation = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeOrient)
        translation.Set(local.ExtractTranslation())
        q = local.ExtractRotationQuat()
        orientation.Set(Gf.Quatf(q) if str(orientation.GetAttr().GetTypeName()) == 'quatf' else q)


def semantic_mask(output, label):
    ids = np.asarray(output["data"])
    if ids.ndim == 3 and ids.shape[-1] == 1:
        ids = ids[..., 0]
    labels = output.get("info", {}).get("idToLabels", {})
    selected = [int(key) for key, value in labels.items()
                if label in (value.values() if isinstance(value, dict) else [value])]
    return np.isin(ids, selected)


def projected_mesh_bounds(points_w, position, quaternion, intrinsic, width, height):
    """Project actual mesh vertices in Isaac's +X-forward/+Z-up camera frame.

    This checks silhouette bounding-box containment, not occlusion or exact
    visible surface fraction. Behind-camera vertices explicitly fail.
    """
    p = (np.asarray(points_w)-position) @ quaternion_to_rotation(quaternion)
    front = p[:, 0] > 1e-8
    uv = np.full((len(p), 2), np.nan)
    k = np.asarray(intrinsic)
    uv[front, 0] = -k[0, 0]*p[front, 1]/p[front, 0]+k[0, 2]
    uv[front, 1] = -k[1, 1]*p[front, 2]/p[front, 0]+k[1, 2]
    if not front.any():
        return {"all_vertices_in_front": False, "complete_in_frame": False,
                "bbox_xyxy": None, "edge_margin_px": None}
    lo, hi = np.nanmin(uv, axis=0), np.nanmax(uv, axis=0)
    margin = float(min(*lo, width-1-hi[0], height-1-hi[1]))
    return {"all_vertices_in_front": bool(front.all()),
            "complete_in_frame": bool(front.all() and margin >= 0.),
            "bbox_xyxy": [*lo.tolist(), *hi.tolist()], "edge_margin_px": margin}


def write_json(path, data):
    def convert(value):
        if isinstance(value, (np.ndarray, np.generic)):
            return convert(value.tolist())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [convert(v) for v in value]
        return value
    Path(path).write_text(json.dumps(convert(data), indent=2, allow_nan=False)+"\n")


def capture_checks(frames):
    """Candidate nominal gates; missing objects never become successful samples.

    These are LDR engineering gates, not claims of real sensor calibration.
    Pose agreement establishes mounting synchronization, not pixel timing by
    itself; retain images, mesh projection, state and sensor times for audit.
    """
    present = bool(frames) and all(f["target_roi"]["pixel_count"] > 0 for f in frames)
    # Installed CPU USD publication skips tiny rotations: the measured first
    # post-step difference is 22 um / .019 deg (<.1 pixel at this resolution).
    synchronized = bool(frames) and all(f["camera_position_error_m"] < 5e-5
                                        and f["camera_orientation_error_deg"] < .05 for f in frames)
    return {
        "target_present_every_frame": present,
        "camera_pose_synchronized": synchronized,
        "target_mesh_complete_every_frame": bool(frames) and all(
            f["target_mesh_projection"]["complete_in_frame"] for f in frames),
        "nominal_target_usable": present and all(
            f["target_roi"]["underexposed_fraction"] < .95
            and f["target_roi"]["overexposed_fraction"] <= .05 for f in frames),
        "thresholds": {"dark_luma_u8": 5, "saturated_channel_u8": 250,
                       "maximum_target_dark_fraction": .95, "maximum_target_saturation_fraction": .05,
                       "maximum_camera_position_error_m": 5e-5, "maximum_camera_orientation_error_deg": .05},
    }


def gpu_preflight():
    """Check CUDA availability before starting an RTX app that might hang."""
    import ctypes
    import os
    result = {"cuda_init_status": None, "cuda_device_count": 0, "device_nodes": {}, "usable": False}
    for path in sorted(Path('/dev').glob('nvidia*')):
        if path.is_char_device():
            try:
                fd = os.open(path, os.O_RDWR)
                os.close(fd)
                result['device_nodes'][str(path)] = 'open OK'
            except OSError as exc:
                result['device_nodes'][str(path)] = {'errno': exc.errno, 'message': str(exc)}
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        result['cuda_init_status'] = int(cuda.cuInit(0))
        if result['cuda_init_status'] == 0:
            count = ctypes.c_int()
            status = cuda.cuDeviceGetCount(ctypes.byref(count))
            result['cuda_device_count'] = count.value if status == 0 else 0
        result['usable'] = result['cuda_device_count'] > 0
    except OSError as exc:
        result['library_error'] = str(exc)
    return result


def save_partial_execution(output, states, duties, refs, solve_times, solver_records, telemetry):
    """Persist execution evidence even when later rendering/analysis raises."""
    output = Path(output)
    states = np.asarray(states).reshape(-1, 13)
    commanded = np.asarray(duties).reshape(-1, 16)
    confirmed = min(len(commanded), max(0, len(states)-1))
    np.savez_compressed(output/'partial_replay.npz', states=states, duties=commanded[:confirmed],
                        commanded_duties=commanded, refs=np.asarray(refs)[:len(states)], solve_wall_s=solve_times)
    write_json(output/'partial_solver_records.json', solver_records)
    write_json(output/'partial_execution_status.json', {
        'confirmed_control_steps': confirmed, 'command_attempts': len(commanded),
        'scope': 'failure evidence, not a successful run; an interrupted env.step may be partially executed'})
    if telemetry:
        np.savez_compressed(output/'partial_executor.npz',
                            **{key: np.asarray([r[key] for r in telemetry]) for key in telemetry[0]})


class ReplayVisualCapture:
    def __init__(self, env, output, camera, dt):
        import omni.replicator.core as rep
        from isaacsim.core.utils.semantics import add_update_semantics
        from pxr import Usd, UsdGeom, UsdPhysics
        from srb.nmpc.surface_approach import read_actor_mesh

        self.env, self.output, self.camera, self.dt = env, Path(output), camera, dt
        self.frames = []
        self.rgb, self.depth, self.masks = [], [], []
        self.annotations = {}
        self.hdr_error = None
        for root, label in (("robot", "actor_satellite"), ("target", "target_satellite")):
            for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{root}")):
                if prim.IsA(UsdGeom.Mesh):
                    add_update_semantics(prim, label)
        body_paths = {}
        for name in ('robot', 'target'):
            body_paths[name] = str(next(p for p in Usd.PrimRange(env.sim.stage.GetPrimAtPath(f'/World/envs/env_0/{name}'))
                                       if p.HasAPI(UsdPhysics.RigidBodyAPI)).GetPath())
        self.target_points, _, target_paths = read_actor_mesh(env.sim.stage, body_paths['target'])
        self.actor_points, _, actor_paths = read_actor_mesh(env.sim.stage, body_paths['robot'])
        # Create/attach the historical external render product before warmup.
        env.render()
        products = {"onboard": camera.render_product_paths[0], "overview": env._render_product}
        for name, product in products.items():
            ann = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", {"colorize": False}, device="cpu")
            ann.attach(product)
            params = rep.AnnotatorRegistry.get_annotator("camera_params", device="cpu")
            params.attach(product)
            self.annotations[name] = (ann, params)
        try:
            self.hdr = rep.AnnotatorRegistry.get_annotator("HdrColor", device="cpu")
            self.hdr.attach(products["onboard"])
            self.direct_diffuse = rep.AnnotatorRegistry.get_annotator('DirectDiffuse', device='cpu')
            self.direct_diffuse.attach(products['onboard'])
            self.diffuse_albedo = rep.AnnotatorRegistry.get_annotator('DiffuseAlbedo', device='cpu')
            self.diffuse_albedo.attach(products['onboard'])
        except Exception as exc:
            self.hdr = None
            self.hdr_error = repr(exc)
        # cfg contains an environment regex; use the initialized concrete prim.
        camera_prim, camera_body, matrix = camera_mount_transform(env.sim.stage, camera._sensor_prims[0])
        self.mount_position = np.asarray(matrix.ExtractTranslation())
        q = matrix.ExtractRotationQuat()
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
        import torch
        self.mount_quaternion = numpy(convert_camera_frame_orientation_convention(
            torch.tensor([[q.GetReal(), *q.GetImaginary()]], dtype=torch.float32),
            origin="opengl", target="world"))[0]
        for folder in ("onboard", "overview", "paired", "static"):
            (self.output/folder).mkdir(parents=True, exist_ok=True)
        write_json(self.output/"capture_config.json", {
            "onboard_prim": str(camera_prim.GetPath()),
            "body_paths": body_paths,
            "onboard_mount_position_root_m": self.mount_position,
            "onboard_mount_quaternion_world_convention_wxyz": self.mount_quaternion,
            "onboard_resolution": [camera.cfg.width, camera.cfg.height],
            "overview_prim": env.cfg.viewer.cam_prim_path,
            "overview_eye": env.cfg.viewer.eye, "overview_lookat": env.cfg.viewer.lookat,
            "overview_resolution": env.cfg.viewer.resolution,
            "overview_source": "historical fixed NMPC replay viewer camera",
            "camera_quaternion_convention": "wxyz, world camera +X forward, +Z up",
            "image_source": "actual headless RTX render, no physics advancement during warmup",
            "mask_source": "semantic_segmentation class labels on actual mesh prims",
            "target_mesh_paths": target_paths, "actor_mesh_paths": actor_paths,
            "target_root_bounds_m": [self.target_points.min(0), self.target_points.max(0)],
            "actor_root_bounds_m": [self.actor_points.min(0), self.actor_points.max(0)],
            "hdr_setup_error": self.hdr_error,
            "fabric_enabled": env.sim.cfg.use_fabric,
            "fabric_force_update_available": hasattr(env.sim._fabric_iface, 'force_update'),
        })

    def settle(self, frames=8):
        for _ in range(frames):
            self.env.sim.render()
        refresh_camera(self.camera)

    def images(self):
        data = self.camera.data
        rgb = numpy(data.output["rgb"][0])[..., :3].copy()
        depth_key = "depth" if "depth" in data.output else "distance_to_image_plane"
        depth = numpy(data.output[depth_key][0]).copy()
        overview = np.asarray(self.env.render()).copy()
        return rgb, depth, overview

    def masks_for(self, view, shape):
        output = self.annotations[view][0].get_data()
        target, actor = semantic_mask(output, "target_satellite"), semantic_mask(output, "actor_satellite")
        if target.shape != shape:
            raise RuntimeError(f"segmentation buffer not ready: {target.shape} != {shape}")
        return target, actor

    def snapshot(self, name):
        import imageio.v2 as imageio
        from datetime import datetime, timezone
        rgb, depth, overview = self.images()
        imageio.imwrite(self.output/"static"/f"{name}_onboard.png", rgb)
        imageio.imwrite(self.output/"static"/f"{name}_overview.png", overview)
        np.save(self.output/"static"/f"{name}_depth.npy", depth)
        target, _ = self.masks_for("onboard", rgb.shape[:2])
        target_ext, actor_ext = self.masks_for("overview", overview.shape[:2])
        np.save(self.output/"static"/f"{name}_target_mask.npy", target)
        np.save(self.output/"static"/f"{name}_overview_target_mask.npy", target_ext)
        np.save(self.output/"static"/f"{name}_overview_actor_mask.npy", actor_ext)
        aov_metadata = {}
        if self.hdr is not None:
            for key in ('hdr', 'direct_diffuse', 'diffuse_albedo'):
                try:
                    values = np.asarray(getattr(self, key).get_data()).copy()
                    np.save(self.output/'static'/f'{name}_{key}.npy', values)
                    aov_metadata[key] = {'shape': list(values.shape), 'dtype': str(values.dtype)}
                except Exception as exc:
                    aov_metadata[key] = {'error': repr(exc)}
        data = self.camera.data
        target_points_w = self.target_points@quaternion_to_rotation(numpy(self.env._target.data.root_quat_w[0])).T+numpy(self.env._target.data.root_pos_w[0])
        from isaacsim.core.utils.xforms import get_world_pose
        from pxr import Usd, UsdGeom, UsdPhysics
        bodies = [p for root in ('robot', 'target')
                  for p in Usd.PrimRange(self.env.sim.stage.GetPrimAtPath(f'/World/envs/env_0/{root}'))
                  if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        render_poses = {}
        cache = UsdGeom.XformCache()
        for prim in [p for b in bodies for p in (b, self.env.sim.stage.GetPrimAtPath(str(b.GetPath())+'/cubesat_mesh'))]:
            matrix = cache.GetLocalToWorldTransform(prim)
            q = matrix.ExtractRotationQuat()
            render_poses[str(prim.GetPath())] = {
                'usd': [np.asarray(matrix.ExtractTranslation()), [q.GetReal(), *q.GetImaginary()]],
                'fabric': get_world_pose(str(prim.GetPath())),
            }
        write_json(self.output/"static"/f"{name}.json", {
            "time_s": 0.0, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "physics_time_s": float(self.env.sim.current_time),
            "sensor_time_s": float(numpy(self.camera._timestamp[0])),
            "sensor_buffer_read_count": int(numpy(self.camera.frame[0])),
            "camera_pos_w": numpy(data.pos_w[0]), "camera_quat_w": numpy(data.quat_w_world[0]),
            "intrinsic": numpy(data.intrinsic_matrices[0]),
            "target_mesh_projection": projected_mesh_bounds(target_points_w, numpy(data.pos_w[0]), numpy(data.quat_w_world[0]),
                                                              numpy(data.intrinsic_matrices[0]), rgb.shape[1], rgb.shape[0]),
            "aov_metadata": aov_metadata,
            "aov_scope": "HdrColor is exposure-sensitive in this installed pipeline; float storage does not establish absolute radiance. DirectDiffuse excludes specular response and may have DLSS internal resolution.",
            "robot_root_state": numpy(self.env._robot.data.root_state_w[0]),
            "target_root_state": numpy(self.env._target.data.root_state_w[0]),
            "onboard_target": summarize_roi(rgb, target),
            "overview_actor": summarize_roi(overview, actor_ext),
            "overview_target": summarize_roi(overview, target_ext),
            "onboard_camera_params": self.annotations["onboard"][1].get_data(),
            "overview_camera_params": self.annotations["overview"][1].get_data(),
            "render_pose_audit": render_poses,
        })

    def _frame_metadata(self, step, measured_state, rgb, overview):
        data = self.camera.data
        target_mask, _ = self.masks_for("onboard", rgb.shape[:2])
        target_ext, actor_ext = self.masks_for("overview", overview.shape[:2])
        r, t = self.env._robot.data, self.env._target.data
        pos, quat = numpy(data.pos_w[0]), numpy(data.quat_w_world[0])
        intrinsic = numpy(data.intrinsic_matrices[0])
        expected_pos = numpy(r.root_pos_w[0])+quaternion_to_rotation(numpy(r.root_quat_w[0]))@self.mount_position
        expected_rotation = quaternion_to_rotation(numpy(r.root_quat_w[0]))@quaternion_to_rotation(self.mount_quaternion)
        error_rotation = expected_rotation.T@quaternion_to_rotation(quat)
        points_w = self.target_points@quaternion_to_rotation(numpy(t.root_quat_w[0])).T+numpy(t.root_pos_w[0])
        frame = {
            "capture_id": step, "time_s": step*self.dt,
            "sensor_buffer_read_count": int(numpy(self.camera.frame[0])),
            "sensor_time_s": float(numpy(self.camera._timestamp[0])),
            "physics_time_s": float(self.env.sim.current_time),
            "measured_state13": numpy(measured_state),
            "robot_root_state_w": numpy(r.root_state_w[0]), "target_root_state_w": numpy(t.root_state_w[0]),
            "camera_pos_w": pos, "camera_quat_w": quat, "intrinsic": intrinsic,
            "expected_camera_pos_w": expected_pos,
            "camera_position_error_m": float(np.linalg.norm(pos-expected_pos)),
            "camera_orientation_error_deg": float(np.rad2deg(np.arccos(np.clip((np.trace(error_rotation)-1)/2, -1, 1)))),
            "target_roi": summarize_roi(rgb, target_mask),
            "overview_actor_roi": summarize_roi(overview, actor_ext),
            "overview_target_roi": summarize_roi(overview, target_ext),
            "target_mesh_projection": projected_mesh_bounds(points_w, pos, quat, intrinsic, rgb.shape[1], rgb.shape[0]),
        }
        return frame, target_mask

    def record(self, step, measured_state):
        import imageio.v2 as imageio
        refresh_camera(self.camera)
        rgb, depth, overview = self.images()
        # Save raw evidence before segmentation or geometric analysis can fail.
        imageio.imwrite(self.output/"onboard"/f"frame_{step:04d}.png", rgb)
        self.last_overview = (step, overview)
        try:
            frame, target_mask = self._frame_metadata(step, measured_state, rgb, overview)
        except Exception as exc:
            imageio.imwrite(self.output/"overview"/f"frame_{step:04d}.png", overview)
            np.save(self.output/'failed_capture_depth.npy', depth)
            write_json(self.output/'capture_error.json', {
                'capture_id': step, 'time_s': step*self.dt, 'error': repr(exc),
                'measured_state13': measured_state,
                'camera_pos_w': numpy(self.camera.data.pos_w[0]),
                'camera_quat_w': numpy(self.camera.data.quat_w_world[0]),
                'intrinsic': numpy(self.camera.data.intrinsic_matrices[0]),
                'robot_root_state_w': numpy(self.env._robot.data.root_state_w[0]),
                'target_root_state_w': numpy(self.env._target.data.root_state_w[0]),
            })
            raise
        self.rgb.append(rgb)
        self.depth.append(depth)
        self.masks.append(target_mask)
        self.frames.append(frame)
        # Overview keyframes each second, plus events. Always retain the last
        # rendered frame so finalize can save it after a solver rejection.
        roi = frame["target_roi"]
        event = (roi["pixel_count"] == 0 or (roi["overexposed_fraction"] or 0.) > .1
                 or (roi['underexposed_fraction'] or 0.) > .95
                 or not frame['target_mesh_projection']['complete_in_frame'])
        if step % 20 == 0 or event:
            self.save_overview(step, overview)
        write_json(self.output/"capture_progress.json", {"last_step": step, "last_frame": frame})

    def save_overview(self, step, overview):
        import imageio.v2 as imageio
        imageio.imwrite(self.output/"overview"/f"frame_{step:04d}.png", overview)
        write_json(self.output/"overview"/f"frame_{step:04d}.json", {
            "capture_id": step, "time_s": step*self.dt,
            "camera_params": self.annotations["overview"][1].get_data(),
        })

    def finalize(self):
        if not self.frames:
            return
        self.save_overview(*self.last_overview)
        np.savez_compressed(self.output/"observations.npz", rgb=np.asarray(self.rgb),
                            depth=np.asarray(self.depth), target_mask=np.asarray(self.masks),
                            time_s=np.array([f["time_s"] for f in self.frames]),
                            capture_id=np.array([f["capture_id"] for f in self.frames]),
                            intrinsic=np.array([f["intrinsic"] for f in self.frames]),
                            camera_pos_w=np.array([f["camera_pos_w"] for f in self.frames]),
                            camera_quat_w=np.array([f["camera_quat_w"] for f in self.frames]),
                            measured_state13=np.array([f["measured_state13"] for f in self.frames]),
                            robot_root_state_w=np.array([f["robot_root_state_w"] for f in self.frames]),
                            target_root_state_w=np.array([f["target_root_state_w"] for f in self.frames]))
        write_json(self.output/"capture_frames.json", self.frames)
        write_json(self.output/"capture_checks.json", capture_checks(self.frames))
