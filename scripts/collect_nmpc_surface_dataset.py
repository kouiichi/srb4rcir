#!/usr/bin/env python3
"""Collect matched-NMPC surface demonstrations and randomized video trials.

Each trial starts from a randomized target translation, approach distance,
small relative state error and one of the requested 0.50/0.30/0.10 m face
gaps. The replay process stores the measured closed-loop states and the
actually executed duty prefix in ``replay.npz``; predicted future duties are
kept separately and are not used as demonstration labels. Visual mode also
stores onboard RGB/depth frames in ``camera.npz``. With ``--physics-config``,
the current DESDynI/NEAR scene instead uses native ``trajectory.npz`` and
``onboard_capture/observations.npz`` with 12 executed RCS channels and a gimbal.
``--dataset-format lerobot`` automatically exports qualified episodes with
the official API in a separate LeRobot Python process.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srb.nmpc.surface_trials import generate_trial_specs, trial_to_json

DEFAULT_PLAN = ROOT / "logs/nmpc_validation_20260908/approach_hcw_repeat"
DEFAULT_ISAAC_PYTHON = "/home/ubuntu/isaac-sim-4.5/python.sh"
DEFAULT_LEROBOT_PYTHON = ROOT / ".venv/lerobot/bin/python"
DEFAULT_CAMERA_RESOLUTION = (256, 256)
CAMERA_STATE_ALIGNMENT = "camera[i] -> states[i]"


def _triple(values):
    # argparse treats negative scientific notation as an option for nargs=3.
    return [format(float(value), ".17f") for value in values]


def _capture_arguments(capture_mode, camera_resolution, fill_intensity=None):
    if capture_mode == "state":
        return ["--physics-only"]
    if capture_mode in ("video", "visual"):
        fill = fill_intensity if fill_intensity is not None else (600.0 if capture_mode == "video" else 1000)
        arguments = ["--video", "--stable-video-lighting", "--surface-display-materials",
                     "--video-fill-intensity", str(fill)]
        if capture_mode == "video":
            return arguments
        width, height = (int(value) for value in camera_resolution)
        return arguments + ["--record-camera", "--segmented-satellite-materials",
                            "--camera-resolution", str(width), str(height)]
    raise ValueError(f"unsupported capture mode: {capture_mode}")


def _gpu_preflight(kit_device):
    result = {"requested_kit_device": str(kit_device)}
    if not str(kit_device).startswith("cuda"):
        result.update({"usable": True, "reason": "non-CUDA Kit device requested"})
        return result
    from srb.visual.replay_capture import gpu_preflight

    result.update(gpu_preflight())
    try:
        device_index = int(str(kit_device).split(":", 1)[1])
    except (IndexError, ValueError):
        device_index = None
    if device_index is not None:
        available = device_index < int(result.get("cuda_device_count", 0))
        result["requested_device_index"] = device_index
        result["requested_device_available"] = available
        result["usable"] = bool(result.get("usable", False) and available)
    return result


def export_visual_dataset(folder, *, write_npz=True):
    """Check alignment and optionally export the combined NPZ artifact."""
    import numpy as np

    folder = Path(folder)
    with np.load(folder / "camera.npz", allow_pickle=False) as observations, \
            np.load(folder / "replay.npz", allow_pickle=False) as replay:
        times = observations["time_s"]
        frames = len(times)
        ids = np.arange(frames, dtype=int)
        aligned = (
            frames > 0 and frames == len(replay["states"])
            and len(replay["duties"]) == frames - 1
            and np.allclose(times, ids * .05, rtol=0., atol=1e-8)
        )
        if not aligned:
            raise ValueError("camera frames, measured states and executed actions are not aligned")
        rgb = observations["rgb"]
        if not np.any(np.max(rgb, axis=-1) > 5):
            raise ValueError("all recorded RGB pixels are black")
        if write_npz:
            np.savez_compressed(
                folder / "dataset.npz",
                rgb=rgb, target_rgb=rgb, target_visible=observations["target_visible"],
                depth=observations["depth"], time_s=times, state_index=ids,
                relative_state=replay["states"],
                self_state_w=observations["robot_state_w"],
                target_state_w=observations["target_state_w"],
                actions=replay["duties"], intrinsic=observations["intrinsic"],
                camera_pos_w=observations["camera_pos_w"], camera_quat_w=observations["camera_quat_w"],
            )
    schema = {
        "frame_count": frames, "aligned": True,
        "frame_alignment": "rgb[i], relative_state[i], self_state_w[i] share time_s[i]; initial frame is t=0",
        "action_alignment": "actions[i] acts from state[i] to state[i+1]; the final frame has no following action",
        "target_rgb": "full-resolution RGB observation; target_visible is a target-center visibility flag, not a pixel mask or crop",
        "relative_state": "[COM position relative to stationary target origin in world axes (m), relative COM velocity in world axes (m/s), actor world quaternion wxyz, actor body angular velocity (rad/s)]",
        "self_state_w": "[root position world (m), root quaternion wxyz, root linear velocity world (m/s), root angular velocity world (rad/s)]",
        "target_state_w": "same field order as self_state_w",
        "actions": "16 executed thruster duties in [0,1]",
        "rendering": "fixed Sun and Earth lighting, segmented satellite materials, synchronized CPU PhysX to USD to RTX",
    }
    if write_npz:
        (folder / "dataset_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return schema


def build_replay_command(
    trial,
    output,
    plan,
    *,
    steps,
    seed,
    isaac_python=DEFAULT_ISAAC_PYTHON,
    wrapper=ROOT / "srb_0.0.5_env.sh",
    sync_s=120.0,
    timeout_s=600,
    kit_device="cuda:1",
    capture_mode="state",
    camera_resolution=DEFAULT_CAMERA_RESOLUTION,
    fill_intensity=None,
    physics_config=None,
    camera_horizontal_fov=110.,
    camera_framing_margin=8.,
    stream_camera=False,
    stop_on_success=False,
    allow_partial_target=False,
    camera_readback_device='cpu',
    kit_worker_threads=None,
):
    """Build the exact subprocess argv for one state, video or visual trial."""
    if physics_config is not None:
        capture_args = (["--physics-only"] if capture_mode == "state" else ["--video"])
        if capture_mode == "visual":
            capture_args += ["--record-camera", "--camera-save-every", "1",
                             "--camera-readback-device", camera_readback_device]
            if stream_camera:
                capture_args.append("--stream-camera")
        scene_args = ["scripts/validate_nmpc_nasa_pair.py", "replay",
                      "--physics-config", str(physics_config),
                      "--reference-mode", "synchronous", "--sync-seconds", str(float(sync_s)),
                      "--camera-tracking", "limited-gimbal",
                      "--camera-resolution", *(str(int(x)) for x in camera_resolution),
                      "--camera-horizontal-fov", str(float(camera_horizontal_fov)),
                      "--camera-framing-margin", str(float(camera_framing_margin))]
        if stop_on_success:
            scene_args.append("--stop-on-success")
        if not allow_partial_target:
            scene_args.append("--camera-fov-gate")
        if kit_worker_threads is not None:
            scene_args += ["--kit-worker-threads", str(int(kit_worker_threads))]
    else:
        capture_args = _capture_arguments(capture_mode, camera_resolution, fill_intensity)
        scene_args = ["scripts/validate_nmpc_replay.py", "replay", "--plan", str(plan),
                      "--closed-loop", "--matched-executor", "--surface-approach",
                      "--surface-sync", str(float(sync_s))]
    return [
        "env", "-u", "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "timeout", "-k", "15", str(int(timeout_s)),
        str(wrapper), "--", str(isaac_python), "-u",
        *scene_args, "--output", str(output),
        *capture_args, "--device", str(kit_device),
        "--surface-gap", str(float(trial.goal_surface_gap_m)),
        "--horizon-steps", "40", "--execute-steps", "3", "--steps", str(int(steps)),
        "--seed", str(int(seed)), "--start", *_triple(trial.initial_relative_root_position_m),
        "--target-position", *_triple(trial.target_position_m),
        "--initial-linear-velocity", *_triple(trial.initial_linear_velocity_m_s),
        "--initial-attitude-rpy-deg", *_triple(trial.initial_attitude_rpy_deg),
        "--initial-angular-velocity", *_triple(trial.initial_angular_velocity_rad_s),
    ]


def _run(command, cwd, log_path, *, env=None):
    with log_path.open("w") as log:
        return subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def _lerobot_environment():
    env = os.environ.copy()
    # Isaac's Python/CUDA search paths belong only to the simulator process.
    for key in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        env.pop(key, None)
    return env


def build_export_command(output, python, repo_id):
    return [str(python), "-I", str(ROOT / "scripts/export_nmpc_lerobot_dataset.py"),
            "--input", str(Path(output).resolve()),
            "--output", str(Path(output).resolve() / "lerobot"), "--repo-id", repo_id]


def build_render_command(folder, output, *, isaac_python, wrapper, goal_gap_m):
    return [str(wrapper), "--", str(isaac_python), "-u",
            str(ROOT / "scripts/render_nmpc_surface_eval.py"), str(folder),
            "--output", str(output), "--goal-gap-m", str(float(goal_gap_m))]


def _nasa_trial_result(folder, spec, returncode, capture_mode, num_thrusters, dataset_format, allow_partial_target=False):
    """Qualify native NASA execution and synchronized camera artifacts."""
    from scripts.export_nmpc_lerobot_dataset import load_episode, save_episode_npz
    record = {**trial_to_json(spec), "folder": str(folder), "scene": "nasa_pair",
              "num_thrusters": num_thrusters, "capture_mode": capture_mode,
              "replay_returncode": returncode, "complete_success": False,
              "steps": 0, "requested_steps": None, "annotation_complete": None,
              "allow_partial_target": allow_partial_target}
    result_path = folder / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        record.update(result)
        record["steps"] = result["executed_steps"]
        record["max_contact_force_N"] = result.get("maximum_contact_force_N")
        record["surface_approach"] = {"final": result.get("final_plane_metrics"),
                                     "min_mesh_plane_gap_m": result.get("minimum_full_mesh_plane_gap_m")}
    elif (folder / "runtime_error.json").exists():
        record["runtime_error"] = json.loads((folder / "runtime_error.json").read_text())
        record["stopped_reason"] = record["runtime_error"].get("message")
    else:
        record["stopped_reason"] = "result.json was not produced"
    record["executed_steps"] = record["steps"]
    record["trajectory_length"] = record["steps"] + 1 if (folder / "trajectory.npz").exists() else 0
    record["trajectory_npz"] = str(folder / "trajectory.npz") if (folder / "trajectory.npz").exists() else None
    record["camera_npz"] = str(folder / "onboard_capture/observations.npz") if capture_mode == "visual" else None
    expected_end = record["steps"] == record["requested_steps"] or (
        record.get("termination") == "goal_reached" and record.get("complete_success")
        and record.get("final_fuel_kg", 0.) > 1e-6 and 0 < record["steps"] <= record["requested_steps"])
    record["collection_complete"] = bool(returncode == 0 and record["trajectory_npz"]
        and expected_end and record.get("execution_backend") == "CPU PhysX"
        and record.get("physics_contact_checked"))
    try:
        if capture_mode in ("state", "visual"):
            episode = load_episode(record, capture_mode)
            record["frame_count"] = len(episode["states"])
        if capture_mode == "visual":
            initial_check = json.loads((folder / "initial_state_check.json").read_text())
            record["visual_initialization_checked"] = bool(initial_check.get("accepted")
                and initial_check.get("stage") == "before_first_force"
                and initial_check.get("visual_setup_complete") and initial_check.get("direct_physx_checked"))
            record["collection_complete"] &= record["visual_initialization_checked"]
            checks = json.loads((folder / "onboard_capture/capture_checks.json").read_text())
            record["visual_checks"] = checks
            required = ["target_present_every_frame", "camera_pose_synchronized", "nominal_target_usable"]
            if not allow_partial_target:
                required.append("target_mesh_complete_every_frame")
            record["collection_complete"] &= all(checks.get(key, False) for key in required)
        elif capture_mode == "video":
            record["collection_complete"] &= (folder / "replay.mp4").exists()
        if (record["collection_complete"] and record["complete_success"]
                and capture_mode in ("state", "visual") and dataset_format == "npz"):
            save_episode_npz(episode, folder)
            record["dataset_npz"] = str(folder / "dataset.npz")
    except (OSError, ValueError, KeyError) as exc:
        record["collection_complete"] = False
        record["dataset_error"] = str(exc)
    return record


def _trial_result(folder, spec, returncode, capture_mode, *, dataset_format="npz", render_returncode=None,
                  scene="surface", num_thrusters=16, allow_partial_target=False):
    if scene == "nasa_pair":
        return _nasa_trial_result(folder, spec, returncode, capture_mode, num_thrusters, dataset_format, allow_partial_target)
    record = trial_to_json(spec)
    record.update({
        "folder": str(folder),
        "replay_returncode": returncode,
        "trajectory_npz": str(folder / "replay.npz") if (folder / "replay.npz").exists() else None,
        "capture_mode": capture_mode,
        "camera_npz": str(folder / "camera.npz") if (folder / "camera.npz").exists() else None,
        "camera_quality_json": str(folder / "camera_quality.json") if (folder / "camera_quality.json").exists() else None,
        "render_returncode": render_returncode,
        "raw_video": str(folder / "replay.mp4") if (folder / "replay.mp4").exists() else None,
        "annotated_video": str(folder / "surface_approach_eval.mp4") if (folder / "surface_approach_eval.mp4").exists() else None,
    })
    replay_path = folder / "replay.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        for key in (
            "complete_success", "steps", "requested_steps", "stopped_reason",
            "max_contact_force_N", "solver_hot_p50_ms", "solver_hot_p99_ms",
            "solver_replan_deadline_miss_fraction", "surface_approach",
        ):
            record[key] = replay.get(key)
        record["trajectory_length"] = int(replay.get("steps", 0)) + 1
    else:
        record.update({"complete_success": False, "steps": 0, "requested_steps": None,
                       "stopped_reason": "replay.json was not produced", "trajectory_length": 0})
    record["collection_complete"] = (
        returncode == 0 and record["trajectory_npz"] is not None
        and record["steps"] == record["requested_steps"]
    )
    record["executed_steps"] = record["steps"]
    if capture_mode == "video":
        record["collection_complete"] = bool(record["collection_complete"] and record["raw_video"])
    record["annotation_complete"] = (
        render_returncode == 0 and record["annotated_video"] is not None) if render_returncode is not None else None
    if capture_mode == "visual":
        try:
            schema = export_visual_dataset(folder, write_npz=dataset_format == "npz")
            checks = json.loads((folder / "camera_quality.json").read_text())["summary"]["checks"]
            if dataset_format == "npz":
                record["dataset_npz"] = str(folder / "dataset.npz")
            record["frame_count"] = schema["frame_count"]
            record["visual_checks"] = checks
            record["collection_complete"] = bool(record["collection_complete"] and all(
                checks.get(key, False) for key in
                ("target_center_visibility_ge_95_percent", "depth_present")))
        except (OSError, ValueError, KeyError) as exc:
            record["collection_complete"] = False
            record["dataset_error"] = str(exc)
    return record


def _write_report(path, config, trials):
    if config["capture_mode"] == "visual":
        mode_line = (
            f"- matched free-floating executor, GPU Kit `{config['kit_device']}`, CPU PhysX, "
            f"rendered camera mode at {config['camera_resolution']}px;"
        )
        alignment_line = "- camera frame `i` includes the initial frame at `t=0` and aligns with `states[i]`;"
    elif config["capture_mode"] == "video":
        mode_line = f"- matched free-floating executor, GPU Kit `{config['kit_device']}`, CPU PhysX, rendered overview video;"
        alignment_line = None
    else:
        mode_line = f"- matched free-floating executor, GPU Kit `{config['kit_device']}`, CPU PhysX, `--physics-only`;"
        alignment_line = None
    lines = [
        "# Matched-NMPC surface-approach dataset",
        "",
        f"The dataset stores measured closed-loop states and actually executed {config.get('num_thrusters', 16)}-channel duties.",
        "",
        "## Collection contract",
        "",
        mode_line,
        "- horizon `N=40`, execute prefix `K=3`, control period 0.05 s;",
        f"- {config.get('front_definition', 'actor +Z face approaches target -X face')};",
        "- goal gaps follow the recorded per-trial specification.",
        "",
        f"`{json.dumps(config, ensure_ascii=False, sort_keys=True)}`",
        "",
        "| Trial | Gap [m] | Initial distance [m] | Executed / requested | Complete | Replay code | Render code |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ]
    if alignment_line:
        lines.insert(8, alignment_line)
    for trial in trials:
        lines.append(
            f"| {trial['trial_index'] + 1} | {trial['goal_surface_gap_m']:.2f} | "
            f"{trial['initial_distance_m']:.3f} | {trial.get('steps', 0)}/{trial.get('requested_steps')} | "
            f"{trial.get('complete_success', False)} | {trial['replay_returncode']} | {trial.get('render_returncode')} |"
        )
    lines.extend([
        "", "| Trial | Final face gap [m] | Facing error [deg] | Min mesh plane gap [m] | Contact [N] | Solver P50/P99 [ms] | Deadline miss fraction |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ])
    for trial in trials:
        surface = trial.get("surface_approach") or {}
        final = surface.get("final") or {}
        lines.append(
            f"| {trial['trial_index'] + 1} | {final.get('face_gap_m')} | {final.get('facing_error_deg')} | "
            f"{surface.get('min_mesh_plane_gap_m')} | {trial.get('max_contact_force_N')} | "
            f"{trial.get('solver_hot_p50_ms')}/{trial.get('solver_hot_p99_ms')} | "
            f"{trial.get('solver_replan_deadline_miss_fraction')} |"
        )
    if config.get("dataset_format") == "lerobot":
        lines.extend([
            "",
            "LeRobot export uses qualified complete-success trials only; its return code is in `summary.json/lerobot_export`.",
            "`lerobot_export.log` records export failures; `lerobot/srb_manifest.json` maps episodes to raw trials and lists exclusions when export succeeds.",
        ])
    lines.extend([
        "",
        "A failed or early-stopped trial remains in the index with its raw log and artifacts; it is not relabeled as successful.",
        "Each trial stores its exact command, initial conditions and replay audit artifacts. Requested annotations use `render_command.txt`, `render.log`, `surface_approach_eval.mp4` and `surface_summary.png`.",
        "",
    ])
    path.write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New dataset directory; it must not already exist")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--physics-config", type=Path,
                        help="Use the current DESDynI/NEAR 12-channel scene with this explicit physics JSON")
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--trial-indices", nargs="+", type=int, help="Select zero-based trials from the seeded --count specifications without resampling")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent native NASA subprocesses; each has its own scene, solver build and output")
    parser.add_argument("--steps", type=int, default=2660)
    parser.add_argument("--stop-on-success", action="store_true", help="NASA trials: stop after the strict one-second goal hold with fuel remaining")
    parser.add_argument("--sync-seconds", type=float, default=120.0,
                        help="simultaneous surface-sync position/attitude reference duration")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--target-position-range", nargs=2, type=float, default=(-2.0, 2.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--distance-range", nargs=2, type=float, default=(14.0, 16.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--lateral-range", nargs=2, type=float, default=(-2.0, 2.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--linear-velocity-range", nargs=2, type=float, default=(-.02, .02), metavar=("LOW", "HIGH"))
    parser.add_argument("--attitude-range-deg", nargs=2, type=float, default=(-5.0, 5.0), metavar=("LOW", "HIGH"))
    parser.add_argument("--angular-velocity-range", nargs=2, type=float, default=(-.005, .005), metavar=("LOW", "HIGH"))
    parser.add_argument("--surface-gaps", nargs="+", type=float, choices=(.1, .3, .5), default=(.5, .3, .1))
    parser.add_argument("--isaac-python", default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--kit-device", default="cuda:1",
                        help="Isaac Kit device passed to replay (for example cuda:1 or cuda:2)")
    parser.add_argument("--kit-worker-threads", type=int, help="NASA Carbonite/TBB/physics worker limit per subprocess")
    parser.add_argument("--capture-mode", choices=("state", "video", "visual"), default="state",
                        help="state: physics only; video: overview recording; visual: overview plus onboard RGB/depth")
    parser.add_argument("--annotate-video", action="store_true", help="Render measured telemetry and plots after each video/visual trial")
    parser.add_argument("--fill-intensity", type=float, default=None,
                        help="Video fill light intensity; defaults to 600 for video, 1000 for visual")
    parser.add_argument("--camera-resolution", nargs=2, type=int, default=DEFAULT_CAMERA_RESOLUTION,
                        metavar=("WIDTH", "HEIGHT"), help="onboard camera resolution in visual mode")
    parser.add_argument("--camera-horizontal-fov", type=float, default=110., help="NASA onboard horizontal FOV in degrees")
    parser.add_argument("--camera-framing-margin", type=float, default=8., help="NASA full-target image margin in pixels")
    parser.add_argument("--stream-camera", action="store_true", help="NASA visual mode: save RGB/depth/masks per frame with a numeric NPZ index")
    parser.add_argument("--allow-partial-target", action="store_true", help="NASA trials: allow near-field target cropping; presence, exposure and synchronization checks remain required")
    parser.add_argument("--camera-readback-device", choices=("cpu", "cuda"), default="cpu", help="NASA Replicator transfer buffers; environment and camera tensors remain CPU")
    parser.add_argument("--dataset-format", choices=("npz", "lerobot"), default="npz",
                        help="lerobot exports qualified trials to OUTPUT/lerobot; raw NPZ files are retained")
    parser.add_argument("--lerobot-python", type=Path, default=DEFAULT_LEROBOT_PYTHON,
                        help="Python >=3.12 with LeRobot dataset dependencies, separate from Isaac")
    parser.add_argument("--lerobot-repo-id", default="srb/rendezvous_nmpc",
                        help="Local dataset identifier; no Hub upload is performed")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.count <= 0 or args.steps <= 0:
        parser.error("--count and --steps must be positive")
    if args.workers <= 0 or (args.workers > 1 and args.physics_config is None):
        parser.error("--workers must be positive; parallel collection requires NASA --physics-config")
    if args.kit_worker_threads is not None and (args.kit_worker_threads <= 0 or args.physics_config is None):
        parser.error("--kit-worker-threads must be positive and requires NASA --physics-config")
    if args.trial_indices is not None and (len(set(args.trial_indices)) != len(args.trial_indices)
            or any(i < 0 or i >= args.count for i in args.trial_indices)):
        parser.error("--trial-indices must be unique values in [0, --count)")
    if any(value <= 0 for value in (args.sync_seconds, args.timeout_seconds)):
        parser.error("sync duration and timeout must be positive")
    if any(value <= 0 for value in args.camera_resolution):
        parser.error("camera resolution values must be positive")
    if args.annotate_video and args.capture_mode == "state":
        parser.error("--annotate-video requires --capture-mode video or visual")
    if args.physics_config is not None and args.annotate_video:
        parser.error("NASA trials use native onboard and overview videos; surface annotations are unavailable")
    if args.stream_camera and (args.physics_config is None or args.capture_mode != "visual"):
        parser.error("--stream-camera requires NASA --physics-config and --capture-mode visual")
    if args.stop_on_success and args.physics_config is None:
        parser.error("--stop-on-success requires NASA --physics-config")
    if (args.allow_partial_target or args.camera_readback_device != 'cpu') and args.physics_config is None:
        parser.error("partial-target and readback options require NASA --physics-config")
    if not 0 < args.camera_horizontal_fov < 180 or not 0 <= args.camera_framing_margin < float("inf"):
        parser.error("camera FOV must be within (0,180); margin must be finite and nonnegative")
    if args.dataset_format == "lerobot" and args.capture_mode == "video":
        parser.error("LeRobot export requires --capture-mode state or visual")
    if args.fill_intensity is None:
        args.fill_intensity = 600.0 if args.capture_mode == "video" else 1000
    if not (0 <= args.fill_intensity < float("inf")):
        parser.error("fill intensity must be finite and nonnegative")
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    if args.physics_config is None and not args.plan.exists():
        raise FileNotFoundError(args.plan)
    physics = None
    if args.physics_config is not None:
        from srb.nmpc.nasa_assets import load_physics_config
        physics = load_physics_config(args.physics_config)
        if len(physics["desdyni"]["thruster_power_N"]) != 12:
            parser.error("NASA dataset collection requires the current 12-channel configuration")

    specs = generate_trial_specs(
        args.seed,
        args.count,
        target_position_bounds_m=tuple(args.target_position_range),
        distance_bounds_m=tuple(args.distance_range),
        lateral_offset_bounds_m=tuple(args.lateral_range),
        linear_velocity_bounds_m_s=tuple(args.linear_velocity_range),
        attitude_rpy_bounds_deg=tuple(args.attitude_range_deg),
        angular_velocity_bounds_rad_s=tuple(args.angular_velocity_range),
        gap_values_m=tuple(args.surface_gaps),
    )
    if physics is not None:
        specs = [replace(spec, target_quaternion_wxyz=(math.sqrt(.5), 0., -math.sqrt(.5), 0.),
                         front_definition="DESDynI -Z bottom rim approaches NEAR +Z solar-array plane") for spec in specs]
    if args.trial_indices is not None:
        specs = [spec for spec in specs if spec.trial_index in args.trial_indices]
    config = {
        "seed": args.seed, "count": len(specs), "steps": args.steps,
        "generated_trial_count": args.count, "selected_trial_indices": [s.trial_index for s in specs],
        "workers": args.workers,
        "kit_worker_threads": args.kit_worker_threads,
        "target_position_bounds_m": list(args.target_position_range),
        "initial_distance_bounds_m": list(args.distance_range),
        "lateral_offset_bounds_m": list(args.lateral_range),
        "initial_linear_velocity_bounds_m_s": list(args.linear_velocity_range),
        "initial_attitude_rpy_bounds_deg": list(args.attitude_range_deg),
        "initial_angular_velocity_bounds_rad_s": list(args.angular_velocity_range),
        "surface_gaps_m": list(args.surface_gaps),
        "target_quaternion_wxyz": list(specs[0].target_quaternion_wxyz),
        "front_definition": specs[0].front_definition,
        "surface_sync_seconds": args.sync_seconds,
        "kit_device": args.kit_device,
        "capture_mode": args.capture_mode,
        "annotate_video": args.annotate_video,
        "fill_intensity": args.fill_intensity if args.capture_mode != "state" and physics is None else None,
        "camera_resolution": list(args.camera_resolution) if args.capture_mode == "visual" else None,
        "camera_state_alignment": CAMERA_STATE_ALIGNMENT if args.capture_mode == "visual" else None,
        "horizon_steps": 40, "execute_steps": 3, "control_period_s": .05,
        "plan": str(args.plan), "cwd": str(ROOT),
        "dataset_format": args.dataset_format,
        "scene": "nasa_pair" if physics is not None else "surface",
        "num_thrusters": 12 if physics is not None else 16,
        "physics_config": str(args.physics_config.resolve()) if physics is not None else None,
        "camera_horizontal_fov_deg": args.camera_horizontal_fov if physics is not None else None,
        "camera_framing_margin_px": args.camera_framing_margin if physics is not None else None,
        "stream_camera": args.stream_camera,
        "stop_on_success": args.stop_on_success,
        "allow_partial_target": args.allow_partial_target,
        "camera_readback_device": args.camera_readback_device,
    }
    if physics is not None:
        config["plan"] = None
        config["physics_model"] = physics
    export_command = None
    if args.dataset_format == "lerobot":
        export_argv = build_export_command(args.output, args.lerobot_python, args.lerobot_repo_id)
        export_command = {"argv": export_argv, "command": shlex.join(export_argv)}
        config["lerobot_export_command"] = export_command["command"]
    commands = []
    for spec in specs:
        folder = args.output / f"trial_{spec.trial_index + 1:03d}"
        argv = build_replay_command(
            spec, folder, args.plan, steps=args.steps, seed=args.seed + spec.trial_index,
            isaac_python=args.isaac_python, wrapper=ROOT / "srb_0.0.5_env.sh",
            sync_s=args.sync_seconds,
            timeout_s=args.timeout_seconds, kit_device=args.kit_device,
            capture_mode=args.capture_mode, camera_resolution=args.camera_resolution,
            fill_intensity=args.fill_intensity,
            physics_config=args.physics_config, camera_horizontal_fov=args.camera_horizontal_fov,
            camera_framing_margin=args.camera_framing_margin,
            stream_camera=args.stream_camera,
            stop_on_success=args.stop_on_success,
            allow_partial_target=args.allow_partial_target,
            camera_readback_device=args.camera_readback_device,
            kit_worker_threads=args.kit_worker_threads,
        )
        commands.append({"trial_index": spec.trial_index, "folder": str(folder),
                         "argv": argv, "command": shlex.join(argv)})
    if args.dry_run:
        print(json.dumps({"config": config, "commands": commands, "export_command": export_command}, indent=2))
        return 0

    if export_command is not None:
        check = subprocess.run(export_command["argv"][:3] + ["--check"], cwd=ROOT,
                               env=_lerobot_environment(), capture_output=True, text=True, timeout=60)
        if check.returncode:
            raise RuntimeError("LeRobot export preflight failed before simulation:\n" + check.stdout + check.stderr)

    args.output.mkdir(parents=True, exist_ok=False)
    invocation = {
        "cwd": str(ROOT), "argv": sys.argv, "command": shlex.join(sys.argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    }
    config["runner_command"] = invocation["command"]
    (args.output / "runner_invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    (args.output / "experiment_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "trial_plan.json").write_text(json.dumps({
        "trials": [trial_to_json(spec) for spec in specs], "commands": commands}, indent=2) + "\n")

    preflight = _gpu_preflight(args.kit_device)
    (args.output / "gpu_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if str(args.kit_device).startswith("cuda") and not preflight.get("usable", False):
        error = {
            "type": "GPUAccessUnavailable",
            "message": "Requested Kit CUDA device is unavailable; no replay trial was started.",
            "requested_kit_device": args.kit_device,
            "gpu_preflight": preflight,
        }
        (args.output / "runtime_error.json").write_text(json.dumps(error, indent=2) + "\n")
        summary = {"experiment": str(args.output), "config": config,
                   "successful_trials": 0, "attempted_trials": 0, "trials": [],
                   "failure": error}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        _write_report(args.output / "report.md", config, [])
        print(json.dumps(error, indent=2), flush=True)
        return 1

    def collect_trial(pair):
        spec, command = pair
        folder = Path(command["folder"])
        staging_log = args.output / f"trial_{spec.trial_index + 1:03d}.replay.log"
        started = time.perf_counter()
        returncode = _run(command["argv"], ROOT, staging_log)
        replay_wall = time.perf_counter() - started
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "trial_spec.json").write_text(json.dumps(trial_to_json(spec), indent=2) + "\n")
        (folder / "command.txt").write_text(command["command"] + "\n")
        staging_log.replace(folder / "replay.log")
        render_returncode = None
        if args.annotate_video:
            render_returncode = 1
            if (folder / "replay.json").exists() and (folder / "replay.mp4").exists():
                render_command = build_render_command(
                    folder, folder / "surface_approach_eval.mp4",
                    isaac_python=args.isaac_python, wrapper=ROOT / "srb_0.0.5_env.sh",
                    goal_gap_m=spec.goal_surface_gap_m)
                (folder / "render_command.txt").write_text(shlex.join(render_command) + "\n")
                render_returncode = _run(render_command, ROOT, folder / "render.log")
        result = _trial_result(folder, spec, returncode, args.capture_mode,
                               dataset_format=args.dataset_format, render_returncode=render_returncode,
                               scene=config["scene"], num_thrusters=config["num_thrusters"],
                               allow_partial_target=args.allow_partial_target)
        result["replay_wall_s"] = replay_wall
        result["replay_wall_s_per_control_step"] = replay_wall / result["steps"] if result["steps"] else None
        return result

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(collect_trial, zip(specs, commands)):
            results.append(result)
            (args.output / "trials.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)

    summary = {
        "experiment": str(args.output), "config": config,
        "successful_trials": sum(bool(item.get("complete_success")) for item in results),
        "complete_collections": sum(bool(item.get("collection_complete")) for item in results),
        "attempted_trials": len(results), "trials": results,
        "complete_annotations": sum(item.get("annotation_complete") is True for item in results),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    export_returncode = 0
    if export_command is not None:
        export_returncode = _run(export_command["argv"], ROOT, args.output / "lerobot_export.log",
                                 env=_lerobot_environment())
        summary["lerobot_export"] = {
            "returncode": export_returncode, "root": str(args.output.resolve() / "lerobot"),
            "log": str(args.output.resolve() / "lerobot_export.log"),
            "command": export_command["command"],
        }
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(args.output / "report.md", config, results)
    print(json.dumps({"successful_trials": summary["successful_trials"],
                      "attempted_trials": summary["attempted_trials"],
                      "report": str(args.output / "report.md")}, indent=2), flush=True)
    return 0 if (export_returncode == 0 and len(results) == len(specs)
                 and all(item["collection_complete"] and item.get("complete_success", False) for item in results)
                 and (not args.annotate_video or all(item["annotation_complete"] for item in results))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
