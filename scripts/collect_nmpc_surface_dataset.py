#!/usr/bin/env python3
"""Collect reproducible matched-NMPC surface-approach demonstrations.

Each trial starts from a randomized target translation, approach distance,
small relative state error and one of the requested 0.50/0.30/0.10 m face
gaps. The replay process stores the measured closed-loop states and the
actually executed duty prefix in ``replay.npz``; predicted future duties are
kept separately and are not used as demonstration labels. Visual mode also
stores onboard RGB/depth frames in ``camera.npz``.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from srb.nmpc.surface_trials import generate_trial_specs, trial_to_json

DEFAULT_PLAN = ROOT / "logs/nmpc_validation_20260908/approach_hcw_repeat"
DEFAULT_ISAAC_PYTHON = "/home/ubuntu/isaac-sim-4.5/python.sh"
DEFAULT_CAMERA_RESOLUTION = (256, 256)
CAMERA_STATE_ALIGNMENT = "camera[i] -> states[i]"


def _triple(values):
    return [str(float(value)) for value in values]


def _capture_arguments(capture_mode, camera_resolution):
    if capture_mode == "state":
        return ["--physics-only"]
    if capture_mode == "visual":
        width, height = (int(value) for value in camera_resolution)
        return ["--video", "--record-camera", "--stable-video-lighting",
                "--surface-display-materials", "--segmented-satellite-materials",
                "--video-fill-intensity", "1000",
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


def export_visual_dataset(folder):
    """Export measured observations without interpolating frames or actions."""
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
    turn_s=12.0,
    approach_s=120.0,
    timeout_s=600,
    kit_device="cuda:1",
    capture_mode="state",
    camera_resolution=DEFAULT_CAMERA_RESOLUTION,
):
    """Build the exact subprocess argv for one state or visual trial."""
    capture_args = _capture_arguments(capture_mode, camera_resolution)
    return [
        "env", "-u", "CUDA_VISIBLE_DEVICES",
        "CUDA_DEVICE_ORDER=PCI_BUS_ID", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "timeout", "-k", "15", str(int(timeout_s)),
        str(wrapper), "--", str(isaac_python), "-u",
        "scripts/validate_nmpc_replay.py", "replay",
        "--plan", str(plan), "--output", str(output),
        *capture_args, "--device", str(kit_device), "--closed-loop",
        "--matched-executor", "--surface-approach",
        "--surface-stages", str(float(turn_s)), str(float(approach_s)),
        "--surface-gap", str(float(trial.goal_surface_gap_m)),
        "--horizon-steps", "40", "--execute-steps", "3", "--steps", str(int(steps)),
        "--seed", str(int(seed)), "--start", *_triple(trial.initial_relative_root_position_m),
        "--target-position", *_triple(trial.target_position_m),
        "--initial-linear-velocity", *_triple(trial.initial_linear_velocity_m_s),
        "--initial-attitude-rpy-deg", *_triple(trial.initial_attitude_rpy_deg),
        "--initial-angular-velocity", *_triple(trial.initial_angular_velocity_rad_s),
    ]


def _run(command, cwd, log_path):
    with log_path.open("w") as log:
        return subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def _trial_result(folder, spec, returncode, capture_mode):
    record = trial_to_json(spec)
    record.update({
        "folder": str(folder),
        "replay_returncode": returncode,
        "trajectory_npz": str(folder / "replay.npz") if (folder / "replay.npz").exists() else None,
        "capture_mode": capture_mode,
        "camera_npz": str(folder / "camera.npz") if (folder / "camera.npz").exists() else None,
        "camera_quality_json": str(folder / "camera_quality.json") if (folder / "camera_quality.json").exists() else None,
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
    if capture_mode == "visual":
        try:
            schema = export_visual_dataset(folder)
            checks = json.loads((folder / "camera_quality.json").read_text())["summary"]["checks"]
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
    else:
        mode_line = f"- matched free-floating executor, GPU Kit `{config['kit_device']}`, CPU PhysX, `--physics-only`;"
        alignment_line = None
    lines = [
        "# Matched-NMPC surface-approach dataset",
        "",
        "The dataset stores measured closed-loop states and actually executed 16-channel duty prefixes.",
        "",
        "## Collection contract",
        "",
        mode_line,
        "- horizon `N=40`, execute prefix `K=3`, control period 0.05 s;",
        "- actor +Z face approaches target -X face;",
        "- requested surface gaps are balanced in the configured sequence.",
        "",
        f"`{json.dumps(config, ensure_ascii=False, sort_keys=True)}`",
        "",
        "| Trial | Gap [m] | Initial distance [m] | Samples | Complete | Return code |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    if alignment_line:
        lines.insert(8, alignment_line)
    for trial in trials:
        lines.append(
            f"| {trial['trial_index'] + 1} | {trial['goal_surface_gap_m']:.2f} | "
            f"{trial['initial_distance_m']:.3f} | {trial.get('trajectory_length', 0)} | "
            f"{trial.get('complete_success', False)} | {trial['replay_returncode']} |"
        )
    lines.extend([
        "",
        "A failed or early-stopped trial remains in the index with its raw log and artifacts; it is not relabeled as successful.",
        "",
    ])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New dataset directory; it must not already exist")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--turn-seconds", type=float, default=12.0)
    parser.add_argument("--approach-seconds", type=float, default=120.0)
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
    parser.add_argument("--capture-mode", choices=("state", "visual"), default="state",
                        help="state stores dynamics only; visual stores onboard RGB/depth frames too")
    parser.add_argument("--camera-resolution", nargs=2, type=int, default=DEFAULT_CAMERA_RESOLUTION,
                        metavar=("WIDTH", "HEIGHT"), help="onboard camera resolution in visual mode")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.count <= 0 or args.steps <= 0:
        parser.error("--count and --steps must be positive")
    if any(value <= 0 for value in (args.turn_seconds, args.approach_seconds, args.timeout_seconds)):
        parser.error("stage durations and timeout must be positive")
    if any(value <= 0 for value in args.camera_resolution):
        parser.error("camera resolution values must be positive")
    if args.output.exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    if not args.plan.exists():
        raise FileNotFoundError(args.plan)

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
    config = {
        "seed": args.seed, "count": args.count, "steps": args.steps,
        "target_position_bounds_m": list(args.target_position_range),
        "initial_distance_bounds_m": list(args.distance_range),
        "lateral_offset_bounds_m": list(args.lateral_range),
        "initial_linear_velocity_bounds_m_s": list(args.linear_velocity_range),
        "initial_attitude_rpy_bounds_deg": list(args.attitude_range_deg),
        "initial_angular_velocity_bounds_rad_s": list(args.angular_velocity_range),
        "surface_gaps_m": list(args.surface_gaps),
        "target_quaternion_wxyz": [1., 0., 0., 0.],
        "turn_seconds": args.turn_seconds, "approach_seconds": args.approach_seconds,
        "kit_device": args.kit_device,
        "capture_mode": args.capture_mode,
        "camera_resolution": list(args.camera_resolution) if args.capture_mode == "visual" else None,
        "camera_state_alignment": CAMERA_STATE_ALIGNMENT if args.capture_mode == "visual" else None,
        "horizon_steps": 40, "execute_steps": 3, "control_period_s": .05,
        "plan": str(args.plan), "cwd": str(ROOT),
    }
    commands = []
    for spec in specs:
        folder = args.output / f"trial_{spec.trial_index + 1:03d}"
        argv = build_replay_command(
            spec, folder, args.plan, steps=args.steps, seed=args.seed + spec.trial_index,
            isaac_python=args.isaac_python, wrapper=ROOT / "srb_0.0.5_env.sh",
            turn_s=args.turn_seconds, approach_s=args.approach_seconds,
            timeout_s=args.timeout_seconds, kit_device=args.kit_device,
            capture_mode=args.capture_mode, camera_resolution=args.camera_resolution,
        )
        commands.append({"trial_index": spec.trial_index, "folder": str(folder),
                         "argv": argv, "command": shlex.join(argv)})
    if args.dry_run:
        print(json.dumps({"config": config, "commands": commands}, indent=2))
        return 0

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

    results = []
    for spec, command in zip(specs, commands):
        folder = Path(command["folder"])
        staging_log = args.output / f"trial_{spec.trial_index + 1:03d}.replay.log"
        returncode = _run(command["argv"], ROOT, staging_log)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "trial_spec.json").write_text(json.dumps(trial_to_json(spec), indent=2) + "\n")
        (folder / "command.txt").write_text(command["command"] + "\n")
        staging_log.replace(folder / "replay.log")
        result = _trial_result(folder, spec, returncode, args.capture_mode)
        results.append(result)
        (args.output / "trials.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "experiment": str(args.output), "config": config,
        "successful_trials": sum(bool(item.get("complete_success")) for item in results),
        "complete_collections": sum(bool(item.get("collection_complete")) for item in results),
        "attempted_trials": len(results), "trials": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(args.output / "report.md", config, results)
    print(json.dumps({"successful_trials": summary["successful_trials"],
                      "attempted_trials": summary["attempted_trials"],
                      "report": str(args.output / "report.md")}, indent=2), flush=True)
    return 0 if len(results) == args.count and all(item["collection_complete"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
