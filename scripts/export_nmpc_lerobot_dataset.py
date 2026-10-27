#!/usr/bin/env python3
"""Write qualified SRB trials with the official LeRobotDataset v3 API.

Run this standalone exporter with LeRobot's Python, without importing SRB or
Isaac. Raw NPZ files remain the source of truth for terminal observations,
diagnostics, fuel records and simulator-only labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

FPS = 20
DEFAULT_REPO_ID = "srb/rendezvous_nmpc"


def check_runtime():
    if sys.version_info < (3, 12):
        raise RuntimeError("LeRobot requires Python >=3.12; pass --lerobot-python from its isolated environment")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def select_trials(trials):
    selected, rejected = [], []
    for record in trials:
        if record.get("collection_complete") and record.get("complete_success"):
            selected.append(record)
        else:
            rejected.append({**record, "export_reason": "requires collection_complete and complete_success"})
    return selected, rejected


def load_episode(record, mode):
    if record.get("scene") == "nasa_pair":
        return load_nasa_episode(record, mode)
    folder = Path(record["folder"])
    with np.load(folder / "replay.npz", allow_pickle=False) as data:
        states, actions = data["states"], data["duties"]
    n = len(actions)
    if (n == 0 or states.shape != (n + 1, 13) or actions.shape != (n, 16)
            or not np.isfinite(states).all() or not np.isfinite(actions).all()
            or np.any((actions < 0) | (actions > 1))):
        raise ValueError(f"{folder}: expected finite states[T+1,13] and executed duty[T,16] in [0,1]")
    gap = float(record["goal_surface_gap_m"])
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError(f"{folder}: invalid goal_surface_gap_m")
    episode = {"states": states.astype(np.float32), "actions": actions.astype(np.float32),
               "gap": gap, "mode": mode}
    if mode == "visual":
        with np.load(folder / "camera.npz", allow_pickle=False) as data:
            rgb, times = data["rgb"], data["time_s"]
        if (times.shape != (n + 1,) or not np.allclose(times, np.arange(n + 1) / FPS, rtol=0, atol=1e-8)
                or rgb.ndim != 4 or rgb.shape[0] != n + 1 or rgb.shape[-1] != 3
                or min(rgb.shape[1:3]) <= 0 or rgb.dtype != np.uint8):
            raise ValueError(f"{folder}: RGB and timestamps must align with every measured state at 20 Hz")
        episode["rgb"] = rgb
    elif mode != "state":
        raise ValueError(f"unsupported capture mode: {mode}")
    return episode


def load_nasa_episode(record, mode):
    """Read native NASA records, using verified executor duties as labels."""
    folder = Path(record["folder"])
    with np.load(folder / "trajectory.npz", allow_pickle=False) as data:
        states, commands, times = data["states"], data["duties"], data["time_s"]
    n, channels = len(commands), int(record["num_thrusters"])
    with np.load(folder / "executor_telemetry.npz", allow_pickle=False) as data:
        executed = data["duty"]
    if (n == 0 or channels != 12 or states.shape != (n + 1, 13)
            or commands.shape != (n, channels) or executed.shape != (n, 1, channels)
            or not np.isfinite(states).all() or not np.isfinite(executed).all()
            or np.any((executed < 0) | (executed > 1))
            or not np.array_equal(commands.astype(np.float32), executed[:, 0, :].astype(np.float32))):
        raise ValueError(f"{folder}: expected measured states[T+1,13] and matching executed duty[T,12]")
    expected_times = np.arange(n + 1) / FPS
    if times.shape != (n + 1,) or not np.allclose(times, expected_times, rtol=0., atol=1e-8):
        raise ValueError(f"{folder}: trajectory timestamps must cover every 20 Hz control step")
    gap = float(record["goal_surface_gap_m"])
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError(f"{folder}: invalid goal_surface_gap_m")
    with np.load(folder / "gimbal_states.npz", allow_pickle=False) as data:
        angles, rates = data["angles_rad"], data["rates_rad_s"]
        if (not np.array_equal(data["step"], np.arange(n + 1))
                or data["time_s"].shape != (n + 1,)
                or not np.allclose(data["time_s"], expected_times, rtol=0., atol=1e-8)):
            raise ValueError(f"{folder}: gimbal records must align with control steps")
    if (angles.shape != (n + 1, 2) or rates.shape != (n + 1, 2)
            or not np.isfinite(angles).all() or not np.isfinite(rates).all()):
        raise ValueError(f"{folder}: expected finite yaw/pitch angles and rates at every state")
    episode = {"states": states.astype(np.float32), "actions": executed[:, 0, :].astype(np.float32),
               "gap": gap, "mode": mode, "scene": "nasa_pair",
               "gimbal": np.concatenate((angles, rates), axis=1).astype(np.float32)}
    if mode == "visual":
        with np.load(folder / "onboard_capture/observations.npz", allow_pickle=False) as data:
            camera_times = data["time_s"]
            if "rgb_files" in data:
                filenames = data["rgb_files"]
                if filenames.shape != (n + 1,) or filenames.dtype.kind not in "US":
                    raise ValueError(f"{folder}: invalid streamed RGB index")
                paths = [folder / "onboard_capture" / str(name) for name in filenames]
                if not all(path.is_file() for path in paths):
                    raise ValueError(f"{folder}: streamed RGB index contains missing images")
                episode["rgb_paths"] = paths
                from PIL import Image
                with Image.open(paths[0]) as first:
                    if first.mode != "RGB":
                        raise ValueError(f"{folder}: streamed images must be RGB PNG")
                    episode["image_shape_hwc"] = (first.height, first.width, 3)
            else:
                rgb = data["rgb"]
                if (rgb.ndim != 4 or rgb.shape[0] != n + 1 or rgb.shape[-1] != 3
                        or min(rgb.shape[1:3]) <= 0 or rgb.dtype != np.uint8):
                    raise ValueError(f"{folder}: expected uint8 RGB at every measured state")
                episode["rgb"] = rgb
            if (camera_times.shape != (n + 1,)
                    or not np.allclose(camera_times, expected_times, rtol=0., atol=1e-8)
                    or not np.array_equal(data["capture_id"], np.arange(n + 1))
                    or data["measured_state13"].shape != states.shape
                    or not np.allclose(data["measured_state13"], states, rtol=0., atol=1e-6)
                    or data["gimbal_angles_rad"].shape != angles.shape
                    or not np.allclose(data["gimbal_angles_rad"], angles, rtol=0., atol=1e-7)
                    or data["gimbal_rates_rad_s"].shape != rates.shape
                    or not np.allclose(data["gimbal_rates_rad_s"], rates, rtol=0., atol=1e-7)):
                raise ValueError(f"{folder}: RGB, measured states and actual gimbal must align at 20 Hz")
            for key, shape in (("camera_mount_position_B_m", (n + 1, 3)),
                               ("camera_mount_quaternion_BC_wxyz", (n + 1, 4)),
                               ("intrinsic", (n + 1, 3, 3))):
                value = data[key]
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"{folder}: invalid per-frame {key}")
                episode[key] = value.astype(np.float32)
    elif mode != "state":
        raise ValueError(f"unsupported capture mode: {mode}")
    return episode


def episode_frames(episode, *, load_images=True):
    actions, states = episode["actions"], episode["states"]
    nasa = episode.get("scene") == "nasa_pair"
    task = (f"Approach NEAR solar-array plane with DESDynI bottom rim; gap {episode['gap']:g} m"
            if nasa else f"Approach target -X face with actor +Z face; gap {episode['gap']:g} m")
    for i, action in enumerate(actions):
        previous = actions[i - 1] if i else np.zeros(actions.shape[1], dtype=np.float32)
        low_state = [states[i, 10:13], previous, np.asarray([episode["gap"]], dtype=np.float32)]
        if nasa:
            low_state.append(episode["gimbal"][i])
        frame = {"action": action.copy(), "task": task,
                 "supervision.relative_state13": states[i].copy(),
                 "observation.state": np.concatenate(low_state)}
        if episode["mode"] == "visual":
            if load_images:
                if "rgb_paths" in episode:
                    from PIL import Image
                    with Image.open(episode["rgb_paths"][i]) as image:
                        rgb = np.asarray(image).copy()
                    if rgb.shape != episode["image_shape_hwc"] or rgb.dtype != np.uint8:
                        raise ValueError("streamed RGB shape or dtype changed within an episode")
                    frame["observation.images.onboard"] = rgb
                else:
                    frame["observation.images.onboard"] = episode["rgb"][i].copy()
            if nasa:
                for key in ("camera_mount_position_B_m", "camera_mount_quaternion_BC_wxyz", "intrinsic"):
                    frame[f"supervision.{key}"] = episode[key][i].reshape(-1).copy()
        else:
            frame["observation.environment_state"] = states[i].copy()
        yield frame


def dataset_features(episode):
    def numeric(names):
        return {"dtype": "float32", "shape": (len(names),), "names": names}
    state_names = ["rx_m", "ry_m", "rz_m", "vx_m_s", "vy_m_s", "vz_m_s",
                   "qw", "qx", "qy", "qz", "wx_body_rad_s", "wy_body_rad_s", "wz_body_rad_s"]
    channels = episode["actions"].shape[1]
    low_names = state_names[-3:] + [f"previous_duty_{i:02d}" for i in range(channels)] + ["goal_surface_gap_m"]
    if episode.get("scene") == "nasa_pair":
        low_names += ["gimbal_yaw_rad", "gimbal_pitch_rad", "gimbal_yaw_rate_rad_s", "gimbal_pitch_rate_rad_s"]
    features = {"action": numeric([f"duty_{i:02d}" for i in range(channels)]),
                "supervision.relative_state13": numeric(state_names),
                "observation.state": numeric(low_names)}
    if episode["mode"] == "visual":
        h, w = episode["image_shape_hwc"][:2] if "rgb_paths" in episode else episode["rgb"].shape[1:3]
        features["observation.images.onboard"] = {
            "dtype": "image", "shape": (3, h, w), "names": ["channels", "height", "width"]}
        if episode.get("scene") == "nasa_pair":
            features.update({
                "supervision.camera_mount_position_B_m": numeric(["x_m", "y_m", "z_m"]),
                "supervision.camera_mount_quaternion_BC_wxyz": numeric(["qw", "qx", "qy", "qz"]),
                "supervision.intrinsic": numeric([f"K_{r}{c}" for r in range(3) for c in range(3)]),
            })
    else:
        features["observation.environment_state"] = numeric(state_names)
    return features


def save_episode_npz(episode, folder):
    """Save aligned ACT rows without requiring the isolated LeRobot runtime."""
    folder = Path(folder)
    features = dataset_features(episode)
    arrays = {key: [] for key in features if not key.startswith("observation.images.")}
    for frame in episode_frames(episode, load_images=False):
        for key in arrays:
            arrays[key].append(frame[key])
    arrays = {key: np.stack(values) for key, values in arrays.items()}
    if episode["mode"] == "visual":
        if "rgb_paths" in episode:
            arrays["observation.images.onboard_path"] = np.asarray([
                str(path.relative_to(folder)) for path in episode["rgb_paths"][:-1]])
        else:
            arrays["observation.images.onboard"] = episode["rgb"][:-1]
    n = len(episode["actions"])
    np.savez_compressed(folder / "dataset.npz", time_s=np.arange(n) / FPS, **arrays)
    schema = {"num_frames": n, "fps": FPS, "scene": episode.get("scene", "surface"),
              "features": features, "npz_shapes": {key: list(value.shape) for key, value in arrays.items()},
              "rgb_storage": ("lossless RGB PNG referenced by observation.images.onboard_path, relative to this dataset.npz"
                              if "rgb_paths" in episode else "uint8 HWC; convert to CHW float32 [0,1] for standard ACT"),
              "terminal_observation_in_raw": True,
              "action_alignment": "observation[i], actual executed duty[i] -> measured state[i+1]",
              "source": ("trajectory.npz, executor_telemetry.npz, gimbal_states.npz" +
                         (", onboard_capture/observations.npz" if episode["mode"] == "visual" else "")),
              "supervision": "supervision.* contains simulator audit fields, excluded from policy inputs",
              "camera_tracking": "limited kinematic gimbal tracks simulator target position; not an ACT action"}
    (folder / "dataset_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    return schema


def export_dataset(source, output, repo_id=DEFAULT_REPO_ID):
    Dataset = check_runtime()
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    summary = json.loads((source / "summary.json").read_text())
    mode = summary["config"]["capture_mode"]
    selected, rejected = select_trials(summary["trials"])
    if not selected:
        raise ValueError("No qualified complete-success trials; raw failure artifacts are retained")
    base = Path(summary["config"].get("cwd", source))
    selected = [{**record, "folder": str((base / record["folder"]).resolve())} for record in selected]
    # Validate all episode boundaries and image sizes before creating the output.
    features = None
    for record in selected:
        episode = load_episode(record, mode)
        current = dataset_features(episode)
        if features is not None and current != features:
            raise ValueError("All exported episodes must have identical feature shapes")
        features = current
        del episode
    nasa = selected[0].get("scene") == "nasa_pair"
    dataset = Dataset.create(repo_id=repo_id, root=output, fps=FPS, features=features,
                             robot_type="srb_desdyni_rcs12_gimbal" if nasa else "srb_rcs16", use_videos=False)
    mapping = []
    try:
        for index, record in enumerate(selected):
            episode = load_episode(record, mode)
            for frame in episode_frames(episode):
                dataset.add_frame(frame)
            dataset.save_episode()
            mapping.append({**record, "episode_index": index, "frames": len(episode["actions"])})
            del episode
    finally:
        dataset.finalize()
    manifest = {"repo_id": repo_id, "root": str(output), "source": str(source),
                "fps": FPS, "capture_mode": mode, "storage": "lossless images + Parquet",
                "num_episodes": len(mapping), "num_frames": sum(item["frames"] for item in mapping),
                "episodes": mapping, "excluded_trials": rejected,
                "collection_config": summary["config"],
                "action_alignment": "observation[i], action[i] -> state[i+1]; terminal frame remains in raw NPZ",
                "visual_state_contract": ("body angular rate (ideal gyro proxy), previous executed duty12 (zero at reset), known goal gap, actual yaw/pitch and joint rates"
                                          if nasa else "body angular rate (ideal gyro proxy), previous executed duty16 (zero at reset), known goal gap"),
                "state_input_contract": "privileged environment_state13 plus the same observation.state20 as visual mode",
                "supervision_contract": "supervision.* is excluded by LeRobot dataset_to_policy_features; state mode is privileged",
                "fuel_contract": "fuel is not recorded in observation.state; raw executor artifacts remain available",
                "camera_contract": "NASA dynamic body-to-camera mount and intrinsics are supervision fields; metric depth and semantic mask remain in raw NPZ; gimbal tracks simulator target position" if nasa else "fixed onboard camera"}
    (output / "srb_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Collection directory containing summary.json")
    parser.add_argument("--output", type=Path, help="New local LeRobot dataset root")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--check", action="store_true", help="Check isolated Python and official dataset imports only")
    args = parser.parse_args()
    if args.check:
        check_runtime()
        print(json.dumps({"python": sys.version, "dataset_api": "LeRobotDataset", "ready": True}))
        return 0
    if args.input is None or args.output is None:
        parser.error("--input and --output are required for export")
    manifest = export_dataset(args.input, args.output, args.repo_id)
    print(json.dumps({key: manifest[key] for key in ("root", "num_episodes", "num_frames")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
