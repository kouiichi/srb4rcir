#!/usr/bin/env python3
"""Validate official DataLoader + ACT paths with explicitly synthetic data.

Run with isolated LeRobot Python. This creates format fixtures, never starts
Isaac/PhysX, and does not generate expert demonstrations or optimize a policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_nmpc_lerobot_dataset import export_dataset


def random_collection(root, mode, seed):
    """Write raw collector-shaped fixtures with normalized wxyz quaternions."""
    rng = np.random.default_rng(seed)
    root.mkdir()
    trials = []
    for i, (steps, gap) in enumerate(zip((47, 53, 41), (.1, .3, .5))):
        folder = root / f"trial_{i + 1:03d}"
        folder.mkdir()
        states = rng.normal(size=(steps + 1, 13))
        states[:, 6:10] /= np.linalg.norm(states[:, 6:10], axis=1, keepdims=True)
        states[:, 3:6] *= .02
        states[:, 10:13] *= .005
        actions = rng.uniform(0., 1., size=(steps, 16))
        np.savez_compressed(folder / "replay.npz", states=states, duties=actions)
        if mode == "visual":
            rgb = rng.integers(0, 256, size=(steps + 1, 64, 64, 3), dtype=np.uint8)
            np.savez_compressed(folder / "camera.npz", rgb=rgb, time_s=np.arange(steps + 1) * .05)
        trials.append({
            "folder": str(folder), "goal_surface_gap_m": gap,
            "synthetic": True, "steps": steps, "requested_steps": steps,
            "collection_complete": True, "complete_success": True,
            "success_provenance": "Fixture acceptance flags only; no physics or controller success measured",
        })
    config = {"capture_mode": mode, "cwd": str(Path.cwd()), "seed": seed,
              "synthetic": True, "data_origin": "random format fixtures",
              "control_period_s": .05, "horizon_steps": 40, "execute_steps": 3}
    (root / "summary.json").write_text(json.dumps({"config": config, "trials": trials}, indent=2) + "\n")
    return trials


def validate_profile(root, mode, seed):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features

    raw = root / "raw"
    trials = random_collection(raw, mode, seed)
    repo_id = f"srb/synthetic_{mode}"
    export_dataset(raw, root / "lerobot", repo_id)
    data = LeRobotDataset(repo_id, root=root / "lerobot", video_backend="pyav",
                         delta_timestamps={"action": [i * .05 for i in range(40)]})
    features = dataset_to_policy_features(data.features)
    assert "supervision.relative_state13" not in features
    config = ACTConfig(
        device="cpu", chunk_size=40, n_action_steps=3,
        input_features={k: v for k, v in features.items() if k != "action"},
        output_features={"action": features["action"]},
        pretrained_backbone_weights=None,
    )
    policy = ACTPolicy(config)
    preprocessor, postprocessor = make_act_pre_post_processors(config, data.meta.stats)
    state_key, dim = "observation.state", 20
    if mode == "state":
        assert tuple(data[0]["observation.environment_state"].shape) == (13,)
    assert tuple(data[0][state_key].shape) == (dim,)
    assert len(data) == 141 and data.num_episodes == 3
    # Include starts, middles and ends from all three episodes in each batch.
    indices = [0, 46, 47, 99, 100, 140, 20, 70]
    batches = []
    for size in (1, 2, 8):
        batch = next(iter(DataLoader(Subset(data, indices[:size]), batch_size=size, num_workers=0)))
        state_shape = list(batch[state_key].shape)
        action_shape = list(batch["action"].shape)
        assert state_shape == [size, dim]
        assert action_shape == [size, 40, 16]
        processed = preprocessor(batch)
        assert list(processed[state_key].shape) == state_shape
        environment_shape = None
        if mode == "state":
            environment_shape = list(processed["observation.environment_state"].shape)
            assert environment_shape == [size, 13]
        policy.train()
        with torch.no_grad():
            loss, diagnostics = policy(processed)
        assert torch.isfinite(loss).all() and all(np.isfinite(value) for value in diagnostics.values())
        policy.reset()
        predicted = policy.predict_action_chunk(processed)
        assert tuple(predicted.shape) == (size, 40, 16) and torch.isfinite(predicted).all()
        action = postprocessor(policy.select_action(processed))
        assert tuple(action.shape) == (size, 16) and torch.isfinite(action).all()
        batches.append({
            "batch_size": size, "state_shape": state_shape,
            "processed_state_shape": list(processed[state_key].shape),
            "environment_state_shape": environment_shape,
            "action_shape": action_shape, "padding_shape": list(batch["action_is_pad"].shape),
            "loss": float(loss), "loss_components": diagnostics,
            "predicted_action_shape": list(predicted.shape), "inference_action_shape": list(action.shape),
        })
    offset = 0
    for record in trials:
        steps = record["steps"]
        end = data[offset + steps - 1]
        assert (~end["action_is_pad"]).sum().item() == 1
        with np.load(Path(record["folder"]) / "replay.npz") as original:
            np.testing.assert_allclose(end["action"][0].numpy(), original["duties"][-1], rtol=1e-6)
        assert np.isclose(data[offset][state_key][-1].item(), record["goal_surface_gap_m"])
        np.testing.assert_allclose(data[offset][state_key][3:19].numpy(), 0.)
        offset += steps
    return {"dataset": str(root / "lerobot"), "num_episodes": data.num_episodes,
            "num_frames": len(data), "state_key": state_key, "chunk_size": config.chunk_size,
            "n_action_steps": config.n_action_steps, "use_vae": config.use_vae,
            "architecture": {"dim_model": config.dim_model, "n_heads": config.n_heads,
                             "dim_feedforward": config.dim_feedforward,
                             "n_encoder_layers": config.n_encoder_layers,
                             "n_decoder_layers": config.n_decoder_layers,
                             "n_vae_encoder_layers": config.n_vae_encoder_layers},
            "pretrained_backbone_weights": config.pretrained_backbone_weights,
            "terminal_valid_actions": 1, "batches": batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory for synthetic fixtures and results")
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    profiles = {}
    for mode in ("state", "visual"):
        root = output / mode
        root.mkdir()
        profiles[mode] = validate_profile(root, mode, args.seed)
    report = {"synthetic": True, "passed": True, "seed": args.seed,
              "command": shlex.join(sys.orig_argv), "cwd": str(Path.cwd()),
              "python": sys.version, "torch": torch.__version__, "device": "cpu",
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "kit": "not started", "physx": "not started", "physics_only": "not applicable",
              "profiles": profiles, "optimizer_steps": 0,
              "scope": "Format, batching, preprocessing, CVAE loss forward and inference only; no physical consistency or learned control quality"}
    (output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
