#!/usr/bin/env python3
"""Dedicated runner for Space Robotics Bench RL experiments."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

SRB_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = SRB_ROOT / "srb_0.0.5_env.sh"
ENV_CACHE = SRB_ROOT / ".cache" / "env.json"
HYPERPARAMS_DIR = SRB_ROOT / "hyperparams"
LOGS_DIR = SRB_ROOT / "logs"


def discover_algorithms() -> list[str]:
    """Derive algorithm ids from the hyperparams directory layout."""
    if not HYPERPARAMS_DIR.is_dir():
        return []

    algorithms: set[str] = set()
    for yaml_path in HYPERPARAMS_DIR.rglob("*.yaml"):
        rel = yaml_path.relative_to(HYPERPARAMS_DIR)
        if len(rel.parts) == 1:
            algorithms.add("dreamer" if rel.stem == "dreamerv3" else rel.stem)
        elif len(rel.parts) == 2:
            algorithms.add(f"{rel.parts[0]}_{rel.stem}")
    return sorted(algorithms)


def load_envs() -> list[str]:
    """Read the env ids cached by the SRB CLI."""
    try:
        with ENV_CACHE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return sorted(data) if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def add_workflow_parser(subparsers, name: str, workflow: str) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(name, help=f"{workflow} an SRB RL agent")
    parser.set_defaults(workflow=workflow)
    parser.add_argument("--env", "-e", required=True, help="SRB env id (e.g. rendezvous)")
    algo_choices = discover_algorithms()
    parser.add_argument(
        "--algo",
        required=True,
        choices=algo_choices or None,
        help="RL algorithm",
    )
    parser.add_argument("--num-envs", type=int, default=4, help="Parallel environments")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value, e.g. 3")
    parser.add_argument("--logs", type=Path, default=LOGS_DIR, help="Log root directory")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--video", action="store_true", help="Record videos")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override, repeatable",
    )
    if workflow == "train":
        mutex = parser.add_mutually_exclusive_group()
        mutex.add_argument(
            "--continue",
            "--continue_training",
            dest="continue_training",
            action="store_true",
            help="Continue from last checkpoint",
        )
        mutex.add_argument("--model", help="Continue from specific checkpoint")
    else:
        parser.add_argument("--model", help="Checkpoint to evaluate")
    return parser


def run_workflow(args: argparse.Namespace, workflow: str) -> int:
    known_envs = load_envs()
    if known_envs and args.env not in known_envs:
        print(
            f"[WARN] {args.env} is not in {ENV_CACHE}; "
            "refresh the cache with `srb ls env` if this is unexpected"
        )

    args.logs = args.logs.expanduser().resolve()
    if not args.dry_run:
        args.logs.mkdir(parents=True, exist_ok=True)

    srb_cmd = ["srb", "agent", workflow, "--algo", args.algo, "--env", args.env]
    if args.headless:
        srb_cmd.append("--headless")
    if args.video:
        srb_cmd.append("--video")
    srb_cmd.extend(["--logs", str(args.logs)])
    if args.model:
        srb_cmd.extend(["--model", str(Path(args.model).expanduser().resolve())])
    elif workflow == "train" and args.continue_training:
        srb_cmd.append("--continue_training")
    srb_cmd.append(f"env.num_envs={args.num_envs}")
    srb_cmd.extend(args.override)

    full_cmd = ["bash", str(WRAPPER), *srb_cmd]
    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("[srb_rl_agent] " + " ".join(shlex.quote(part) for part in full_cmd))
    if args.dry_run:
        return 0
    return subprocess.run(full_cmd, env=env, check=False).returncode


def cmd_list(_args: argparse.Namespace) -> int:
    algorithms = discover_algorithms()
    envs = load_envs()

    print(f"Algorithms ({len(algorithms)}):")
    for algo in algorithms:
        print(f"  {algo}")
    print(f"Envs ({len(envs)}):")
    if envs:
        for env in envs:
            print(f"  {env}")
    else:
        print("  (empty; run: source ./srb_0.0.5_env.sh && srb ls env)")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not LOGS_DIR.is_dir():
        print(f"No logs yet: {LOGS_DIR}")
        return 0

    found = False
    for env_dir in sorted(path for path in LOGS_DIR.iterdir() if path.is_dir()):
        if args.env and env_dir.name != args.env:
            continue
        for algo_dir in sorted(path for path in env_dir.iterdir() if path.is_dir()):
            if args.algo and algo_dir.name != args.algo:
                continue
            runs = [path for path in algo_dir.iterdir() if path.is_dir()]
            if not runs:
                continue
            latest = max(runs, key=lambda path: path.stat().st_mtime)
            ckpts = sorted(latest.glob("ckpt/*.zip"))
            eval_ckpts = sorted(latest.glob("eval_ckpt/*.zip"))
            found = True
            print(f"{env_dir.name}/{algo_dir.name}: {latest.name}")
            for ckpt in ckpts[-3:]:
                print(f"  ckpt: {ckpt.name}")
            for ckpt in eval_ckpts[-1:]:
                print(f"  eval_ckpt: {ckpt.name}")

    if not found:
        print("No matching SRB RL runs.")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    problems = []
    if not WRAPPER.is_file():
        problems.append(f"Missing wrapper: {WRAPPER}")
    if not ENV_CACHE.is_file():
        problems.append(f"Missing env cache: {ENV_CACHE}")
    if not HYPERPARAMS_DIR.is_dir():
        problems.append(f"Missing hyperparams dir: {HYPERPARAMS_DIR}")
    if not LOGS_DIR.is_dir():
        problems.append(f"Missing logs dir: {LOGS_DIR} (created on first run)")

    print(f"Wrapper: {WRAPPER}")
    print(f"Algorithms: {len(discover_algorithms())}")
    print(f"Envs cached: {len(load_envs())}")
    if problems:
        for problem in problems:
            print(f"[ERROR] {problem}")
        return 1
    print("[OK] SRB RL agent prerequisites are present.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SRB 0.0.5 RL experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_workflow_parser(subparsers, "train", "train")
    add_workflow_parser(subparsers, "eval", "eval")

    list_parser = subparsers.add_parser("list", help="List algorithms and cached envs")
    list_parser.set_defaults(func=cmd_list)

    status_parser = subparsers.add_parser("status", help="Show latest SRB RL runs")
    status_parser.add_argument("--env", "-e", default=None)
    status_parser.add_argument("--algo", default=None)
    status_parser.set_defaults(func=cmd_status)

    doctor_parser = subparsers.add_parser("doctor", help="Check SRB RL agent prerequisites")
    doctor_parser.set_defaults(func=cmd_doctor)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "train":
        return run_workflow(args, "train")
    if args.command == "eval":
        return run_workflow(args, "eval")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
