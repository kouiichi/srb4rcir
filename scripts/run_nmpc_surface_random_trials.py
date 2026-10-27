#!/usr/bin/env python3
"""Video-trial entry point for the unified NMPC collection runner.

Uses five trials, seed 20260918, fill intensity 600 and annotated overview
videos by default. Randomization, execution and reporting belong to the
collection runner.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_nmpc_surface_dataset as collector

DEFAULT_PLAN = collector.DEFAULT_PLAN
DEFAULT_ISAAC_PYTHON = collector.DEFAULT_ISAAC_PYTHON


def build_replay_command(trial, output, plan, *, steps, seed, isaac_python,
                         wrapper, sync_s=120., fill_intensity=600.):
    return collector.build_replay_command(
        trial, output, plan, steps=steps, seed=seed, isaac_python=isaac_python,
        wrapper=wrapper, sync_s=sync_s, capture_mode="video", fill_intensity=fill_intensity)


def main(argv=None):
    return collector.main([
        "--capture-mode", "video", "--annotate-video", "--count", "5",
        "--seed", "20260918", "--fill-intensity", "600",
        *(sys.argv[1:] if argv is None else argv),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
