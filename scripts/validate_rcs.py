#!/usr/bin/env python3
"""Validate the canonical 16-channel RCS geometry and residual gate."""

from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srb.nmpc import (  # noqa: E402
    build_srb16_matrix,
    load_nmpc_config,
    validate_rcs_geometry,
    validate_srb8_geometry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nmpc/default.yaml")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_nmpc_config(config_path)
    full_rank = validate_rcs_geometry(
        tmax_N=config.task_params.tmax_N,
        raise_on_failure=False,
    )
    srb16 = validate_rcs_geometry(
        build_srb16_matrix(
            power_N=config.task_params.tmax_N,
            com_B_m=config.task_params.com_B_m,
        ),
        tmax_N=config.task_params.tmax_N,
        raise_on_failure=False,
    )
    legacy = validate_srb8_geometry()
    result = {
        "config": str(config.source_path),
        "canonical_16": full_rank,
        "srb16_asset_matrix": srb16,
        "legacy_srb8_contrast": legacy,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    required = ("rank_ok", "condition_ok", "faults_ok", "residual_ok")
    return 0 if all(bool(full_rank[key]) and bool(srb16[key]) for key in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
