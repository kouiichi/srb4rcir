#!/usr/bin/env python3
"""Run a deterministic standalone NMPC build/smoke check.

This script does not initialize Isaac Sim.  The HCW/RCS expert is intentionally
independent of the SRB renderer; run it through ``srb_0.0.5_env.sh`` when the
IPOPT/acados checks are desired.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srb.nmpc import (  # noqa: E402
    AcadosExpert,
    IpoptOracle,
    SafetyAllocator,
    Srb16Adapter,
    SrbAdapter,
    build_rcs_matrix,
    load_nmpc_config,
    validate_rcs_geometry,
)
from srb.nmpc.dynamics import rk4_step  # noqa: E402


def _initial_state() -> np.ndarray:
    return np.array(
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-ipopt", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    config = load_nmpc_config(config_path)
    params = config.task_params
    geometry = validate_rcs_geometry(
        tmax_N=params.tmax_N,
        rho_m=0.30,
        raise_on_failure=True,
    )
    state = _initial_state()
    zero_next = rk4_step(state, np.zeros(16), params)
    assert np.isclose(np.linalg.norm(zero_next[6:10]), 1.0, atol=1.0e-12)
    nominal_duty = np.full(16, 0.05, dtype=np.float64)
    safety = SafetyAllocator(params)
    filtered = safety.filter_and_allocate(
        build_rcs_matrix(tmax_N=params.tmax_N, com_B_m=params.com_B_m) @ nominal_duty,
        state,
        np.zeros(16, dtype=np.float64),
    )
    if filtered.status not in {"solved", "intervened", "residual_exceeded"}:
        raise RuntimeError(f"safety smoke failed: {filtered.error_message}")
    _, _, adapter_diagnostics = SrbAdapter(task_params=params).step(nominal_duty)
    _, _, srb16_diagnostics = Srb16Adapter(task_params=params).step(nominal_duty)

    print(f"config={config_path}")
    print(f"timing=20Hz Ts={params.control_period_s:.3f}s N={params.horizon_steps}")
    print(f"rcs_rank={geometry['rank']} condition={geometry['normalized_condition']:.6g}")
    print(f"safety_status={filtered.status} runtime_ms={filtered.runtime_s * 1.0e3:.3f}")
    print(f"srb_contrast_rank={adapter_diagnostics['rank']} label_valid={adapter_diagnostics['label_valid']}")
    print(f"srb16_rank={srb16_diagnostics['rank']} label_valid={srb16_diagnostics['label_valid']}")
    if not srb16_diagnostics["label_valid"]:
        raise RuntimeError("SRB 16-RCS adapter did not produce an exact executable label")

    if not args.skip_ipopt:
        ipopt = IpoptOracle(
            max_iterations=config.solver.oracle_max_iterations,
            tolerance=config.solver.tolerance,
        ).solve(state, params)
        print(f"ipopt_status={ipopt.status} quality={ipopt.quality_verdict} objective={ipopt.objective:.9g}")
        if not ipopt.accepted:
            raise RuntimeError(f"IPOPT smoke failed: {ipopt.failure_reason}")

    acados_available = bool(os.environ.get("ACADOS_SOURCE_DIR")) and (
        Path(os.environ.get("ACADOS_SOURCE_DIR", "")) / "lib" / "libacados.so"
    ).exists()
    if acados_available:
        acados = AcadosExpert(
            params,
            nlp_solver_type=config.solver.nlp_solver_type,
            max_iterations=config.solver.production_max_iterations,
            qp_solver_cond_N=config.solver.qp_solver_cond_N,
            qp_solver_iter_max=config.solver.qp_solver_iter_max,
            code_export_root=config.resolved_code_export_root(ROOT),
        ).solve(state)
        print(f"acados_status={acados.status} quality={acados.quality_verdict} objective={acados.objective:.9g}")
        if not acados.accepted:
            raise RuntimeError(f"acados smoke failed: {acados.failure_reason}")
    else:
        print("acados_status=not_configured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
