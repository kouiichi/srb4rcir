#!/usr/bin/env python3
"""Measure hot-loop acados and OSQP runtimes on the 20 Hz contract."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srb.nmpc import (  # noqa: E402
    AcadosExpert,
    SafetyAllocator,
    build_rcs_matrix,
    load_nmpc_config,
)


def _percentiles(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "min_ms": float(np.min(data) * 1.0e3),
        "median_ms": float(np.median(data) * 1.0e3),
        "p99_ms": float(np.percentile(data, 99.0) * 1.0e3),
        "max_ms": float(np.max(data) * 1.0e3),
    }


def _solver_time_seconds(acados: AcadosExpert) -> float:
    """Read acados' own ``time_tot`` statistic when the backend exposes it."""

    solver = getattr(acados, "_solver", None)
    if solver is None:
        return float("nan")
    try:
        value = np.asarray(solver.get_stats("time_tot"), dtype=np.float64)
        return float(value.reshape(-1)[-1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nmpc/default.yaml")
    args = parser.parse_args()
    if args.iterations < 3:
        raise ValueError("--iterations must be at least 3")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_nmpc_config(config_path)
    params = config.task_params
    state = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    duty = np.full(16, 0.05, dtype=np.float64)
    acados = AcadosExpert(
        params,
        nlp_solver_type=config.solver.nlp_solver_type,
        max_iterations=(
            min(config.solver.production_max_iterations, 1)
            if config.solver.nlp_solver_type == "SQP_RTI"
            else config.solver.production_max_iterations
        ),
        qp_solver_cond_N=config.solver.qp_solver_cond_N,
        qp_solver_iter_max=config.solver.qp_solver_iter_max,
        code_export_root=config.resolved_code_export_root(ROOT),
    )
    safety = SafetyAllocator(params)
    # One call initializes generated solver memory and one call initializes
    # the persistent OSQP workspace; neither cold-start cost is a 20 Hz loop
    # cost because both objects are constructed before runtime operation.
    previous_result = acados.solve(state)
    safety.filter_and_allocate(
        build_rcs_matrix(tmax_N=params.tmax_N, com_B_m=params.com_B_m) @ duty,
        state,
        np.zeros(16),
    )
    acados_times: list[float] = []
    acados_solver_times: list[float] = []
    safety_times: list[float] = []
    accepted = 0
    for _ in range(args.iterations):
        start = time.perf_counter()
        result = acados.solve(state, warm_start=previous_result)
        acados_times.append(time.perf_counter() - start)
        acados_solver_times.append(_solver_time_seconds(acados))
        previous_result = result
        accepted += int(result.accepted)
        start = time.perf_counter()
        filtered = safety.filter_and_allocate(
            build_rcs_matrix(tmax_N=params.tmax_N, com_B_m=params.com_B_m) @ duty,
            state,
            result.actual_duty,
        )
        safety_times.append(time.perf_counter() - start)
        if filtered.status not in {"solved", "intervened", "residual_exceeded"}:
            raise RuntimeError(filtered.error_message or "safety filter failed")
    summary = {
        "config": str(config.source_path),
        "control_frequency_hz": 20,
        "iterations": args.iterations,
        "acados": _percentiles(acados_times),
        "acados_solver_internal": (
            _percentiles(acados_solver_times)
            if all(np.isfinite(acados_solver_times))
            else {"available": False}
        ),
        "osqp_safety": _percentiles(safety_times),
        "acados_accepted": accepted / args.iterations,
        "acados_p99_gate_ms": 40.0,
        "osqp_p99_gate_ms": 2.0,
        # The endpoint wall time includes Python-side state preparation,
        # independent rollout, validation, and OS scheduling.  Keep this
        # strict gate separate from the solver's internal statistic so a
        # shared workstation cannot make the two measurements ambiguous.
        "acados_p99_gate_pass": _percentiles(acados_times)["p99_ms"] <= 40.0,
        "osqp_p99_gate_pass": _percentiles(safety_times)["p99_ms"] <= 2.0,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["acados_p99_gate_pass"] and summary["osqp_p99_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
