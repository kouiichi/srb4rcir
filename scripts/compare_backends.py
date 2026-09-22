#!/usr/bin/env python3
"""Compare IPOPT and acados on deterministic State13 initial conditions."""

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

from srb.nmpc import AcadosExpert, IpoptOracle, load_nmpc_config  # noqa: E402


def initial_state(seed: int) -> np.ndarray:
    """Generate one reproducible safe initial State13 for comparison."""

    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = rng.uniform(-0.20, 0.20)
    quaternion = np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis))
    return np.concatenate(
        (
            np.array([rng.uniform(1.6, 3.0), rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)]),
            rng.uniform(-0.03, 0.03, size=3),
            quaternion,
            rng.uniform(-0.02, 0.02, size=3),
        )
    ).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/nmpc/default.yaml")
    parser.add_argument("--ipopt-max-iterations", type=int)
    parser.add_argument("--acados-max-iterations", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("--seeds must be positive")

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_nmpc_config(config_path)
    params = config.task_params
    ipopt_iterations = (
        config.solver.oracle_max_iterations
        if args.ipopt_max_iterations is None
        else args.ipopt_max_iterations
    )
    acados_iterations = (
        config.solver.production_max_iterations
        if args.acados_max_iterations is None
        else args.acados_max_iterations
    )
    oracle = IpoptOracle(
        max_iterations=ipopt_iterations,
        tolerance=config.solver.tolerance,
    )
    production = AcadosExpert(
        params,
        nlp_solver_type=config.solver.nlp_solver_type,
        max_iterations=(
            min(acados_iterations, 1)
            if config.solver.nlp_solver_type == "SQP_RTI"
            else acados_iterations
        ),
        qp_solver_cond_N=config.solver.qp_solver_cond_N,
        qp_solver_iter_max=config.solver.qp_solver_iter_max,
        code_export_root=config.resolved_code_export_root(ROOT),
    )
    rows: list[dict[str, object]] = []
    for seed in range(args.seeds):
        state = initial_state(seed)
        start = time.perf_counter()
        ipopt = oracle.solve(state, params)
        ipopt_wall = time.perf_counter() - start
        start = time.perf_counter()
        acados = production.solve(state, params)
        acados_wall = time.perf_counter() - start
        row: dict[str, object] = {
            "seed": seed,
            "ipopt_status": ipopt.status,
            "ipopt_quality": ipopt.quality_verdict,
            "ipopt_objective": ipopt.objective,
            "ipopt_runtime_s": ipopt.runtime_s,
            "ipopt_wall_s": ipopt_wall,
            "acados_status": acados.status,
            "acados_quality": acados.quality_verdict,
            "acados_objective": acados.objective,
            "acados_runtime_s": acados.runtime_s,
            "acados_wall_s": acados_wall,
            "objective_relative_difference": np.nan,
            "failure_reason": "; ".join(
                value for value in (ipopt.failure_reason, acados.failure_reason) if value
            ),
        }
        if ipopt.accepted and acados.accepted:
            row["objective_relative_difference"] = abs(ipopt.objective - acados.objective) / max(
                abs(ipopt.objective), 1.0
            )
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    valid = [
        float(row["objective_relative_difference"])
        for row in rows
        if np.isfinite(float(row["objective_relative_difference"]))
    ]
    accepted = len(valid)
    max_difference = max(valid, default=float("nan"))
    summary = {
        "config": str(config.source_path),
        "control_frequency_hz": 1.0 / params.control_period_s,
        "horizon_steps": params.horizon_steps,
        "seeds": args.seeds,
        "paired_accepted": accepted,
        "paired_acceptance_rate": accepted / args.seeds,
        "max_relative_objective_difference": max_difference,
        "objective_gate_2_percent": bool(valid) and max_difference < 0.02,
        "ipopt_acceptance": sum(row["ipopt_quality"] == "accepted" for row in rows) / args.seeds,
        "acados_acceptance": sum(row["acados_quality"] == "accepted" for row in rows) / args.seeds,
    }
    print(json.dumps({"summary": summary}, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return 0 if summary["objective_gate_2_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
