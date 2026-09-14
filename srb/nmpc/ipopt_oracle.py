"""CasADi/IPOPT direct-transcription oracle for NMPC demonstrations."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from .constraints import casadi_fov_terms, casadi_koz_terms, validate_solution
from .dynamics import (
    camera_target_point_L,
    casadi_quaternion_error_vector,
    casadi_rk4_map,
    reference_trajectory,
    rk4_step,
)
from .rcs import rcs_matrix_from_task
from .types import ExpertResult, FloatArray, TaskParams


def _require_casadi() -> Any:
    try:
        import casadi as ca
    except ModuleNotFoundError as exc:  # pragma: no cover - environment error
        raise RuntimeError(
            "CasADi is required for IpoptOracle; run srb_0.0.5_env.sh with Isaac Sim Python"
        ) from exc
    return ca


def _as_state(value: npt.ArrayLike) -> FloatArray:
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (13,) or not np.all(np.isfinite(state)):
        raise ValueError("x0 must be a finite shape-(13,) float64 array")
    state = state.copy()
    norm = np.linalg.norm(state[6:10])
    if norm <= 1.0e-12:
        raise ValueError("x0 quaternion norm is too small")
    state[6:10] /= norm
    return np.ascontiguousarray(state)


def _previous_duty(warm_start: Any) -> FloatArray:
    if isinstance(warm_start, ExpertResult):
        value = warm_start.actual_duty
    elif isinstance(warm_start, Mapping) and "previous_duty" in warm_start:
        value = warm_start["previous_duty"]
    else:
        value = np.zeros(16, dtype=np.float64)
    previous = np.asarray(value, dtype=np.float64)
    if previous.shape != (16,) or not np.all(np.isfinite(previous)):
        raise ValueError("warm_start previous_duty must have shape (16,) and be finite")
    if np.any(previous < 0.0) or np.any(previous > 1.0):
        raise ValueError("warm_start previous_duty must lie in [0,1]")
    return np.ascontiguousarray(previous)


def _initial_guess(
    task_params: TaskParams,
    x0: FloatArray,
    warm_start: Any,
    rcs_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    refs = reference_trajectory(
        task_params,
        np.arange(task_params.horizon_steps + 1, dtype=np.float64) * task_params.control_period_s,
    )
    # A feasible zero-input rollout is much more robust than initializing all
    # future states at the reference, which generally violates the multiple-
    # shooting equalities and can make IPOPT classify a feasible problem as
    # infeasible before it has recovered the dynamics manifold.
    state_guess = np.empty_like(refs)
    state_guess[0] = x0
    duty_guess = np.zeros((task_params.horizon_steps, 16), dtype=np.float64)
    for index in range(task_params.horizon_steps):
        state_guess[index + 1] = rk4_step(
            state_guess[index], duty_guess[index], task_params, rcs_matrix=rcs_matrix
        )
    if isinstance(warm_start, ExpertResult):
        old_state = warm_start.predicted_state
        old_duty = warm_start.predicted_duty
    elif isinstance(warm_start, Mapping):
        old_state = warm_start.get("x")
        old_duty = warm_start.get("u")
    else:
        old_state = old_duty = None
    if old_state is not None:
        candidate = np.asarray(old_state, dtype=np.float64)
        if candidate.shape == state_guess.shape and np.all(np.isfinite(candidate)):
            state_guess[1:] = candidate[:-1]
            state_guess[0] = x0
    if old_duty is not None:
        candidate = np.asarray(old_duty, dtype=np.float64)
        if candidate.shape == duty_guess.shape and np.all(np.isfinite(candidate)):
            duty_guess[:-1] = candidate[1:]
            duty_guess[-1] = candidate[-1]
    return state_guess, np.clip(duty_guess, 0.0, 1.0)


def _stats_iterations(stats: Mapping[str, Any]) -> tuple[int, float]:
    count = stats.get("iter_count", 0)
    try:
        iterations = int(np.asarray(count).reshape(-1)[-1])
    except (TypeError, ValueError, IndexError):
        iterations = 0
    values: list[float] = []
    iteration_stats = stats.get("iterations", {})
    if isinstance(iteration_stats, Mapping):
        for key in ("inf_pr", "inf_du", "compl_inf"):
            if key in iteration_stats:
                try:
                    values.append(float(np.abs(np.asarray(iteration_stats[key], dtype=float).reshape(-1)[-1])))
                except (TypeError, ValueError):
                    pass
    return iterations, (max(values) if values else float("nan"))


class IpoptOracle:
    """High-accuracy CasADi/IPOPT NMPC oracle.

    ``solve`` receives one timestamped State13 and one validated TaskParams.
    It constructs a 20-interval direct transcription with 16 duty controls,
    normalized five-substep RK4 dynamics, hard KOZ/ECBF and actuator limits,
    and optional FOV limits.  Output arrays use float64 shapes ``(21,13)``
    and ``(20,16)/(20,6)``; wrenches are body-frame SI values and timestamps
    lie on the 20 Hz grid.  IPOPT status is not a quality verdict: the result
    is independently re-simulated and rejected on any hard violation, NaN,
    or dynamics residual.  ``warm_start`` may be an earlier ExpertResult or a
    mapping with ``x``, ``u``, and optional ``previous_duty`` arrays.
    """

    def __init__(self, *, max_iterations: int = 80, tolerance: float = 1.0e-6) -> None:
        if max_iterations <= 0 or tolerance <= 0.0:
            raise ValueError("max_iterations and tolerance must be positive")
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def solve(
        self,
        x0: npt.ArrayLike,
        task_params: TaskParams,
        warm_start: Any = None,
    ) -> ExpertResult:
        """Solve one expert OCP and return diagnostics plus the quality gate."""

        ca = _require_casadi()
        x_initial = _as_state(x0)
        previous = _previous_duty(warm_start)
        params = task_params
        start = time.perf_counter()
        N = params.horizon_steps
        B = rcs_matrix_from_task(params)
        references = reference_trajectory(
            params,
            np.arange(N + 1, dtype=np.float64) * params.control_period_s,
        )
        fov_targets = np.vstack(
            [camera_target_point_L(params, reference) for reference in references]
        )
        state_guess, duty_guess = _initial_guess(params, x_initial, warm_start, B)
        opti = ca.Opti()
        X = opti.variable(13, N + 1)
        U = opti.variable(16, N)
        block_starts = set(params.action_block_starts)

        opti.subject_to(X[:, 0] == ca.DM(x_initial))
        opti.subject_to(opti.bounded(0.0, U, 1.0))
        total_cost = 0.0
        for k in range(N):
            opti.subject_to(
                X[:, k + 1]
                == casadi_rk4_map(X[:, k], U[:, k], params)
            )
            opti.subject_to(
                opti.bounded(
                    -params.velocity_limit_m_s,
                    X[3:6, k],
                    params.velocity_limit_m_s,
                )
            )
            opti.subject_to(
                opti.bounded(
                    -params.angular_velocity_limit_rad_s,
                    X[10:13, k],
                    params.angular_velocity_limit_rad_s,
                )
            )
            h, ecbf = casadi_koz_terms(X[:, k], U[:, k], params)
            opti.subject_to(h >= 0.0)
            opti.subject_to(ecbf >= 0.0)
            for pair in range(8):
                opti.subject_to(U[2 * pair, k] + U[2 * pair + 1, k] <= params.reverse_pair_limit)
            prior = ca.DM(previous) if k == 0 else U[:, k - 1]
            opti.subject_to(
                opti.bounded(-params.max_duty_delta, U[:, k] - prior, params.max_duty_delta)
            )
            if k not in block_starts:
                opti.subject_to(U[:, k] == U[:, k - 1])
            q_error = casadi_quaternion_error_vector(
                ca.DM(references[k, 6:10]), X[6:10, k]
            )
            position_error = X[0:3, k] - ca.DM(references[k, 0:3])
            velocity_error = X[3:6, k] - ca.DM(references[k, 3:6])
            omega_error = X[10:13, k] - ca.DM(references[k, 10:13])
            delta_u = U[:, k] - prior
            # Match acados NONLINEAR_LS scaling: a stage is a 0.5-weighted
            # integral over one control interval, while the terminal term is
            # a 0.5-weighted endpoint cost.  Keeping this definition shared is
            # required for a meaningful backend objective comparison.
            total_cost += 0.5 * params.control_period_s * (
                params.position_cost * ca.dot(position_error, position_error)
                + params.velocity_cost * ca.dot(velocity_error, velocity_error)
                + params.attitude_cost * ca.dot(q_error, q_error)
                + params.angular_velocity_cost * ca.dot(omega_error, omega_error)
                + params.duty_cost * ca.dot(U[:, k], U[:, k])
                + params.duty_rate_cost * ca.dot(delta_u, delta_u)
            )
            if params.enable_fov:
                depth_lower, depth_upper, cone = casadi_fov_terms(
                    X[:, k], fov_targets[k], params
                )
                opti.subject_to(depth_lower >= 0.0)
                opti.subject_to(depth_upper >= 0.0)
                opti.subject_to(cone >= 0.0)
            opti.subject_to(ca.dot(q_error, q_error) <= params.attitude_limit_rad**2)

        opti.subject_to(
            opti.bounded(
                -params.velocity_limit_m_s,
                X[3:6, N],
                params.velocity_limit_m_s,
            )
        )
        opti.subject_to(
            opti.bounded(
                -params.angular_velocity_limit_rad_s,
                X[10:13, N],
                params.angular_velocity_limit_rad_s,
            )
        )
        h_terminal = ca.dot(
            X[0:3, N] - ca.DM(params.koz_center_L_m),
            X[0:3, N] - ca.DM(params.koz_center_L_m),
        ) - params.koz_radius_m**2
        opti.subject_to(h_terminal >= 0.0)
        terminal_error = casadi_quaternion_error_vector(
            ca.DM(references[N, 6:10]), X[6:10, N]
        )
        opti.subject_to(ca.dot(terminal_error, terminal_error) <= params.attitude_limit_rad**2)
        if params.enable_fov:
            depth_lower, depth_upper, cone = casadi_fov_terms(
                X[:, N], fov_targets[N], params
            )
            opti.subject_to(depth_lower >= 0.0)
            opti.subject_to(depth_upper >= 0.0)
            opti.subject_to(cone >= 0.0)
        terminal_position_error = X[0:3, N] - ca.DM(references[N, 0:3])
        terminal_velocity_error = X[3:6, N] - ca.DM(references[N, 3:6])
        terminal_omega_error = X[10:13, N] - ca.DM(references[N, 10:13])
        total_cost += 0.5 * params.terminal_cost_scale * (
            params.position_cost * ca.dot(terminal_position_error, terminal_position_error)
            + params.velocity_cost * ca.dot(terminal_velocity_error, terminal_velocity_error)
            + params.attitude_cost * ca.dot(terminal_error, terminal_error)
            + params.angular_velocity_cost * ca.dot(terminal_omega_error, terminal_omega_error)
        )
        opti.minimize(total_cost)
        opti.set_initial(X, state_guess.T)
        opti.set_initial(U, duty_guess.T)
        opti.solver(
            "ipopt",
            {"expand": True},
            {
                "max_iter": self.max_iterations,
                "tol": self.tolerance,
                "acceptable_tol": max(self.tolerance * 10.0, 1.0e-5),
                "print_level": 0,
                "sb": "yes",
            },
        )

        status = "failed"
        status_code = -1
        iterations = 0
        kkt = float("nan")
        objective = float("nan")
        failure_reason: str | None = None
        try:
            solution = opti.solve()
            stats = opti.stats()
            return_status = str(stats.get("return_status", ""))
            status = "success" if bool(stats.get("success", False)) else "failed"
            status_code = 0 if status == "success" else -1
            iterations, kkt = _stats_iterations(stats)
            objective = float(solution.value(opti.f))
            solved_state = np.asarray(solution.value(X), dtype=np.float64).T
            solved_duty = np.asarray(solution.value(U), dtype=np.float64).T
            if status != "success":
                failure_reason = return_status or "IPOPT did not report success"
        except Exception as exc:  # IPOPT may throw for infeasible or unavailable plugins.
            failure_reason = str(exc)
            solved_state = np.tile(x_initial, (N + 1, 1))
            solved_duty = np.zeros((N, 16), dtype=np.float64)

        solved_wrench = np.asarray(solved_duty @ B.T, dtype=np.float64)
        accepted, margins, validation_reason = validate_solution(
            solved_state, solved_duty, params, previous
        )
        if status != "success":
            quality = "failed"
        elif accepted:
            quality = "accepted"
        else:
            quality = "rejected"
            failure_reason = validation_reason or "independent quality gate rejected solution"
        return ExpertResult(
            status=status,
            solver_status_code=status_code,
            iterations=iterations,
            kkt=kkt,
            runtime_s=time.perf_counter() - start,
            objective=objective,
            constraint_margins=margins,
            slacks={"recovery": 0.0},
            allocation_residual=0.0,
            warm_start={
                "x": np.ascontiguousarray(solved_state),
                "u": np.ascontiguousarray(solved_duty),
                "previous_duty": np.ascontiguousarray(solved_duty[0]),
            },
            predicted_state=np.ascontiguousarray(solved_state),
            predicted_duty=np.ascontiguousarray(solved_duty),
            predicted_wrench=np.ascontiguousarray(solved_wrench),
            actual_duty=np.ascontiguousarray(solved_duty[0]),
            actual_wrench=np.ascontiguousarray(solved_wrench[0]),
            quality_verdict=quality,
            failure_reason=failure_reason,
            sample_type="normal" if quality == "accepted" else "failed",
            action_blocks=np.ascontiguousarray(
                solved_duty[list(params.action_block_starts)]
            ),
        )
