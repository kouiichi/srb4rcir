"""Six-degree-of-freedom 16-channel RCS geometry and allocation."""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import numpy.typing as npt

from .types import AllocationResult, FloatArray, RcsDuty16, TaskParams, Wrench6Body


_SQRT2 = float(np.sqrt(2.0))
_SQRT3 = float(np.sqrt(3.0))
_SQRT6 = float(np.sqrt(6.0))


def build_rcs_matrix(
    tmax_N: float = 0.1,
    com_B_m: npt.ArrayLike | None = None,
    rho_m: float = 0.30,
    delta_m: float = 0.05,
) -> FloatArray:
    """Construct the fixed ``B`` matrix for the 16 one-sided RCS thrusters.

    Args:
        tmax_N: Maximum force of one thruster in newtons.
        com_B_m: Center of mass in body coordinates, in meters.  ``None`` is
            the body origin.
        rho_m: Installation-bay radius in meters; fixed by the contract to
            0.30 m unless an explicit geometry test overrides it.
        delta_m: Tangential thrust-line offset in meters; fixed to 0.05 m.

    Returns:
        A float64 array of shape ``(6,16)``.  Column order is
        ``[11+,11-,12+,12-,...,42+,42-]`` and each column is
        ``[Tmax*fhat_B; (r_B-COM_B) x (Tmax*fhat_B)]`` in SI units.

    Raises:
        ValueError: if dimensions or geometric scalars are invalid.
    """

    if not np.isfinite(tmax_N) or tmax_N <= 0.0:
        raise ValueError("tmax_N must be finite and > 0")
    if not np.isclose(rho_m, 0.30) or not np.isclose(delta_m, 0.05):
        raise ValueError("the NMPC RCS contract fixes rho=0.30 m and delta=0.05 m")
    if com_B_m is None:
        com = np.zeros(3, dtype=np.float64)
    else:
        com = np.asarray(com_B_m, dtype=np.float64)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("com_B_m must be a finite shape-(3,) array")

    bay_directions = (
        np.array([1.0, 1.0, 1.0], dtype=np.float64) / _SQRT3,
        np.array([1.0, -1.0, -1.0], dtype=np.float64) / _SQRT3,
        np.array([-1.0, 1.0, -1.0], dtype=np.float64) / _SQRT3,
        np.array([-1.0, -1.0, 1.0], dtype=np.float64) / _SQRT3,
    )
    a0 = np.array([1.0, -1.0, 0.0], dtype=np.float64) / _SQRT2
    b0 = np.array([1.0, 1.0, -2.0], dtype=np.float64) / _SQRT6
    reflections = (
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )

    columns: list[FloatArray] = []
    for bay, (v_j, reflection) in enumerate(zip(bay_directions, reflections), start=1):
        for tangent, e_j in enumerate((reflection @ a0, reflection @ b0), start=1):
            for sign in ("+", "-"):
                if sign == "+":
                    position = rho_m * v_j - delta_m * e_j
                    force_direction = e_j
                else:
                    position = rho_m * v_j + delta_m * e_j
                    force_direction = -e_j
                force = tmax_N * force_direction
                torque = np.cross(position - com, force)
                columns.append(np.concatenate((force, torque)))

    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    if matrix.shape != (6, 16):
        raise RuntimeError(f"internal RCS construction produced {matrix.shape}")
    return np.ascontiguousarray(matrix)


def rcs_matrix_from_task(task_params: TaskParams) -> FloatArray:
    """Build the contract RCS matrix for a validated :class:`TaskParams`."""

    if task_params.rcs_matrix_B is not None:
        return np.asarray(task_params.rcs_matrix_B, dtype=np.float64).copy()
    return build_rcs_matrix(task_params.tmax_N, task_params.com_B_m)


def _rank(matrix: FloatArray) -> int:
    return int(np.linalg.matrix_rank(matrix, tol=1.0e-10))


def validate_rcs_geometry(
    matrix: npt.ArrayLike | None = None,
    *,
    tmax_N: float = 0.1,
    rho_m: float = 0.30,
    residual_tolerance: float = 1.0e-4,
    raise_on_failure: bool = False,
) -> dict[str, float | int | bool]:
    """Check rank, conditioning, specified faults, and reachable residuals.

    The condition number is computed after row scaling
    ``[F/Tmax, tau/(rho*Tmax)]``.  Fault checks remove every single thruster,
    each complete reverse pair, and each complete four-thruster bay.  The
    residual test uses deterministic random duties and is therefore a check of
    the matrix implementation rather than a claim about arbitrary unreachable
    wrenches.
    """

    B = (
        build_rcs_matrix(tmax_N=tmax_N, rho_m=rho_m)
        if matrix is None
        else np.asarray(matrix, dtype=np.float64)
    )
    if B.shape != (6, 16) or not np.all(np.isfinite(B)):
        raise ValueError("RCS matrix must be a finite shape-(6,16) array")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")

    scale = np.diag(
        [1.0 / tmax_N] * 3 + [1.0 / (rho_m * tmax_N)] * 3
    )
    singular_values = np.linalg.svd(scale @ B, compute_uv=False)
    condition = float(singular_values[0] / singular_values[-1])

    fault_ranks: list[int] = []
    fault_names: list[str] = []
    for index in range(16):
        fault_ranks.append(_rank(np.delete(B, index, axis=1)))
        fault_names.append(f"single_{index}")
    for pair in range(8):
        removed = [2 * pair, 2 * pair + 1]
        fault_ranks.append(_rank(np.delete(B, removed, axis=1)))
        fault_names.append(f"reverse_pair_{pair}")
    for bay in range(4):
        removed = list(range(4 * bay, 4 * bay + 4))
        fault_ranks.append(_rank(np.delete(B, removed, axis=1)))
        fault_names.append(f"bay_{bay + 1}")

    rng = np.random.default_rng(20260902)
    pseudoinverse = np.linalg.pinv(B)
    residuals: list[float] = []
    for _ in range(128):
        wrench = B @ rng.random(16)
        residuals.append(float(np.linalg.norm(B @ (pseudoinverse @ wrench) - wrench)))

    result: dict[str, float | int | bool] = {
        "rank": _rank(B),
        "normalized_condition": condition,
        "minimum_fault_rank": int(min(fault_ranks)),
        "maximum_reachable_residual": float(max(residuals)),
        "rank_ok": _rank(B) == 6,
        "condition_ok": condition <= 1.5,
        "faults_ok": min(fault_ranks) == 6,
        "residual_ok": max(residuals) < residual_tolerance,
    }
    result["fault_rank_min_name"] = fault_names[int(np.argmin(fault_ranks))]
    if raise_on_failure and not all(
        bool(result[key]) for key in ("rank_ok", "condition_ok", "faults_ok", "residual_ok")
    ):
        raise ValueError(f"RCS geometry validation failed: {result}")
    return result


def _as_wrench(value: npt.ArrayLike) -> FloatArray:
    wrench = np.asarray(value, dtype=np.float64)
    if wrench.shape != (6,) or not np.all(np.isfinite(wrench)):
        raise ValueError("wrench_student must be a finite shape-(6,) float64 array")
    return wrench


def allocate_wrench(
    wrench_student: npt.ArrayLike,
    task_params: TaskParams,
    previous_duty: npt.ArrayLike | None = None,
    *,
    residual_tolerance: float = 1.0e-4,
) -> AllocationResult:
    """Allocate one body wrench with bounded, rate-limited OSQP/QP duties.

    Args:
        wrench_student: Shape ``(6,)`` body-frame ``[N,N m]`` nominal input.
        task_params: Validated 20 Hz limits and actuator geometry.
        previous_duty: Optional shape ``(16,)`` duty at the preceding control
            timestamp.  If omitted, a zero duty is used and the rate bound is
            still applied.
        residual_tolerance: Maximum acceptable SI wrench residual.

    Returns:
        :class:`AllocationResult` with the exact executed wrench ``B@duty16``.
        A nonzero residual is reported and never relabeled as the requested
        student action.  ``status`` is ``solved`` only for a successful OSQP
        solve; fallback results remain explicitly marked.
    """

    start = time.perf_counter()
    wrench = _as_wrench(wrench_student)
    B = rcs_matrix_from_task(task_params)
    if previous_duty is None:
        previous = np.zeros(16, dtype=np.float64)
    else:
        previous = np.asarray(previous_duty, dtype=np.float64)
        if previous.shape != (16,) or not np.all(np.isfinite(previous)):
            raise ValueError("previous_duty must be a finite shape-(16,) array")
        if np.any(previous < 0.0) or np.any(previous > 1.0):
            raise ValueError("previous_duty must lie in [0,1]")

    unconstrained = np.linalg.pinv(B) @ wrench
    delta = task_params.max_duty_delta
    duty: FloatArray
    status = "fallback"
    solver_status_code = -1
    error_message: str | None = None

    try:
        import osqp
        from scipy import sparse

        # Normalize force and torque rows so the QP has comparable numerical
        # scales while retaining the physical residual calculation below.
        row_scale = np.diag(
            [1.0 / task_params.tmax_N] * 3
            + [1.0 / (0.30 * task_params.tmax_N)] * 3
        )
        weighted_B = row_scale @ B
        weighted_wrench = row_scale @ wrench
        regularization = 1.0e-8
        H = weighted_B.T @ weighted_B + regularization * np.eye(16)
        q = -(weighted_B.T @ weighted_wrench + regularization * previous)

        pair_rows = np.zeros((8, 16), dtype=np.float64)
        for pair in range(8):
            pair_rows[pair, 2 * pair : 2 * pair + 2] = 1.0
        constraint_matrix = sparse.csc_matrix(
            np.vstack((np.eye(16), np.eye(16), pair_rows))
        )
        lower = np.concatenate(
            (
                np.zeros(16),
                previous - delta,
                np.full(8, -np.inf),
            )
        )
        upper = np.concatenate(
            (
                np.ones(16),
                previous + delta,
                np.full(8, task_params.reverse_pair_limit),
            )
        )
        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix((H + H.T) * 0.5),
            q=q,
            A=constraint_matrix,
            l=lower,
            u=upper,
            eps_abs=1.0e-9,
            eps_rel=1.0e-9,
            max_iter=2000,
            polish=True,
            verbose=False,
        )
        solution = problem.solve()
        solver_status_code = int(solution.info.status_val)
        if solution.x is None or solver_status_code not in (1, 2):
            raise RuntimeError(f"OSQP status {solution.info.status}")
        duty = np.asarray(solution.x, dtype=np.float64)
        status = "solved"
    except Exception as exc:  # pragma: no cover - exercised only without OSQP
        error_message = str(exc)
        duty = np.clip(unconstrained, 0.0, 1.0)
        for pair in range(8):
            pair_slice = slice(2 * pair, 2 * pair + 2)
            pair_sum = float(np.sum(duty[pair_slice]))
            if pair_sum > task_params.reverse_pair_limit:
                duty[pair_slice] *= task_params.reverse_pair_limit / pair_sum
        duty = np.clip(previous + np.clip(duty - previous, -delta, delta), 0.0, 1.0)

    duty = np.ascontiguousarray(np.clip(duty, 0.0, 1.0))
    executed = np.ascontiguousarray(B @ duty)
    residual = float(np.linalg.norm(executed - wrench))
    intervention = np.ascontiguousarray(duty - unconstrained)
    pair_margins = [
        task_params.reverse_pair_limit - float(duty[2 * i] + duty[2 * i + 1])
        for i in range(8)
    ]
    margins = {
        "duty_lower": float(np.min(duty)),
        "duty_upper": float(np.min(1.0 - duty)),
        "rate": float(np.min(delta - np.abs(duty - previous))),
        "reverse_pair": float(min(pair_margins)),
        "residual_tolerance": float(residual_tolerance - residual),
    }
    saturated = bool(
        np.any(duty <= 1.0e-8)
        or np.any(duty >= 1.0 - 1.0e-8)
        or margins["rate"] <= 1.0e-8
        or margins["reverse_pair"] <= 1.0e-8
    )
    if residual >= residual_tolerance and status == "solved":
        status = "residual_exceeded"
    return AllocationResult(
        status=status,
        duty16=duty,
        wrench6_exec=executed,
        residual=residual,
        saturated=saturated,
        intervention=intervention,
        constraint_margins=margins,
        runtime_s=time.perf_counter() - start,
        solver_status_code=solver_status_code,
        error_message=error_message,
    )
