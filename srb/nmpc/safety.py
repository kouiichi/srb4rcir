"""Runtime safety filtering and 16-channel wrench allocation.

The filter is deliberately a small QP around the student's *raw body wrench*.
It never changes the NMPC label: the returned executed wrench is recomputed as
``B @ duty16`` and the residual is reported to the caller.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt

from .constraints import fov_margins, koz_terms
from .dynamics import camera_target_point_L, reference_at
from .quaternion import quaternion_to_rotation
from .rcs import rcs_matrix_from_task
from .types import AllocationResult, FloatArray, TaskParams


def _finite_vector(value: npt.ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
    return np.ascontiguousarray(result)


def _safe_clip_pair(duty: FloatArray, params: TaskParams) -> FloatArray:
    """Return a deterministic actuator-safe duty vector for fallback execution."""

    result = np.clip(np.asarray(duty, dtype=np.float64), 0.0, 1.0).copy()
    for pair in range(8):
        pair_slice = slice(2 * pair, 2 * pair + 2)
        pair_sum = float(np.sum(result[pair_slice]))
        if pair_sum > params.reverse_pair_limit and pair_sum > 0.0:
            result[pair_slice] *= params.reverse_pair_limit / pair_sum
    return np.ascontiguousarray(result)


class SafetyAllocator:
    """OSQP safety filter for a student body wrench at one 20 Hz timestamp.

    ``wrench_student`` has shape ``(6,)`` and float64 SI units ``[N, N m]`` in
    the service body frame.  ``state_estimate`` has shape ``(13,)`` and is a
    target-relative LVLH/body State13 measured at the same timestamp;
    ``previous_duty`` has shape ``(16,)`` and is the duty actually executed at
    the preceding 20 Hz sample.  The result contains the projected duty, the
    exact executed ``B @ duty16`` wrench, residual, hard margins, OSQP status,
    and runtime.  No future state, solver diagnostic, or reference trajectory
    is consumed.

    KOZ and actuator constraints are hard.  Corridor/FOV recovery slacks are
    intentionally not introduced in this runtime QP: if no safe feasible
    solution exists, a predefined hold/zero action is returned with a failure
    status and must not be treated as a safe expert label.
    """

    def __init__(
        self,
        task_params: TaskParams | None = None,
        *,
        residual_tolerance: float = 1.0e-4,
        safety_margin_tolerance: float = 1.0e-9,
        regularization: float = 1.0e-8,
    ) -> None:
        if residual_tolerance <= 0.0 or not np.isfinite(residual_tolerance):
            raise ValueError("residual_tolerance must be finite and positive")
        if safety_margin_tolerance < 0.0 or not np.isfinite(safety_margin_tolerance):
            raise ValueError("safety_margin_tolerance must be finite and non-negative")
        if regularization <= 0.0 or not np.isfinite(regularization):
            raise ValueError("regularization must be finite and positive")
        self.task_params = task_params or TaskParams()
        self.residual_tolerance = float(residual_tolerance)
        self.safety_margin_tolerance = float(safety_margin_tolerance)
        self.regularization = float(regularization)
        self.last_diagnostics: dict[str, Any] = {}
        self._osqp_problem: Any | None = None
        self._osqp_signature: tuple[object, ...] | None = None
        self._cached_B: FloatArray | None = None
        self._cached_B_pinv: FloatArray | None = None
        self._cached_weighted_B: FloatArray | None = None
        self._cached_H: FloatArray | None = None
        self._cached_A: Any | None = None

    @staticmethod
    def _ecbf_affine_row(
        state: FloatArray, params: TaskParams, B: FloatArray
    ) -> tuple[float, FloatArray, float]:
        """Return ``ecbf(state,duty)=base+coeff@duty`` and current ``h``.

        The HCW acceleration and rigid-body equations are affine in duty.  A
        deterministic zero/one evaluation therefore gives the exact row used
        by the QP without finite-difference tuning or a separate approximate
        safety model.
        """

        zero = np.zeros(16, dtype=np.float64)
        h, _, _, base_ecbf = koz_terms(state, zero, params, rcs_matrix=B)
        # Only the translational force enters h_ddot.  The HCW acceleration is
        # affine in F_L=R_LB F_B, so this is the exact ECBF coefficient row,
        # not a finite-difference approximation and not an alternate model.
        offset = state[:3] - params.koz_center_L_m
        rotation_LB = quaternion_to_rotation(state[6:10])
        coeff = 2.0 * offset @ (rotation_LB @ B[:3, :] / params.mass_kg)
        return float(base_ecbf), np.ascontiguousarray(coeff), float(h)

    def _prepare_qp(self, params: TaskParams, B: FloatArray) -> None:
        """Create one reusable OSQP workspace for fixed actuator parameters."""

        signature = params.signature()
        if self._osqp_problem is not None and self._osqp_signature == signature:
            return
        import osqp
        from scipy import sparse

        row_scale = np.diag(
            [1.0 / params.tmax_N] * 3 + [1.0 / (0.30 * params.tmax_N)] * 3
        )
        weighted_B = row_scale @ B
        H = weighted_B.T @ weighted_B + self.regularization * np.eye(16)
        pair_rows = sparse.lil_matrix((8, 16), dtype=np.float64)
        for pair in range(8):
            pair_rows[pair, 2 * pair : 2 * pair + 2] = 1.0
        # Store a structural nonzero in every ECBF column.  The values are
        # updated in-place on every call, including coefficients that happen
        # to be exactly zero for a particular state.
        ecbf_structure = sparse.csc_matrix(
            (np.ones(16), (np.zeros(16, dtype=np.int64), np.arange(16))),
            shape=(1, 16),
        )
        A = sparse.vstack(
            (sparse.eye(16, format="csc"), sparse.eye(16, format="csc"), pair_rows.tocsc(), ecbf_structure),
            format="csc",
        )
        lower = np.zeros(41, dtype=np.float64)
        upper = np.zeros(41, dtype=np.float64)
        lower[32:40] = -np.inf
        upper[:16] = 1.0
        upper[16:32] = 1.0
        upper[32:40] = params.reverse_pair_limit
        upper[40] = np.inf
        problem = osqp.OSQP()
        problem.setup(
            P=sparse.csc_matrix((H + H.T) * 0.5),
            q=np.zeros(16, dtype=np.float64),
            A=A,
            l=lower,
            u=upper,
            eps_abs=1.0e-8,
            eps_rel=1.0e-8,
            max_iter=2000,
            polish=True,
            verbose=False,
        )
        self._osqp_problem = problem
        self._osqp_signature = signature
        self._cached_B = np.ascontiguousarray(B)
        self._cached_B_pinv = np.ascontiguousarray(np.linalg.pinv(B))
        self._cached_weighted_B = np.ascontiguousarray(weighted_B)
        self._cached_H = np.ascontiguousarray(H)
        self._cached_A = A

    @staticmethod
    def _current_hard_state_check(
        state: FloatArray,
        params: TaskParams,
        timestamp_s: float | None,
    ) -> str | None:
        quaternion_norm = float(np.linalg.norm(state[6:10]))
        if abs(quaternion_norm - 1.0) > 1.0e-6:
            return "state quaternion norm is outside the hard tolerance"
        if np.max(np.abs(state[3:6])) > params.velocity_limit_m_s + 1.0e-12:
            return "state velocity limit already violated"
        if np.max(np.abs(state[10:13])) > params.angular_velocity_limit_rad_s + 1.0e-12:
            return "state angular-velocity limit already violated"
        h = float(
            np.dot(state[:3] - params.koz_center_L_m, state[:3] - params.koz_center_L_m)
            - params.koz_radius_m**2
        )
        if h < -1.0e-12:
            return "state is already inside the hard KOZ"
        if params.enable_fov:
            if timestamp_s is None:
                return "FOV is enabled but the current timestamp was not supplied"
            target = camera_target_point_L(params, reference_at(params, timestamp_s))
            depth, upper, cone = fov_margins(state, target, params)
            if min(depth, upper, cone) < -1.0e-12:
                return "state already violates a hard FOV/depth constraint"
        return None

    def _fallback(
        self,
        wrench: FloatArray,
        state: FloatArray,
        previous: FloatArray,
        params: TaskParams,
        B: FloatArray,
        reason: str,
        start: float,
        status_code: int = -1,
    ) -> AllocationResult:
        """Execute and report the predefined hold/zero fallback."""

        # Holding the previous command is preferred if it is actuator-safe;
        # otherwise zero duty is the deterministic safe actuator command.  A
        # failed ECBF check remains failed even if neither fallback is safe.
        candidates = (_safe_clip_pair(previous, params), np.zeros(16, dtype=np.float64))
        selected = candidates[0]
        selected_ecbf = koz_terms(state, selected, params)[3]
        if selected_ecbf < -self.safety_margin_tolerance:
            selected = candidates[1]
            selected_ecbf = koz_terms(state, selected, params)[3]
        executed = np.ascontiguousarray(B @ selected)
        residual = float(np.linalg.norm(executed - wrench))
        margins = {
            "koz_h": float(koz_terms(state, selected, params)[0]),
            "koz_ecbf": float(selected_ecbf),
            "duty_lower": float(np.min(selected)),
            "duty_upper": float(np.min(1.0 - selected)),
            "rate": float(np.min(params.max_duty_delta - np.abs(selected - previous))),
            "reverse_pair": float(
                min(
                    params.reverse_pair_limit - selected[2 * i] - selected[2 * i + 1]
                    for i in range(8)
                )
            ),
            "residual_tolerance": float(self.residual_tolerance - residual),
        }
        self.last_diagnostics = {
            "status": "failed",
            "fallback": "hold" if np.shares_memory(selected, candidates[0]) else "zero",
            "reason": reason,
            "hard_fallback_ecbf": float(selected_ecbf),
        }
        return AllocationResult(
            status="failed",
            duty16=np.ascontiguousarray(selected),
            wrench6_exec=executed,
            residual=residual,
            saturated=True,
            intervention=np.ascontiguousarray(selected - np.linalg.pinv(B) @ wrench),
            constraint_margins=margins,
            runtime_s=time.perf_counter() - start,
            solver_status_code=status_code,
            error_message=reason,
        )

    def filter_and_allocate(
        self,
        wrench_student: npt.ArrayLike,
        state_estimate: npt.ArrayLike,
        previous_duty: npt.ArrayLike | None = None,
        task_params: TaskParams | None = None,
        timestamp_s: float | None = None,
    ) -> AllocationResult:
        """Filter one raw student wrench and allocate it to executable duties.

        Args:
            wrench_student: Shape ``(6,)`` body wrench in ``[N, N m]``.
            state_estimate: Shape ``(13,)`` State13 at the current timestamp.
            previous_duty: Shape ``(16,)`` duty executed at the prior 20 Hz
                timestamp.  Omitted means all-zero previous duty.
            task_params: Optional validated episode parameters; defaults to
                the parameters supplied at construction.
            timestamp_s: Current episode timestamp in seconds.  Required when
                FOV constraints are enabled so the moving target point is
                evaluated at the same sample; it is never a future quantity.

        Returns:
            :class:`AllocationResult`; ``wrench6_exec`` is always recomputed
            from the returned duty.  ``status='failed'`` means the fallback
            action was used and the sample requires failure isolation.

        Raises:
            ValueError: for malformed/non-finite inputs.  Infeasibility and
                OSQP errors are returned as an explicit failed result.
        """

        start = time.perf_counter()
        params = task_params or self.task_params
        wrench = _finite_vector(wrench_student, (6,), "wrench_student")
        state = _finite_vector(state_estimate, (13,), "state_estimate")
        if previous_duty is None:
            previous = np.zeros(16, dtype=np.float64)
        else:
            previous = _finite_vector(previous_duty, (16,), "previous_duty")
        if np.any(previous < 0.0) or np.any(previous > 1.0):
            raise ValueError("previous_duty must lie in [0,1]")

        B = rcs_matrix_from_task(params)
        if timestamp_s is not None and (not np.isfinite(timestamp_s) or timestamp_s < 0.0):
            raise ValueError("timestamp_s must be finite and non-negative")
        state_error = self._current_hard_state_check(state, params, timestamp_s)
        if state_error is not None:
            return self._fallback(wrench, state, previous, params, B, state_error, start)

        try:
            import osqp
            from scipy import sparse

            self._prepare_qp(params, B)
            assert self._osqp_problem is not None
            assert self._cached_weighted_B is not None
            assert self._cached_A is not None
            assert self._cached_B_pinv is not None
            weighted_B = self._cached_weighted_B
            row_scale = np.diag(
                [1.0 / params.tmax_N] * 3 + [1.0 / (0.30 * params.tmax_N)] * 3
            )
            weighted_wrench = row_scale @ wrench
            q = -(weighted_B.T @ weighted_wrench + self.regularization * previous)
            base_ecbf, ecbf_coeff, h = self._ecbf_affine_row(state, params, B)
            lower = np.concatenate(
                (
                    np.zeros(16),
                    previous - params.max_duty_delta,
                    np.full(8, -np.inf),
                    np.array([-base_ecbf]),
                )
            )
            upper = np.concatenate(
                (
                    np.ones(16),
                    previous + params.max_duty_delta,
                    np.full(8, params.reverse_pair_limit),
                    np.array([np.inf]),
                )
            )
            # Replace the cached ECBF row values without changing the CSC
            # sparsity pattern, then update only the per-state QP vectors.
            for column in range(16):
                begin, end = self._cached_A.indptr[column], self._cached_A.indptr[column + 1]
                rows = self._cached_A.indices[begin:end]
                row_offset = np.flatnonzero(rows == 40)
                if row_offset.size != 1:
                    raise RuntimeError("cached OSQP ECBF sparsity pattern is invalid")
                self._cached_A.data[begin + int(row_offset[0])] = ecbf_coeff[column]
            self._osqp_problem.update(q=q, l=lower, u=upper, Ax=self._cached_A.data)
            solution = self._osqp_problem.solve()
            status_code = int(solution.info.status_val)
            if solution.x is None or status_code not in (1, 2):
                return self._fallback(
                    wrench,
                    state,
                    previous,
                    params,
                    B,
                    f"OSQP status {solution.info.status}",
                    start,
                    status_code,
                )
            duty = np.ascontiguousarray(np.clip(solution.x, 0.0, 1.0))
            executed = np.ascontiguousarray(B @ duty)
            residual = float(np.linalg.norm(executed - wrench))
            pair_margins = [
                params.reverse_pair_limit - float(duty[2 * i] + duty[2 * i + 1])
                for i in range(8)
            ]
            ecbf_value = float(base_ecbf + ecbf_coeff @ duty)
            margins = {
                "koz_h": h,
                "koz_ecbf": ecbf_value,
                "duty_lower": float(np.min(duty)),
                "duty_upper": float(np.min(1.0 - duty)),
                "rate": float(np.min(params.max_duty_delta - np.abs(duty - previous))),
                "reverse_pair": float(min(pair_margins)),
                "residual_tolerance": float(self.residual_tolerance - residual),
                "intervention_norm": float(np.linalg.norm(duty - previous)),
            }
            if residual > self.residual_tolerance:
                status = "residual_exceeded"
            elif np.linalg.norm(duty - np.linalg.pinv(B) @ wrench) > 1.0e-7:
                status = "intervened"
            else:
                status = "solved"
            self.last_diagnostics = {
                "status": status,
                "osqp_status": str(solution.info.status),
                "iterations": int(solution.info.iter),
                "objective": float(solution.info.obj_val),
                "state_timestamp": "current_20hz_sample",
            }
            return AllocationResult(
                status=status,
                duty16=duty,
                wrench6_exec=executed,
                residual=residual,
                saturated=bool(
                    np.any(duty <= 1.0e-8)
                    or np.any(duty >= 1.0 - 1.0e-8)
                    or margins["rate"] <= 1.0e-8
                    or margins["reverse_pair"] <= 1.0e-8
                ),
                intervention=np.ascontiguousarray(duty - self._cached_B_pinv @ wrench),
                constraint_margins=margins,
                runtime_s=time.perf_counter() - start,
                solver_status_code=status_code,
                error_message=(
                    "executable residual exceeded tolerance"
                    if residual > self.residual_tolerance
                    else None
                ),
            )
        except Exception as exc:
            return self._fallback(
                wrench,
                state,
                previous,
                params,
                B,
                f"safety QP exception: {exc}",
                start,
            )
