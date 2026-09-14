"""acados/HPIPM production NMPC expert backend."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
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
from .physical import pack_physical_parameters, structural_signature


def _as_state(value: npt.ArrayLike) -> FloatArray:
    state = np.asarray(value, dtype=np.float64)
    if state.shape != (13,) or not np.all(np.isfinite(state)):
        raise ValueError("x0 must be a finite shape-(13,) float64 array")
    result = state.copy()
    norm = float(np.linalg.norm(result[6:10]))
    if norm <= 1.0e-12:
        raise ValueError("x0 quaternion norm is too small")
    result[6:10] /= norm
    return np.ascontiguousarray(result)


def _previous_duty(warm_start: Any) -> FloatArray:
    if isinstance(warm_start, ExpertResult):
        value = warm_start.actual_duty
    elif isinstance(warm_start, Mapping) and "previous_duty" in warm_start:
        value = warm_start["previous_duty"]
    else:
        value = np.zeros(16, dtype=np.float64)
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (16,) or not np.all(np.isfinite(result)):
        raise ValueError("warm_start previous_duty must have shape (16,) and be finite")
    if np.any(result < 0.0) or np.any(result > 1.0):
        raise ValueError("warm_start previous_duty must lie in [0,1]")
    return np.ascontiguousarray(result)


def _warm_guess(
    task_params: TaskParams,
    x0: FloatArray,
    previous: FloatArray,
    warm_start: Any,
    rcs_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    N = task_params.horizon_steps
    duty_guess = np.zeros((N, 16), dtype=np.float64)
    if isinstance(warm_start, ExpertResult):
        candidate_x = warm_start.predicted_state
        candidate_u = warm_start.predicted_duty
    elif isinstance(warm_start, Mapping):
        candidate_x = warm_start.get("x")
        candidate_u = warm_start.get("u")
    else:
        candidate_x = candidate_u = None
    state_guess: FloatArray
    if candidate_x is not None:
        candidate = np.asarray(candidate_x, dtype=np.float64)
        if candidate.shape == (N + 1, 13) and np.all(np.isfinite(candidate)):
            # A shifted warm trajectory is a much better initial point than a
            # fresh zero rollout.  Its first node is replaced by the current
            # measured state; acados then enforces the nonlinear manifold.
            state_guess = np.ascontiguousarray(candidate.copy())
            state_guess[0] = x0
        else:
            candidate_x = None
    if candidate_x is None:
        state_guess = np.empty((N + 1, 13), dtype=np.float64)
        state_guess[0] = x0
        for index in range(N):
            state_guess[index + 1] = rk4_step(
                state_guess[index], duty_guess[index], task_params, rcs_matrix=rcs_matrix
            )
    if candidate_u is not None:
        candidate = np.asarray(candidate_u, dtype=np.float64)
        if candidate.shape == duty_guess.shape and np.all(np.isfinite(candidate)):
            duty_guess[:-1] = candidate[1:]
            duty_guess[-1] = candidate[-1]
    # The augmented previous-duty state must be consistent with the proposed
    # control sequence; this is what makes the rate bound a true hard bound.
    duty_guess = np.clip(duty_guess, 0.0, 1.0)
    return np.ascontiguousarray(state_guess), np.ascontiguousarray(duty_guess)


def _scalar_stat(solver: Any, name: str, default: float = float("nan")) -> float:
    try:
        value = np.asarray(solver.get_stats(name), dtype=np.float64)
        return float(value.reshape(-1)[-1])
    except (AttributeError, TypeError, ValueError, IndexError):
        return default


def _max_stat(solver: Any, name: str, default: float = float("nan")) -> float:
    try:
        value = np.asarray(solver.get_stats(name), dtype=np.float64)
        return float(np.max(np.abs(value)))
    except (AttributeError, TypeError, ValueError, IndexError):
        return default


class AcadosExpert:
    """acados SQP/HPIPM expert with hard actuator-rate memory.

    The public physical state remains State13.  Internally the generated
    acados model augments it with the preceding 16 duties (29 solver states),
    so ``u_k-u_{k-1}`` and reverse-pair constraints are enforced inside the
    optimization rather than repaired after the solve.  The model uses the
    same normalized five-substep RK4 map, 20 intervals, scalar-first
    quaternion convention, and body-frame 16-channel RCS matrix as the IPOPT
    oracle.  ``solve`` returns ``(21,13)``, ``(20,16)``, and ``(20,6)`` arrays;
    `status` is numerical solver status while `quality_verdict` is an
    independent finite/dynamics/hard-margin gate.  Solver generation requires
    the Isaac Sim Python environment with CasADi, acados_template, and
    ``ACADOS_SOURCE_DIR`` configured by ``srb_0.0.5_env.sh``.
    """

    solver_state_dim = 29

    def __init__(
        self,
        task_params: TaskParams | None = None,
        *,
        code_export_root: str | Path | None = None,
        nlp_solver_type: str = "SQP_RTI",
        max_iterations: int = 1,
        qp_solver_cond_N: int = 5,
        qp_solver_iter_max: int = 100,
        build_on_init: bool = False,
    ) -> None:
        if nlp_solver_type not in ("SQP", "SQP_RTI"):
            raise ValueError("nlp_solver_type must be SQP or SQP_RTI")
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if qp_solver_cond_N <= 0 or qp_solver_iter_max <= 0:
            raise ValueError("qp_solver_cond_N and qp_solver_iter_max must be positive")
        self.task_params = task_params or TaskParams()
        self.code_export_root = (
            Path(code_export_root)
            if code_export_root is not None
            else Path(__file__).resolve().parents[2] / "third_party" / "acados" / "generated"
        )
        self.nlp_solver_type = nlp_solver_type
        self.max_iterations = int(max_iterations)
        self.qp_solver_cond_N = int(qp_solver_cond_N)
        self.qp_solver_iter_max = int(qp_solver_iter_max)
        self._solver: Any | None = None
        self._solver_signature: tuple[object, ...] | None = None
        self._reference_cache: FloatArray | None = None
        self._reference_signature: tuple[object, ...] | None = None
        self._numeric_rollout: Any | None = None
        if build_on_init:
            self._build_solver(self.task_params)

    @staticmethod
    def _ensure_acados_environment() -> Path:
        root = Path(
            os.environ.get(
                "ACADOS_SOURCE_DIR",
                Path(__file__).resolve().parents[2] / "third_party" / "acados",
            )
        ).resolve()
        if not (root / "lib" / "libacados.so").exists():
            raise RuntimeError(f"acados shared library not found under {root}")
        os.environ["ACADOS_SOURCE_DIR"] = str(root)
        library_path = str(root / "lib")
        current = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
        if library_path not in current:
            os.environ["LD_LIBRARY_PATH"] = ":".join([library_path, *current])
        return root

    def _build_solver(self, params: TaskParams) -> None:
        """Generate and compile one acados model for fixed physical parameters."""

        self._ensure_acados_environment()
        try:
            import casadi as ca
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except ModuleNotFoundError as exc:  # pragma: no cover - environment error
            raise RuntimeError(
                "CasADi and acados_template are required for AcadosExpert"
            ) from exc

        N = params.horizon_steps
        nx_phys = 13
        nx = self.solver_state_dim
        nu = 16
        # Keep the public TaskParams contract unchanged.  acados receives the
        # 13-state reference plus the moving target point as internal
        # parameters, allowing FOV constraints to use the same timestamped
        # point as the IPOPT oracle and runtime filter.
        np_model = 131
        x_aug = ca.SX.sym("x_aug", nx)
        u = ca.SX.sym("u", nu)
        p = ca.SX.sym("p", np_model)
        x_phys = x_aug[0:nx_phys]
        x_next_phys = casadi_rk4_map(x_phys, u, params, physical_parameters=p[16:])
        x_next = ca.vertcat(x_next_phys, u)

        model = AcadosModel()
        short_hash = hashlib.sha1(
            repr(
                (
                    'runtime_physics_v1', structural_signature(params),
                    self.nlp_solver_type,
                    self.max_iterations,
                    self.qp_solver_cond_N,
                    self.qp_solver_iter_max,
                )
            ).encode("utf-8")
        ).hexdigest()[:10]
        model.name = f"srb_nmpc_{short_hash}"
        model.x = x_aug
        model.u = u
        model.p = p
        model.disc_dyn_expr = x_next
        # Keep a separate CasADi numerical map for post-solve rollout and
        # independent consistency validation.  acados still owns the OCP
        # solve; this map only avoids running hundreds of Python RK4 helper
        # calls in the 20 Hz execution path.
        numeric_x = ca.SX.sym(f"{model.name}_numeric_x", nx_phys)
        numeric_u = ca.SX.sym(f"{model.name}_numeric_u", nu)
        numeric_physical = ca.SX.sym(f"{model.name}_numeric_physical", 115)
        numeric_map = ca.Function(
            f"{model.name}_numeric_map",
            [numeric_x, numeric_u, numeric_physical],
            [casadi_rk4_map(numeric_x, numeric_u, params, physical_parameters=numeric_physical)],
        )
        self._numeric_rollout = numeric_map.mapaccum(f"{model.name}_numeric_rollout", N)

        reference = p[0:13]
        target_point_L = p[13:16]
        q_error = casadi_quaternion_error_vector(reference[6:10], x_phys[6:10])
        position_error = x_phys[0:3] - reference[0:3]
        velocity_error = x_phys[3:6] - reference[3:6]
        omega_error = x_phys[10:13] - reference[10:13]
        # The augmented state stores the preceding duty vector.  Including
        # this difference in the running least-squares cost keeps the IPOPT
        # and acados objectives identical while the hard path constraint below
        # enforces the same rate/block semantics.
        duty_delta = u - x_aug[nx_phys:nx]
        y_expr = ca.vertcat(
            position_error,
            velocity_error,
            q_error,
            omega_error,
            u,
            duty_delta,
        )
        y_expr_e = ca.vertcat(position_error, velocity_error, q_error, omega_error)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = params.horizon_seconds
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.nlp_solver_type = self.nlp_solver_type
        ocp.solver_options.nlp_solver_max_iter = self.max_iterations
        ocp.solver_options.qp_solver_iter_max = self.qp_solver_iter_max
        ocp.solver_options.print_level = 0
        ocp.solver_options.tol = 1.0e-6
        ocp.solver_options.qp_solver_cond_N = min(self.qp_solver_cond_N, N)

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.model.cost_y_expr = y_expr
        ocp.cost.yref = np.zeros(3 + 3 + 3 + 3 + 16 + 16, dtype=np.float64)
        ocp.cost.W = np.diag(
            [
                *([params.position_cost] * 3),
                *([params.velocity_cost] * 3),
                *([params.attitude_cost] * 3),
                *([params.angular_velocity_cost] * 3),
                *([params.duty_cost] * 16),
                *([params.duty_rate_cost] * 16),
            ]
        )
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.model.cost_y_expr_e = y_expr_e
        ocp.cost.yref_e = np.zeros(12, dtype=np.float64)
        ocp.cost.W_e = np.diag(
            [
                *([params.position_cost * params.terminal_cost_scale] * 3),
                *([params.velocity_cost * params.terminal_cost_scale] * 3),
                *([params.attitude_cost * params.terminal_cost_scale] * 3),
                *([params.angular_velocity_cost * params.terminal_cost_scale] * 3),
            ]
        )

        # Duty bounds are represented as standard acados box constraints.
        ocp.constraints.lbu = np.zeros(nu, dtype=np.float64)
        ocp.constraints.ubu = np.ones(nu, dtype=np.float64)
        ocp.constraints.idxbu = np.arange(nu, dtype=np.int64)

        # Physical velocity/rate bounds and the augmented preceding-duty
        # memory.  The first stage is overwritten by x0 in solve().
        box_indices = np.array([3, 4, 5, 10, 11, 12, *range(13, 29)], dtype=np.int64)
        box_lower = np.array(
            [-params.velocity_limit_m_s] * 3
            + [-params.angular_velocity_limit_rad_s] * 3
            + [0.0] * 16,
            dtype=np.float64,
        )
        box_upper = np.array(
            [params.velocity_limit_m_s] * 3
            + [params.angular_velocity_limit_rad_s] * 3
            + [1.0] * 16,
            dtype=np.float64,
        )
        ocp.constraints.idxbx = box_indices
        ocp.constraints.lbx = box_lower
        ocp.constraints.ubx = box_upper
        ocp.constraints.idxbx_e = box_indices
        ocp.constraints.lbx_e = box_lower
        ocp.constraints.ubx_e = box_upper
        ocp.constraints.x0 = np.zeros(nx, dtype=np.float64)

        koz_h, koz_ecbf = casadi_koz_terms(x_phys, u, params, physical_parameters=p[16:])
        pair_sums = ca.vertcat(
            *[u[2 * pair] + u[2 * pair + 1] for pair in range(8)]
        )
        path_expr = ca.vertcat(koz_h, koz_ecbf, pair_sums, duty_delta)
        path_lower = np.concatenate(
            (
                np.array([0.0, 0.0]),
                np.full(8, -1.0e10),
                np.full(16, -params.max_duty_delta),
            )
        )
        path_upper = np.concatenate(
            (
                np.array([1.0e10, 1.0e10]),
                np.full(8, params.reverse_pair_limit),
                np.full(16, params.max_duty_delta),
            )
        )
        if params.enable_fov:
            fov_path = ca.vertcat(*casadi_fov_terms(x_phys, target_point_L, params))
            path_expr = ca.vertcat(path_expr, fov_path)
            path_lower = np.concatenate((path_lower, np.zeros(3)))
            path_upper = np.concatenate((path_upper, np.full(3, 1.0e10)))
        model.con_h_expr = path_expr
        # acados treats the initial shooting node as a separate constraint
        # family.  Register the same hard path constraints there as well;
        # otherwise stage 0 has nh_0=0 even though later stages have nh=26,
        # and the first control interval would escape the rate/KOZ checks.
        model.con_h_expr_0 = path_expr
        ocp.constraints.lh = path_lower
        ocp.constraints.uh = path_upper
        ocp.constraints.lh_0 = path_lower.copy()
        ocp.constraints.uh_0 = path_upper.copy()

        h_terminal = ca.dot(
            x_phys[0:3] - ca.DM(params.koz_center_L_m),
            x_phys[0:3] - ca.DM(params.koz_center_L_m),
        ) - params.koz_radius_m**2
        terminal_expr = h_terminal
        terminal_lower = np.array([0.0], dtype=np.float64)
        terminal_upper = np.array([1.0e10], dtype=np.float64)
        if params.enable_fov:
            fov_terminal = ca.vertcat(*casadi_fov_terms(x_phys, target_point_L, params))
            terminal_expr = ca.vertcat(terminal_expr, fov_terminal)
            terminal_lower = np.concatenate((terminal_lower, np.zeros(3)))
            terminal_upper = np.concatenate((terminal_upper, np.full(3, 1.0e10)))
        model.con_h_expr_e = terminal_expr
        ocp.constraints.lh_e = terminal_lower
        ocp.constraints.uh_e = terminal_upper

        ocp.parameter_values = np.r_[np.zeros(16), pack_physical_parameters(params)]
        export_dir = self.code_export_root / model.name
        export_dir.mkdir(parents=True, exist_ok=True)
        ocp.code_gen_options.code_export_directory = str(export_dir)
        # Keep generated metadata beside the compiled solver instead of
        # dropping a transient ``*_ocp.json`` in the repository root.
        json_file = export_dir / f"{model.name}_ocp.json"
        ocp.code_gen_options.json_file = str(json_file)
        self._solver = AcadosOcpSolver(ocp, verbose=False)
        self._solver_signature = structural_signature(params)

    def _ensure_solver(self, params: TaskParams) -> Any:
        if self._solver is None or self._solver_signature != structural_signature(params):
            self.task_params = params
            self._build_solver(params)
        return self._solver

    def solve(
        self,
        x0: npt.ArrayLike,
        task_params: TaskParams | None = None,
        warm_start: Any = None,
        *,
        time_s: float = 0.0,
        reference_states: npt.ArrayLike | None = None,
    ) -> ExpertResult:
        """Solve one trajectory on the global 20 Hz grid."""

        params = task_params or self.task_params
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError('time_s must be finite and nonnegative')
        x_initial = _as_state(x0)
        previous = _previous_duty(warm_start)
        solver = self._ensure_solver(params)
        start = time.perf_counter()
        N = params.horizon_steps
        if reference_states is not None:
            references = np.asarray(reference_states, dtype=np.float64)
            if references.shape != (N + 1, 13) or not np.all(np.isfinite(references)):
                raise ValueError("reference_states must have shape (N+1,13) and finite values")
            references = np.ascontiguousarray(references)
        elif self._reference_cache is None or self._reference_signature != (params.signature(), time_s):
            self._reference_cache = reference_trajectory(
                params,
                time_s + np.arange(N + 1, dtype=np.float64) * params.control_period_s,
            )
            self._reference_signature = (params.signature(), time_s)
            references = self._reference_cache
        else:
            references = self._reference_cache
        if params.enable_fov:
            fov_targets = np.vstack(
                [camera_target_point_L(params, reference) for reference in references]
            )
        else:
            fov_targets = np.zeros((N + 1, 3), dtype=np.float64)
        B = rcs_matrix_from_task(params)
        physical = pack_physical_parameters(params)
        state_guess, duty_guess = _warm_guess(params, x_initial, previous, warm_start, B)
        x0_aug = np.concatenate((x_initial, previous))
        independent_rollout: FloatArray | None = None

        try:
            solver.set(0, "lbx", x0_aug)
            solver.set(0, "ubx", x0_aug)
            for index in range(N):
                solver.set(
                    index,
                    "p",
                    np.concatenate((references[index], fov_targets[index], physical)),
                )
                # Hold the duty constant inside each K=8 action block.  At a
                # block boundary the normal rate limit is active; otherwise
                # the same augmented-memory constraint is an equality.
                path_lower = np.concatenate(
                    (
                        np.array([0.0, 0.0]),
                        np.full(8, -1.0e10),
                        np.full(16, -params.max_duty_delta),
                    )
                )
                path_upper = np.concatenate(
                    (
                        np.array([1.0e10, 1.0e10]),
                        np.full(8, params.reverse_pair_limit),
                        np.full(16, params.max_duty_delta),
                    )
                )
                if params.enable_fov:
                    path_lower = np.concatenate((path_lower, np.zeros(3)))
                    path_upper = np.concatenate((path_upper, np.full(3, 1.0e10)))
                if index not in set(params.action_block_starts):
                    # Path order is [KOZ(2), reverse-pairs(8), duty_delta(16),
                    # optional FOV(3)].  Do not use a tail slice here: with
                    # FOV enabled the last three entries are not duty rates.
                    duty_delta_start = 2 + 8
                    path_lower[duty_delta_start : duty_delta_start + 16] = 0.0
                    path_upper[duty_delta_start : duty_delta_start + 16] = 0.0
                solver.constraints_set(index, "lh", path_lower)
                solver.constraints_set(index, "uh", path_upper)
                solver.set(
                    index,
                    "x",
                    np.concatenate((state_guess[index], previous if index == 0 else duty_guess[index - 1])),
                )
                solver.set(index, "u", duty_guess[index])
            solver.set(N, "p", np.concatenate((references[N], fov_targets[N], physical)))
            solver.set(N, "x", np.concatenate((state_guess[N], duty_guess[N - 1])))
            status_code = int(solver.solve())
            solved_state_aug = np.vstack([np.asarray(solver.get(i, "x"), dtype=np.float64) for i in range(N + 1)])
            solved_duty = np.vstack([np.asarray(solver.get(i, "u"), dtype=np.float64) for i in range(N)])
            status = "success" if status_code == 0 else "failed"
            failure_reason = None if status == "success" else f"acados status {status_code}"
            if status == "success":
                # SQP_RTI returns a first-order trajectory that can have a
                # small nonlinear defect.  The public prediction is the
                # executable-duty RK4 rollout, so the independent quality
                # gate checks the same trajectory that will be applied.
                if self._numeric_rollout is not None:
                    rollout_tail = np.asarray(
                        self._numeric_rollout(x_initial, solved_duty.T, np.repeat(physical[:, None], N, axis=1)),
                        dtype=np.float64,
                    )
                    rollout = np.vstack((x_initial, rollout_tail.T))
                else:  # pragma: no cover - only if map construction is unavailable
                    rollout = np.empty((N + 1, 13), dtype=np.float64)
                    rollout[0] = x_initial
                    for index in range(N):
                        rollout[index + 1] = rk4_step(
                            rollout[index], solved_duty[index], params, rcs_matrix=B
                        )
                if rollout.shape != (N + 1, 13) or not np.all(np.isfinite(rollout)):
                    raise RuntimeError("independent CasADi rollout returned an invalid shape/value")
                independent_rollout = np.ascontiguousarray(rollout)
                solved_state_aug[:, :13] = rollout
                solved_state_aug[0, 13:] = previous
                solved_state_aug[1:, 13:] = solved_duty
        except Exception as exc:
            status_code = -1
            status = "failed"
            failure_reason = str(exc)
            solved_state_aug = np.zeros((N + 1, self.solver_state_dim), dtype=np.float64)
            solved_state_aug[:, :13] = np.tile(x_initial, (N + 1, 1))
            solved_state_aug[:, 13:] = np.vstack((previous, np.zeros((N, 16))))
            solved_duty = np.zeros((N, 16), dtype=np.float64)

        solved_state = np.ascontiguousarray(solved_state_aug[:, :13])
        solved_wrench = np.ascontiguousarray(solved_duty @ B.T)
        accepted, margins, validation_reason = validate_solution(
            solved_state,
            solved_duty,
            params,
            previous,
            reference_states=references,
            dynamics_rollout=independent_rollout,
        )
        if status != "success":
            quality = "failed"
        elif accepted:
            quality = "accepted"
        else:
            quality = "rejected"
            failure_reason = validation_reason or "independent quality gate rejected solution"

        kkt_terms = [
            _scalar_stat(solver, "res_stat_all"),
            _scalar_stat(solver, "res_eq_all"),
            _max_stat(solver, "residuals"),
        ]
        finite_kkt = [value for value in kkt_terms if np.isfinite(value)]
        kkt = max(finite_kkt) if finite_kkt else float("nan")
        iterations_value = _scalar_stat(solver, "sqp_iter", 0.0)
        iterations = int(iterations_value) if np.isfinite(iterations_value) else 0
        try:
            objective = float(solver.get_cost())
        except (AttributeError, TypeError, ValueError):
            objective = float("nan")
        if not np.isfinite(objective):
            objective = float("nan")
        runtime = time.perf_counter() - start
        return ExpertResult(
            status=status,
            solver_status_code=status_code,
            iterations=iterations,
            kkt=kkt,
            runtime_s=runtime,
            objective=objective,
            constraint_margins=margins,
            slacks={"recovery": 0.0},
            allocation_residual=0.0,
            warm_start={
                "x": solved_state.copy(),
                "u": solved_duty.copy(),
                "previous_duty": solved_duty[0].copy(),
            },
            predicted_state=solved_state,
            predicted_duty=np.ascontiguousarray(solved_duty),
            predicted_wrench=solved_wrench,
            actual_duty=np.ascontiguousarray(solved_duty[0]),
            actual_wrench=np.ascontiguousarray(solved_wrench[0]),
            quality_verdict=quality,
            failure_reason=failure_reason,
            sample_type="normal" if quality == "accepted" else "failed",
            action_blocks=np.ascontiguousarray(
                solved_duty[list(params.action_block_starts)]
            ),
        )

    def solve_batch(
        self,
        batch_x0: Sequence[npt.ArrayLike],
        batch_task_params: Sequence[TaskParams] | TaskParams,
    ) -> list[ExpertResult]:
        """Solve a batch serially with one reusable generated solver.

        The returned order matches ``batch_x0``.  Each result is independently
        validated; the method is serial because one generated acados solver is
        stateful and not thread-safe.  Inputs are timestamped at the same
        global 20 Hz origin and contain no future trajectory information.
        """

        if isinstance(batch_task_params, TaskParams):
            params_list = [batch_task_params] * len(batch_x0)
        else:
            params_list = list(batch_task_params)
            if len(params_list) != len(batch_x0):
                raise ValueError("batch_x0 and batch_task_params must have equal length")
        return [self.solve(x0, params) for x0, params in zip(batch_x0, params_list)]
