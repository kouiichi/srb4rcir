"""Hard-constraint expressions and independent post-solve validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .dynamics import (
    camera_target_point_L,
    casadi_continuous_dynamics,
    cw_acceleration,
    reference_trajectory,
    rk4_step,
)
from .quaternion import quaternion_to_rotation, shortest_quaternion_error
from .rcs import rcs_matrix_from_task
from .types import FloatArray, TaskParams


def _as_trajectory(
    predicted_state: npt.ArrayLike, predicted_duty: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    states = np.asarray(predicted_state, dtype=np.float64)
    duties = np.asarray(predicted_duty, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 13:
        raise ValueError(f"predicted_state must have shape (N+1,13), got {states.shape}")
    if duties.ndim != 2 or duties.shape != (states.shape[0] - 1, 16):
        raise ValueError(
            f"predicted_duty must have shape ({states.shape[0] - 1},16), got {duties.shape}"
        )
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(duties)):
        raise ValueError("predicted state and duty arrays must contain finite values")
    return np.ascontiguousarray(states), np.ascontiguousarray(duties)


def koz_terms(
    state: npt.ArrayLike,
    duty16: npt.ArrayLike,
    task_params: TaskParams,
    *,
    rcs_matrix: FloatArray | None = None,
) -> tuple[float, float, float, float]:
    """Return ``h``, ``h_dot``, ``h_ddot`` and ECBF margin for one stage.

    ``h=||r-r_koz||^2-r_koz^2`` has units m².  The returned ECBF margin is
    ``h_ddot+k1*h_dot+k0*h-safety_margin`` and must be non-negative.  Control
    enters through the HCW acceleration calculated from the executable RCS
    duty, so this check cannot accidentally validate an unallocated wrench.
    """

    x = np.asarray(state, dtype=np.float64)
    u = np.asarray(duty16, dtype=np.float64)
    if x.shape != (13,) or u.shape != (16,):
        raise ValueError("state and duty must have shapes (13,) and (16,)")
    offset = x[:3] - task_params.koz_center_L_m
    h = float(np.dot(offset, offset) - task_params.koz_radius_m**2)
    h_dot = float(2.0 * np.dot(offset, x[3:6]))
    matrix = rcs_matrix_from_task(task_params) if rcs_matrix is None else np.asarray(rcs_matrix, dtype=np.float64)
    if matrix.shape != (6, 16) or not np.all(np.isfinite(matrix)):
        raise ValueError("rcs_matrix must be a finite shape-(6,16) array")
    acceleration = cw_acceleration(x[:3], x[3:6], task_params.mean_motion_rad_s)
    acceleration += quaternion_to_rotation(x[6:10]) @ (matrix[:3] @ u) / task_params.mass_kg
    h_ddot = float(2.0 * np.dot(x[3:6], x[3:6]) + 2.0 * np.dot(offset, acceleration))
    ecbf = (
        h_ddot
        + task_params.koz_k1_s * h_dot
        + task_params.koz_k0_s2 * h
        - task_params.koz_safety_margin_m2_s2
    )
    return h, h_dot, h_ddot, float(ecbf)


def fov_margins(
    state: npt.ArrayLike,
    target_point_L: npt.ArrayLike,
    task_params: TaskParams,
) -> tuple[float, float, float]:
    """Return camera depth lower/upper and cone margins in meters.

    The camera is fixed by ``camera_axis_B`` at the service-body origin.  The
    line of sight is from service position to the moving pre-docking point,
    both expressed in LVLH.  The margins are only active when
    ``TaskParams.enable_fov`` is true.
    """

    x = np.asarray(state, dtype=np.float64)
    target = np.asarray(target_point_L, dtype=np.float64)
    if x.shape != (13,) or target.shape != (3,):
        raise ValueError("state must have shape (13,) and target_point_L shape (3,)")
    line_of_sight = target - x[:3]
    axis_L = quaternion_to_rotation(x[6:10]) @ task_params.camera_axis_B
    depth = float(np.dot(line_of_sight, axis_L))
    lateral = line_of_sight - depth * axis_L
    cone = float(np.tan(task_params.fov_half_angle_rad) * depth - np.linalg.norm(lateral))
    return (
        depth - task_params.min_camera_depth_m,
        task_params.max_camera_depth_m - depth,
        cone,
    )


def constraint_margins(
    predicted_state: npt.ArrayLike,
    predicted_duty: npt.ArrayLike,
    task_params: TaskParams,
    previous_duty: npt.ArrayLike | None = None,
    reference_states: npt.ArrayLike | None = None,
    *,
    dynamics_rollout: npt.ArrayLike | None = None,
) -> dict[str, float]:
    """Recompute every hard margin from a candidate trajectory.

    This routine is intentionally independent of IPOPT/acados status.  It
    returns minimum margins over all relevant stages and a maximum RK4
    dynamics residual.  Negative hard margins, non-finite values, or a
    residual above tolerance must cause the caller to reject the expert label.
    """

    states, duties = _as_trajectory(predicted_state, predicted_duty)
    if previous_duty is None:
        previous = np.zeros(16, dtype=np.float64)
    else:
        previous = np.asarray(previous_duty, dtype=np.float64)
        if previous.shape != (16,) or not np.all(np.isfinite(previous)):
            raise ValueError("previous_duty must be a finite shape-(16,) array")

    if reference_states is None:
        refs = reference_trajectory(
            task_params,
            np.arange(states.shape[0], dtype=np.float64) * task_params.control_period_s,
        )
    else:
        refs = np.asarray(reference_states, dtype=np.float64)
        if refs.shape != states.shape or not np.all(np.isfinite(refs)):
            raise ValueError("reference_states must be finite and match predicted_state shape")
    h_values: list[float] = []
    ecbf_values: list[float] = []
    velocity_values: list[float] = []
    omega_values: list[float] = []
    attitude_values: list[float] = []
    fov_depth_values: list[float] = []
    fov_upper_values: list[float] = []
    fov_cone_values: list[float] = []
    dynamics_residuals: list[float] = []
    if dynamics_rollout is not None:
        expected_rollout = np.asarray(dynamics_rollout, dtype=np.float64)
        if expected_rollout.shape != states.shape or not np.all(np.isfinite(expected_rollout)):
            raise ValueError("dynamics_rollout must be finite and match predicted_state shape")
    else:
        expected_rollout = None
    matrix = rcs_matrix_from_task(task_params)
    for index, state in enumerate(states):
        if not np.isclose(np.linalg.norm(state[6:10]), 1.0, atol=1.0e-8):
            attitude_values.append(-np.inf)
        if index < duties.shape[0]:
            h, _, _, ecbf = koz_terms(state, duties[index], task_params, rcs_matrix=matrix)
            ecbf_values.append(ecbf)
            expected_next = (
                rk4_step(state, duties[index], task_params, rcs_matrix=matrix)
                if expected_rollout is None
                else expected_rollout[index + 1]
            )
            dynamics_residuals.append(float(np.linalg.norm(expected_next - states[index + 1])))
            if task_params.enable_fov:
                depth, upper, cone = fov_margins(
                    state, camera_target_point_L(task_params, refs[index]), task_params
                )
                fov_depth_values.append(depth)
                fov_upper_values.append(upper)
                fov_cone_values.append(cone)
        else:
            h = float(
                np.dot(state[:3] - task_params.koz_center_L_m, state[:3] - task_params.koz_center_L_m)
                - task_params.koz_radius_m**2
            )
            if task_params.enable_fov:
                depth, upper, cone = fov_margins(
                    state, camera_target_point_L(task_params, refs[index]), task_params
                )
                fov_depth_values.append(depth)
                fov_upper_values.append(upper)
                fov_cone_values.append(cone)
        h_values.append(h)
        velocity_values.append(float(task_params.velocity_limit_m_s - np.max(np.abs(state[3:6]))))
        omega_values.append(
            float(task_params.angular_velocity_limit_rad_s - np.max(np.abs(state[10:13])))
        )
        attitude_error = np.linalg.norm(shortest_quaternion_error(refs[index, 6:10], state[6:10]))
        attitude_values.append(float(task_params.attitude_limit_rad - attitude_error))

    rate_margins: list[float] = []
    block_residuals: list[float] = []
    block_starts = set(task_params.action_block_starts)
    for index, duty in enumerate(duties):
        prior = previous if index == 0 else duties[index - 1]
        rate_margins.append(float(task_params.max_duty_delta - np.max(np.abs(duty - prior))))
        if index not in block_starts:
            block_residuals.append(float(np.max(np.abs(duty - prior))))
    pair_margins = [
        float(task_params.reverse_pair_limit - duty[2 * i] - duty[2 * i + 1])
        for duty in duties
        for i in range(8)
    ]

    margins = {
        "koz_h": float(min(h_values)),
        "koz_ecbf": float(min(ecbf_values)) if ecbf_values else np.inf,
        "duty_lower": float(np.min(duties)),
        "duty_upper": float(np.min(1.0 - duties)),
        "duty_rate": float(min(rate_margins)) if rate_margins else np.inf,
        "action_block_residual": float(max(block_residuals)) if block_residuals else 0.0,
        "reverse_pair": float(min(pair_margins)) if pair_margins else np.inf,
        "velocity": float(min(velocity_values)),
        "angular_velocity": float(min(omega_values)),
        "attitude": float(min(attitude_values)),
        "quaternion_norm": float(
            np.max(np.abs(np.linalg.norm(states[:, 6:10], axis=1) - 1.0))
        ),
        "dynamics_residual": float(max(dynamics_residuals)) if dynamics_residuals else 0.0,
    }
    if task_params.enable_fov:
        margins.update(
            {
                "fov_depth": float(min(fov_depth_values)),
                "fov_depth_upper": float(min(fov_upper_values)),
                "fov_cone": float(min(fov_cone_values)),
            }
        )
    return margins


def validate_solution(
    predicted_state: npt.ArrayLike,
    predicted_duty: npt.ArrayLike,
    task_params: TaskParams,
    previous_duty: npt.ArrayLike | None = None,
    *,
    tolerance: float = 1.0e-6,
    reference_states: npt.ArrayLike | None = None,
    dynamics_rollout: npt.ArrayLike | None = None,
) -> tuple[bool, dict[str, float], str | None]:
    """Return ``(accepted, margins, reason)`` after independent validation."""

    try:
        margins = constraint_margins(
            predicted_state,
            predicted_duty,
            task_params,
            previous_duty,
            reference_states,
            dynamics_rollout=dynamics_rollout,
        )
    except (ValueError, FloatingPointError) as exc:
        return False, {"validation_error": np.nan}, str(exc)
    hard_margin_keys = [
        "koz_h",
        "koz_ecbf",
        "duty_lower",
        "duty_upper",
        "duty_rate",
        "reverse_pair",
        "velocity",
        "angular_velocity",
        "attitude",
    ]
    if task_params.enable_fov:
        hard_margin_keys.extend(("fov_depth", "fov_depth_upper", "fov_cone"))
    if not all(np.isfinite(margins[key]) for key in hard_margin_keys):
        return False, margins, "non-finite hard-constraint margin"
    if margins["quaternion_norm"] > 1.0e-8:
        return False, margins, "quaternion norm residual exceeded"
    if margins["action_block_residual"] > tolerance:
        return False, margins, "action-block hold residual exceeded"
    if margins["dynamics_residual"] > tolerance:
        return False, margins, "RK4 dynamics residual exceeded"
    violated = [key for key in hard_margin_keys if margins[key] < -tolerance]
    if violated:
        return False, margins, "hard constraints violated: " + ", ".join(violated)
    return True, margins, None


def casadi_koz_terms(state: Any, duty16: Any, task_params: TaskParams, *, physical_parameters: Any = None) -> tuple[Any, Any]:
    """Return symbolic KOZ ``h`` and ECBF margin for acados/IPOPT models."""

    import casadi as ca

    offset = state[0:3] - ca.DM(task_params.koz_center_L_m)
    velocity = state[3:6]
    acceleration = casadi_continuous_dynamics(state, duty16, task_params, physical_parameters=physical_parameters)[3:6]
    h = ca.dot(offset, offset) - task_params.koz_radius_m**2
    h_dot = 2.0 * ca.dot(offset, velocity)
    h_ddot = 2.0 * ca.dot(velocity, velocity) + 2.0 * ca.dot(offset, acceleration)
    ecbf = (
        h_ddot
        + task_params.koz_k1_s * h_dot
        + task_params.koz_k0_s2 * h
        - task_params.koz_safety_margin_m2_s2
    )
    return h, ecbf


def casadi_fov_terms(state: Any, target_point_L: npt.ArrayLike, task_params: TaskParams) -> tuple[Any, Any, Any]:
    """Return symbolic FOV depth lower/upper and a squared cone margin.

    The positive depth lower bound is imposed alongside the cone.  Therefore
    ``tan(angle)^2*depth^2-||lateral||^2 >= 0`` is equivalent to the numeric
    ``tan(angle)*depth-||lateral|| >= 0`` condition, while avoiding the
    square-root kink at the optical axis during acados QP linearization.
    """

    import casadi as ca

    if isinstance(target_point_L, (ca.SX, ca.MX, ca.DM)):
        target = target_point_L
    else:
        target = ca.DM(np.asarray(target_point_L, dtype=np.float64))
    line_of_sight = target - state[0:3]
    from .dynamics import casadi_quaternion_to_rotation

    axis_L = casadi_quaternion_to_rotation(state[6:10]) @ ca.DM(task_params.camera_axis_B)
    depth = ca.dot(line_of_sight, axis_L)
    lateral = line_of_sight - depth * axis_L
    cone = (
        ca.tan(task_params.fov_half_angle_rad) ** 2 * depth**2
        - ca.dot(lateral, lateral)
    )
    return (
        depth - task_params.min_camera_depth_m,
        task_params.max_camera_depth_m - depth,
        cone,
    )
