"""HCW/CW plus rigid-body quaternion dynamics for the privileged expert."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .quaternion import (
    normalize_quaternion,
    omega_body_matrix,
    quaternion_to_rotation,
)
from .rcs import rcs_matrix_from_task
from .types import FloatArray, RcsDuty16, State13, TaskParams


def _state(value: npt.ArrayLike, *, normalize_quat: bool = True) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (13,) or not np.all(np.isfinite(result)):
        raise ValueError("state must be a finite shape-(13,) float64 array")
    result = np.ascontiguousarray(result.copy())
    if normalize_quat:
        result[6:10] = normalize_quaternion(result[6:10])
    return result


def _duty(value: npt.ArrayLike) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (16,) or not np.all(np.isfinite(result)):
        raise ValueError("duty must be a finite shape-(16,) float64 array")
    return np.ascontiguousarray(result)


def cw_acceleration(
    r_L_m: npt.ArrayLike,
    v_L_m_s: npt.ArrayLike,
    mean_motion_rad_s: float,
) -> FloatArray:
    """Return unforced HCW/CW relative acceleration in LVLH coordinates.

    The components are exactly ``[3*n^2*x + 2*n*vy, -2*n*vx,
    -n^2*z]``.  Inputs are target-relative LVLH position/velocity in SI
    units, and the result is m/s^2.
    """

    r = np.asarray(r_L_m, dtype=np.float64)
    v = np.asarray(v_L_m_s, dtype=np.float64)
    if r.shape != (3,) or v.shape != (3,) or not np.all(np.isfinite(r)) or not np.all(
        np.isfinite(v)
    ):
        raise ValueError("r_L_m and v_L_m_s must be finite shape-(3,) arrays")
    if not np.isfinite(mean_motion_rad_s) or mean_motion_rad_s < 0.0:
        raise ValueError("mean_motion_rad_s must be finite and >= 0")
    n = float(mean_motion_rad_s)
    return np.array(
        [
            3.0 * n * n * r[0] + 2.0 * n * v[1],
            -2.0 * n * v[0],
            -n * n * r[2],
        ],
        dtype=np.float64,
    )


def continuous_dynamics(
    state: State13,
    duty16: RcsDuty16,
    task_params: TaskParams,
    *,
    rcs_matrix: FloatArray | None = None,
) -> FloatArray:
    """Evaluate the 13-state continuous plant model.

    Args:
        state: Shape ``(13,)`` ``[r_L,v_L,q_LB,omega_BI_B]`` at timestamp t.
        duty16: Shape ``(16,)`` one-sided body RCS duties.  The function does
            not clip them; optimization and allocation code enforce bounds.
        task_params: SI physical parameters and LVLH mean motion.

    Returns:
        Shape ``(13,)`` derivative ``[m/s,m/s^2,1/s,rad/s^2]``.  The
        quaternion derivative uses the *relative* ``omega_B/L_B`` computed
        from inertial body rate and LVLH rotation; it never substitutes
        ``omega_BI_B`` directly.
    """

    # Do not renormalize RK4 intermediate states here.  The discrete map
    # normalizes once after all five substeps, as required by the contract;
    # rotation conversion remains well-defined for the near-unit stages.
    x = _state(state, normalize_quat=False)
    duty = _duty(duty16)
    B = rcs_matrix_from_task(task_params) if rcs_matrix is None else np.asarray(rcs_matrix, dtype=np.float64)
    if B.shape != (6, 16) or not np.all(np.isfinite(B)):
        raise ValueError("rcs_matrix must be a finite shape-(6,16) array")
    force_B = B[:3] @ duty
    torque_B = B[3:] @ duty

    r = x[:3]
    v = x[3:6]
    q = x[6:10]
    omega_BI_B = x[10:13]
    R_LB = quaternion_to_rotation(q)
    acceleration = cw_acceleration(r, v, task_params.mean_motion_rad_s)
    acceleration += R_LB @ force_B / task_params.mass_kg

    # LVLH rotates with the circular reference orbit at [0, 0, n] in L.
    # Expressing that rate in B is required before subtracting it from the
    # inertial body rate.
    omega_LI_L = np.array([0.0, 0.0, task_params.mean_motion_rad_s])
    omega_B_L_B = omega_BI_B - R_LB.T @ omega_LI_L
    q_dot = 0.5 * omega_body_matrix(omega_B_L_B) @ q

    inertia = task_params.inertia_B_kg_m2
    omega_dot = np.linalg.solve(
        inertia, torque_B - np.cross(omega_BI_B, inertia @ omega_BI_B)
    )
    return np.concatenate((v, acceleration, q_dot, omega_dot)).astype(np.float64)


def _rotation_matrix_fast(quaternion: FloatArray) -> FloatArray:
    """Low-allocation normalized quaternion rotation for repeated RK4 calls."""

    norm = float(np.sqrt(np.dot(quaternion, quaternion)))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion norm is too small to normalize")
    w, qx, qy, qz = quaternion / norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - w * qz), 2.0 * (qx * qz + w * qy)],
            [2.0 * (qx * qy + w * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - w * qx)],
            [2.0 * (qx * qz - w * qy), 2.0 * (qy * qz + w * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _continuous_dynamics_fast(
    state: FloatArray,
    duty16: FloatArray,
    task_params: TaskParams,
    matrix: FloatArray,
    inertia_inverse: FloatArray,
) -> FloatArray:
    """Validated-input fast path used by repeated fixed-parameter RK4 steps."""

    r = state[:3]
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    rotation_LB = _rotation_matrix_fast(q)
    n = task_params.mean_motion_rad_s
    acceleration = np.array(
        [
            3.0 * n * n * r[0] + 2.0 * n * v[1],
            -2.0 * n * v[0],
            -n * n * r[2],
        ],
        dtype=np.float64,
    )
    acceleration += rotation_LB @ (matrix[:3] @ duty16) / task_params.mass_kg
    relative = omega - rotation_LB.T @ np.array([0.0, 0.0, n], dtype=np.float64)
    relative_cross_q = np.cross(relative, q[1:])
    q_dot = 0.5 * np.concatenate(
        (
            np.array([-np.dot(relative, q[1:])], dtype=np.float64),
            q[0] * relative - relative_cross_q,
        )
    )
    angular_momentum = task_params.inertia_B_kg_m2 @ omega
    omega_dot = inertia_inverse @ (matrix[3:] @ duty16 - np.cross(omega, angular_momentum))
    return np.concatenate((v, acceleration, q_dot, omega_dot)).astype(np.float64)


def rk4_step(
    state: State13,
    duty16: RcsDuty16,
    task_params: TaskParams,
    *,
    dt_s: float | None = None,
    rcs_matrix: FloatArray | None = None,
) -> FloatArray:
    """Integrate one control interval with fixed-substep RK4.

    The default uses the global SRB interval ``Ts=0.05 s`` and exactly five
    ``0.01 s`` substeps.  The quaternion is normalized once after the complete
    discrete step, matching the contract.  ``dt_s`` is intended for short
    reference integration and must remain a positive multiple-compatible
    interval.
    """

    x = _state(state)
    u = _duty(duty16)
    total_dt = task_params.control_period_s if dt_s is None else float(dt_s)
    if not np.isfinite(total_dt) or total_dt <= 0.0:
        raise ValueError("dt_s must be finite and > 0")
    if dt_s is None:
        substeps = task_params.rk4_substeps
        sub_dt = task_params.rk4_substep_dt_s
    else:
        substeps = max(1, int(np.ceil(total_dt / task_params.rk4_substep_dt_s)))
        sub_dt = total_dt / substeps

    matrix = rcs_matrix_from_task(task_params) if rcs_matrix is None else np.asarray(rcs_matrix, dtype=np.float64)
    if matrix.shape != (6, 16) or not np.all(np.isfinite(matrix)):
        raise ValueError("rcs_matrix must be a finite shape-(6,16) array")
    inertia_inverse = np.linalg.inv(task_params.inertia_B_kg_m2)
    result = x.copy()
    for _ in range(substeps):
        k1 = _continuous_dynamics_fast(result, u, task_params, matrix, inertia_inverse)
        k2 = _continuous_dynamics_fast(
            result + 0.5 * sub_dt * k1, u, task_params, matrix, inertia_inverse
        )
        k3 = _continuous_dynamics_fast(
            result + 0.5 * sub_dt * k2, u, task_params, matrix, inertia_inverse
        )
        k4 = _continuous_dynamics_fast(
            result + sub_dt * k3, u, task_params, matrix, inertia_inverse
        )
        result = result + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[6:10] = normalize_quaternion(result[6:10])
    return np.ascontiguousarray(result)


def _reference_quaternion_step(
    quaternion: FloatArray,
    target_omega_BI_B: FloatArray,
    mean_motion_rad_s: float,
    dt_s: float,
) -> FloatArray:
    """RK4 update for the desired target attitude with constant inertial rate."""

    omega_LI_L = np.array([0.0, 0.0, mean_motion_rad_s], dtype=np.float64)

    def derivative(q_value: FloatArray) -> FloatArray:
        R = quaternion_to_rotation(q_value)
        relative = target_omega_BI_B - R.T @ omega_LI_L
        return 0.5 * omega_body_matrix(relative) @ q_value

    k1 = derivative(quaternion)
    k2 = derivative(normalize_quaternion(quaternion + 0.5 * dt_s * k1))
    k3 = derivative(normalize_quaternion(quaternion + 0.5 * dt_s * k2))
    k4 = derivative(normalize_quaternion(quaternion + dt_s * k3))
    return normalize_quaternion(quaternion + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def reference_trajectory(
    task_params: TaskParams,
    times_s: npt.ArrayLike,
) -> FloatArray:
    """Generate desired pre-docking states for monotonically increasing times.

    The target point is ``target_point_T_m`` rotated by the target attitude.
    Its velocity is the relative rotation velocity in LVLH, so a moving target
    produces a nonzero terminal velocity when appropriate.  Returned shape is
    ``(len(times_s),13)`` with the same frame/unit order as :class:`State13`.
    """

    times = np.asarray(times_s, dtype=np.float64)
    if times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times_s must be a non-empty finite shape-(T,) array")
    if np.any(np.diff(times) < -1.0e-12) or times[0] < -1.0e-12:
        raise ValueError("times_s must be non-negative and non-decreasing")

    q = normalize_quaternion(task_params.target_q_LT0)
    target_omega = task_params.target_omega_BI_B_rad_s
    current_time = 0.0
    trajectory: list[FloatArray] = []
    for timestamp in times:
        delta_t = float(timestamp - current_time)
        if delta_t > 0.0:
            substeps = max(1, int(np.ceil(delta_t / task_params.rk4_substep_dt_s)))
            for _ in range(substeps):
                q = _reference_quaternion_step(
                    q,
                    target_omega,
                    task_params.mean_motion_rad_s,
                    delta_t / substeps,
                )
        current_time = float(timestamp)
        R_LT = quaternion_to_rotation(q)
        omega_LI_L = np.array([0.0, 0.0, task_params.mean_motion_rad_s])
        omega_T_L_T = target_omega - R_LT.T @ omega_LI_L
        position = R_LT @ task_params.target_point_T_m
        velocity = R_LT @ np.cross(omega_T_L_T, task_params.target_point_T_m)
        trajectory.append(
            np.concatenate((position, velocity, q, target_omega)).astype(np.float64)
        )
    return np.ascontiguousarray(np.vstack(trajectory))


def reference_at(task_params: TaskParams, time_s: float) -> FloatArray:
    """Return one desired pre-docking state at a non-negative timestamp."""

    return reference_trajectory(task_params, np.array([time_s], dtype=np.float64))[0]


def camera_target_point_L(
    task_params: TaskParams, reference_state: npt.ArrayLike
) -> FloatArray:
    """Convert the target feature point from ``T`` to the current ``L`` frame.

    ``target_point_T_m`` is the service hover point and is intentionally not
    reused here: the camera observes ``camera_target_point_T_m`` (the target
    origin by default), so the ideal hover state has a meaningful positive
    camera depth.  ``reference_state`` must be the target-attitude reference
    at the same timestamp and have shape ``(13,)``.
    """

    reference = np.asarray(reference_state, dtype=np.float64)
    if reference.shape != (13,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_state must be a finite shape-(13,) array")
    return np.ascontiguousarray(
        quaternion_to_rotation(reference[6:10]) @ task_params.camera_target_point_T_m
    )


def _ca_skew(vector: Any) -> Any:
    import casadi as ca

    return ca.vertcat(
        ca.horzcat(0.0, -vector[2], vector[1]),
        ca.horzcat(vector[2], 0.0, -vector[0]),
        ca.horzcat(-vector[1], vector[0], 0.0),
    )


def casadi_quaternion_to_rotation(quaternion: Any) -> Any:
    """CasADi equivalent of :func:`quaternion_to_rotation` for code generation."""

    import casadi as ca

    w = quaternion[0]
    v = quaternion[1:4]
    return (w * w - ca.dot(v, v)) * ca.DM.eye(3) + 2.0 * (v @ v.T) + 2.0 * w * _ca_skew(v)


def casadi_omega_body_matrix(omega_body: Any) -> Any:
    """CasADi matrix for the scalar-first body-rate quaternion convention."""

    import casadi as ca

    return ca.vertcat(
        ca.horzcat(0.0, -omega_body.T),
        ca.horzcat(omega_body, -_ca_skew(omega_body)),
    )


def casadi_continuous_dynamics(
    state: Any,
    duty16: Any,
    task_params: TaskParams,
    *, physical_parameters: Any = None,
) -> Any:
    """Build the symbolic continuous HCW/rigid-body dynamics expression."""

    import casadi as ca

    if physical_parameters is None:
        B = ca.DM(rcs_matrix_from_task(task_params))
        mass = task_params.mass_kg
        inertia = ca.DM(task_params.inertia_B_kg_m2)
        inertia_inverse = ca.DM(np.linalg.inv(task_params.inertia_B_kg_m2))
    else:
        mass = physical_parameters[0]
        inertia = ca.reshape(physical_parameters[1:10], 3, 3).T
        inertia_inverse = ca.reshape(physical_parameters[10:19], 3, 3).T
        B = ca.reshape(physical_parameters[19:115], 16, 6).T
    r = state[0:3]
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]
    n = task_params.mean_motion_rad_s
    force_B = B[0:3, :] @ duty16
    torque_B = B[3:6, :] @ duty16
    R_LB = casadi_quaternion_to_rotation(q)
    acceleration = ca.vertcat(
        3.0 * n * n * r[0] + 2.0 * n * v[1],
        -2.0 * n * v[0],
        -n * n * r[2],
    ) + R_LB @ force_B / mass
    omega_LI_L = ca.DM([0.0, 0.0, n])
    relative_omega = omega - R_LB.T @ omega_LI_L
    q_dot = 0.5 * casadi_omega_body_matrix(relative_omega) @ q
    # Use a constant inverse rather than symbolic ``ca.solve``.  CasADi 3.8
    # may lower the latter to an SX QR linsol during acados/IPOPT expansion,
    # while the rigid-body inertia is fixed for one episode and can be
    # inverted exactly once here.
    omega_dot = inertia_inverse @ (torque_B - ca.cross(omega, inertia @ omega))
    return ca.vertcat(v, acceleration, q_dot, omega_dot)


def casadi_rk4_map(state: Any, duty16: Any, task_params: TaskParams, *, physical_parameters: Any = None) -> Any:
    """Build the normalized five-substep RK4 map used by both solvers."""

    import casadi as ca

    dt = task_params.rk4_substep_dt_s
    result = state
    for _ in range(task_params.rk4_substeps):
        k1 = casadi_continuous_dynamics(result, duty16, task_params, physical_parameters=physical_parameters)
        k2 = casadi_continuous_dynamics(result + 0.5 * dt * k1, duty16, task_params, physical_parameters=physical_parameters)
        k3 = casadi_continuous_dynamics(result + 0.5 * dt * k2, duty16, task_params, physical_parameters=physical_parameters)
        k4 = casadi_continuous_dynamics(result + dt * k3, duty16, task_params, physical_parameters=physical_parameters)
        result = result + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    q = result[6:10]
    q = q / ca.sqrt(ca.sumsqr(q) + 1.0e-16)
    return ca.vertcat(result[0:6], q, result[10:13])


def casadi_quaternion_error_vector(reference: Any, actual: Any) -> Any:
    """Build a shortest-sign quaternion error vector for CasADi costs."""

    import casadi as ca

    ref_conjugate = ca.vertcat(reference[0], -reference[1:4])
    p = ref_conjugate
    q = actual
    error = ca.vertcat(
        p[0] * q[0] - ca.dot(p[1:4], q[1:4]),
        p[0] * q[1:4] + q[0] * p[1:4] + ca.cross(p[1:4], q[1:4]),
    )
    vector = 2.0 * error[1:4]
    return ca.if_else(error[0] >= 0.0, vector, -vector)
