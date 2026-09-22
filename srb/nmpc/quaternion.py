"""Scalar-first quaternion helpers used by the NMPC model.

The convention is ``q_LB=[qw,qx,qy,qz]`` mapping a vector from body frame B
to LVLH frame L.  Angular velocity arguments are body-frame vectors.  The
helpers intentionally distinguish shortest-error computation from state
normalization so that the equivalent ``q`` and ``-q`` representations never
create an artificial attitude error.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .types import FloatArray


def skew(vector: npt.ArrayLike) -> FloatArray:
    """Return the 3x3 cross-product matrix for a finite 3-vector."""

    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("vector must be a finite shape-(3,) array")
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )


def normalize_quaternion(quaternion: npt.ArrayLike, eps: float = 1.0e-12) -> FloatArray:
    """Normalize a scalar-first quaternion, rejecting a near-zero norm."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion must be a finite shape-(4,) array")
    norm = float(np.linalg.norm(q))
    if norm <= eps:
        raise ValueError("quaternion norm is too small to normalize")
    return np.ascontiguousarray(q / norm)


def quaternion_conjugate(quaternion: npt.ArrayLike) -> FloatArray:
    """Return the conjugate of a scalar-first quaternion."""

    q = normalize_quaternion(quaternion)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quaternion_multiply(first: npt.ArrayLike, second: npt.ArrayLike) -> FloatArray:
    """Multiply scalar-first quaternions using the Hamilton convention."""

    p = np.asarray(first, dtype=np.float64)
    q = np.asarray(second, dtype=np.float64)
    if p.shape != (4,) or q.shape != (4,):
        raise ValueError("quaternion operands must both have shape (4,)")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        raise ValueError("quaternion operands must be finite")
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    return np.array(
        [
            pw * qw - float(np.dot(pv, qv)),
            *(pw * qv + qw * pv + np.cross(pv, qv)),
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation(quaternion: npt.ArrayLike) -> FloatArray:
    """Return ``R_LB`` mapping body-frame vectors into LVLH."""

    q = normalize_quaternion(quaternion)
    w = q[0]
    v = q[1:]
    return np.ascontiguousarray(
        (w * w - float(np.dot(v, v))) * np.eye(3)
        + 2.0 * np.outer(v, v)
        + 2.0 * w * skew(v)
    )


def omega_body_matrix(omega_body: npt.ArrayLike) -> FloatArray:
    """Return ``Omega(omega)`` for ``q_dot=0.5*Omega*q``.

    This is the right-Hamilton matrix for angular velocity expressed in body
    axes: ``q_dot = 0.5 * q ⊗ [0, omega_body]``.  It is used for the relative
    ``omega_B/L_B`` state derivative, not for the inertial angular-velocity
    state itself.
    """

    omega = np.asarray(omega_body, dtype=np.float64)
    if omega.shape != (3,) or not np.all(np.isfinite(omega)):
        raise ValueError("omega_body must be a finite shape-(3,) array")
    return np.block(
        [
            [np.zeros((1, 1)), -omega.reshape(1, 3)],
            [omega.reshape(3, 1), -skew(omega)],
        ]
    )


def shortest_quaternion_error(
    reference: npt.ArrayLike, actual: npt.ArrayLike
) -> FloatArray:
    """Return the shortest local attitude error vector ``2*q_err[1:]``.

    ``q_err = conjugate(reference) * actual`` is sign-corrected to have a
    non-negative scalar component.  Consequently a state represented by
    ``-reference`` has zero error, as required by quaternion equivalence.
    """

    q_ref = normalize_quaternion(reference)
    q_actual = normalize_quaternion(actual)
    q_error = quaternion_multiply(quaternion_conjugate(q_ref), q_actual)
    if q_error[0] < 0.0:
        q_error = -q_error
    return np.ascontiguousarray(2.0 * q_error[1:])


def integrate_quaternion(
    quaternion: npt.ArrayLike,
    omega_body: npt.ArrayLike,
    dt_s: float,
    *,
    normalize: bool = True,
) -> FloatArray:
    """Integrate a constant body-rate quaternion with first-order Euler.

    This helper is only for short reference updates and tests; the plant and
    expert use the RK4 functions in :mod:`srb.nmpc.dynamics`.  ``dt_s`` is in
    seconds and the returned quaternion is normalized by default.
    """

    if not np.isfinite(dt_s):
        raise ValueError("dt_s must be finite")
    q = normalize_quaternion(quaternion)
    result = q + 0.5 * float(dt_s) * omega_body_matrix(omega_body) @ q
    return normalize_quaternion(result) if normalize else result
