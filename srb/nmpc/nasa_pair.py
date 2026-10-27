"""Plane geometry and actuator diagnostics for original DESDynI / NEAR models."""
from __future__ import annotations

import numpy as np

from .quaternion import quaternion_to_rotation
from .surface_approach import synchronous_reference


def bottom_to_top_goal(target_position, target_quaternion, actor_bottom, target_top, gap_m=.5):
    """Align source -Z actor bottom opposite source +Z target top.

    Both bodies share the final rotation. A target -90 degree Y rotation
    makes its top face world -X and the actor bottom face world +X.
    """
    if not np.isfinite(gap_m) or gap_m <= 0:
        raise ValueError('surface gap must be positive and finite')
    q = np.asarray(target_quaternion, dtype=float)
    q = q / np.linalg.norm(q)
    rotation = quaternion_to_rotation(q)
    p = (np.asarray(target_position) + rotation @ np.asarray(target_top)
         + gap_m * (rotation @ [0., 0., 1.]) - rotation @ np.asarray(actor_bottom))
    return p, q


def plane_sample(state, com, actor_bottom, target_top, target_quaternion):
    """Measure signed plane separation using a target-relative COM state."""
    x = np.asarray(state)
    actor_r = quaternion_to_rotation(x[6:10])
    target_r = quaternion_to_rotation(target_quaternion)
    normal = target_r @ [0., 0., 1.]
    root = x[:3] - actor_r @ np.asarray(com)
    delta = root + actor_r @ np.asarray(actor_bottom) - target_r @ np.asarray(target_top)
    gap = float(delta @ normal)
    return {
        'face_gap_m': gap,
        'lateral_error_m': float(np.linalg.norm(delta - gap * normal)),
        'facing_error_deg': float(np.rad2deg(np.arccos(np.clip(
            (actor_r @ [0., 0., -1.]) @ -normal, -1., 1.)))),
    }


def full_mesh_plane_gaps(states, com, actor_points, target_points, target_quaternion):
    """Signed support-plane separation including all appendages.

    Convex-hull vertices preserve the exact support values of the original
    mesh; they are used for this scalar check, never as collision geometry.
    """
    xs = np.asarray(states)
    rotations = np.asarray([quaternion_to_rotation(x[6:10]) for x in xs])
    target_r = quaternion_to_rotation(target_quaternion)
    normal = target_r @ [0.,0.,1.]
    roots = xs[:,:3] - np.einsum('nij,j->ni',rotations,np.asarray(com))
    local_normals = np.einsum('nji,j->ni',rotations,normal)
    actor_support = np.min(np.asarray(actor_points) @ local_normals.T,axis=0) + roots@normal
    target_support = np.max((np.asarray(target_points)@target_r.T)@normal)
    return actor_support-target_support


def turn_then_approach_reference(times, actor_start, quaternion_start, actor_goal,
                                 quaternion_goal, com, turn_s, approach_s):
    """Turn at the initial root position, then translate at the final attitude."""
    if not np.isfinite(approach_s) or approach_s <= 0:
        raise ValueError('approach duration must be positive and finite')
    times = np.asarray(times, dtype=float)
    result = synchronous_reference(times, actor_start, quaternion_start, actor_start,
                                   quaternion_goal, com, turn_s)
    s = np.clip((times - turn_s) / approach_s, 0., 1.)
    rate = 30 * s**2 * (1-s)**2 / approach_s
    progress = 10*s**3 - 15*s**4 + 6*s**5
    delta = np.asarray(actor_goal) - actor_start
    result[:, :3] += progress[:, None] * delta
    result[:, 3:6] += rate[:, None] * delta
    return result


def thruster_wrench_matrix(offsets, exhaust_directions, powers, com):
    """Use ThrustAction's exhaust sign and actual actuator count, without padding."""
    positions = np.asarray(offsets, dtype=float)
    directions = np.asarray(exhaust_directions, dtype=float)
    powers = np.asarray(powers, dtype=float)
    if (positions.ndim != 2 or positions.shape[1] != 3 or directions.shape != positions.shape
            or powers.shape != (len(positions),) or len(positions) == 0
            or not all(np.all(np.isfinite(a)) for a in (positions, directions, powers))):
        raise ValueError('thruster positions/directions must be finite (n,3), powers finite (n,)')
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0) or np.any(powers < 0):
        raise ValueError('exhaust directions must be nonzero and powers nonnegative')
    forces = -powers[:, None] * directions / norms[:, None]
    return np.vstack((forces.T, np.cross(positions - np.asarray(com), forces).T))


def actuation_diagnostics(matrix, approach_axis_body):
    """Report instantaneous rank and one-sided force authority at fixed attitude."""
    b = np.asarray(matrix, dtype=float)
    axis = np.asarray(approach_axis_body, dtype=float)
    axis = axis / np.linalg.norm(axis)
    forward = axis @ b[:3]
    minimum = float(np.minimum(forward, 0.).sum())
    return {
        'actuator_count': b.shape[1],
        'wrench_rank': int(np.linalg.matrix_rank(b)),
        'translation_rank': int(np.linalg.matrix_rank(b[:3])),
        'minimum_forward_force_N': minimum,
        'maximum_forward_force_N': float(np.maximum(forward, 0.).sum()),
        'can_brake_at_fixed_attitude': bool(minimum < -1e-10),
        'scope': 'fixed-attitude one-sided force authority; not nonlinear global controllability',
    }
