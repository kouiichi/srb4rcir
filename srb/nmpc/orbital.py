"""Circular-orbit local frames, independent of Kit/PhysX.

World axes are nonrotating and translate with the reference orbit origin.
R_IL rotates LVLH (radial, along-track, orbit-normal) into world axes.
The differential gravity tensor in these axes produces HCW after the full
velocity transformation; no fictitious angular-rate dynamics are required.
"""
import numpy as np
from .quaternion import quaternion_multiply, quaternion_to_rotation


def orbital_quaternion(time_s, mean_motion):
    if not np.isfinite(time_s) or not np.isfinite(mean_motion) or mean_motion < 0:
        raise ValueError('finite time and nonnegative mean motion required')
    angle = .5 * mean_motion * time_s
    return np.array([np.cos(angle), 0., 0., np.sin(angle)])


def lvlh_to_world_state(state, time_s, mean_motion):
    x = np.asarray(state, dtype=float).copy()
    q_IL = orbital_quaternion(time_s, mean_motion)
    R = quaternion_to_rotation(q_IL)
    x[:3] = R @ state[:3]
    x[3:6] = R @ (state[3:6] + np.cross([0, 0, mean_motion], state[:3]))
    x[6:10] = quaternion_multiply(q_IL, state[6:10])
    return x


def world_to_lvlh_state(state, time_s, mean_motion):
    x = np.asarray(state, dtype=float).copy()
    q_IL = orbital_quaternion(time_s, mean_motion)
    R = quaternion_to_rotation(q_IL)
    x[:3] = R.T @ state[:3]
    x[3:6] = R.T @ state[3:6] - np.cross([0, 0, mean_motion], x[:3])
    q_IL[1:] *= -1
    x[6:10] = quaternion_multiply(q_IL, state[6:10])
    return x


def tidal_acceleration_world(position_world, time_s, mean_motion):
    R = quaternion_to_rotation(orbital_quaternion(time_s, mean_motion))
    return R @ (mean_motion**2 * np.array([2., -1., -1.]) * (R.T @ position_world))
