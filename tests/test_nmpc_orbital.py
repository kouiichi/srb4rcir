"""Independent frame and dynamics checks; no Kit required."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from srb.nmpc import TaskParams
from srb.nmpc.dynamics import cw_acceleration, casadi_rk4_map, rk4_step
from srb.nmpc.orbital import lvlh_to_world_state, world_to_lvlh_state, tidal_acceleration_world
from srb.nmpc.physical import pack_physical_parameters


def test_zero_mean_motion_is_free_floating():
    assert np.array_equal(cw_acceleration([25, 1, 0], [0.1, 0, 0], 0), np.zeros(3))
    with pytest.raises(ValueError):
        cw_acceleration([1, 0, 0], [0, 0, 0], -0.1)


def test_frame_roundtrip_nonzero_time_attitude_and_rates():
    x = np.array([-25, .2, .15, .03, -.02, .01, .8, .2, -.4, .4, .01, -.02, .03])
    x[6:10] /= np.linalg.norm(x[6:10])
    w = lvlh_to_world_state(x, 83., .001027)
    np.testing.assert_allclose(world_to_lvlh_state(w, 83., .001027), x, atol=1e-12)
    # omega_BI_B must not change with the frame used to represent position.
    np.testing.assert_array_equal(w[10:13], x[10:13])


def test_tidal_world_dynamics_equals_independent_hcw_matrix_exponential():
    n = .001027
    x = np.array([-25, .2, .15, .03, -.02, .01, 1., 0, 0, 0, 0, 0, 0])
    w = lvlh_to_world_state(x, 0., n)
    def rhs(t, y):
        return np.r_[y[3:6], tidal_acceleration_world(y[:3], t, n)]
    y = solve_ivp(rhs, (0, 120), w[:6], rtol=1e-11, atol=1e-12).y[:, -1]
    actual = world_to_lvlh_state(np.r_[y, 1., 0, 0, 0, 0, 0, 0], 120, n)[:6]
    A = np.zeros((6, 6)); A[:3, 3:] = np.eye(3)
    A[3, 0] = 3*n*n; A[3, 4] = 2*n; A[4, 3] = -2*n; A[5, 2] = -n*n
    np.testing.assert_allclose(actual, expm(A*120) @ x[:6], atol=2e-9)


def test_symbolic_runtime_parameters_match_numeric_dynamics():
    import casadi as ca
    x = ca.SX.sym('x', 13); u = ca.SX.sym('u', 16); p = ca.SX.sym('p', 115)
    nominal = TaskParams()
    f = ca.Function('runtime_map', [x, u, p], [casadi_rk4_map(x, u, nominal, physical_parameters=p)])
    initial = np.array([-25, .2, .15, .01, -.02, .01, 1., 0, 0, 0, .01, .02, -.01])
    duty = np.linspace(.01, .2, 16)
    from srb.nmpc.rcs import rcs_matrix_from_task
    params = replace(nominal, mass_kg=8.7, inertia_B_kg_m2=np.diag([.31, .18, .26]),
                     rcs_matrix_B=rcs_matrix_from_task(nominal)*.95)
    np.testing.assert_allclose(np.array(f(initial, duty, pack_physical_parameters(params))).ravel(),
                               rk4_step(initial, duty, params), atol=1e-11)
