"""Executor timing tests independent of the optimizer and Kit."""
import numpy as np
import pytest
from srb.nmpc import TaskParams
from srb.nmpc.physical import pack_physical_parameters


def test_force_uses_pre_update_mass_and_next_state_uses_wet_mass():
    from srb.nmpc import matched_executor as module
    model = module.MatchedExecutor(6., np.ones(16)*.001, .05, .05)
    x = np.zeros(14); x[6] = 1; x[13] = 6.
    B = np.zeros((6,16)); B[0,0] = 1.
    params = TaskParams(mass_kg=6., inertia_B_kg_m2=np.eye(3), rcs_matrix_B=B)
    u = np.zeros(16); u[0] = 1
    # The initialization discontinuity is represented separately from fuel.
    y = model.step(x,u,pack_physical_parameters(params), fuel=5.)
    assert y[3] == pytest.approx(.05/6)
    assert y[0] == pytest.approx(.05*.05/6)
    assert y[13] == pytest.approx(10.999)


def test_zero_force_keeps_com_velocity_and_angular_damping():
    from srb.nmpc import matched_executor as module
    model = module.MatchedExecutor(6., np.ones(16)*.001, .05, .05)
    x = np.zeros(14); x[6] = 1; x[3] = .2; x[10] = .1; x[13] = 11.
    p=pack_physical_parameters(TaskParams(mass_kg=11., inertia_B_kg_m2=np.eye(3)))
    y=model.step(x,np.zeros(16),p)
    assert y[3] == pytest.approx(.2)
    assert y[0] == pytest.approx(.01)
    assert y[10] == pytest.approx(.1*(1-.05*.05))
    assert np.linalg.norm(y[6:10]) == pytest.approx(1.)


def test_empty_fuel_disables_force():
    from srb.nmpc import matched_executor as module
    model = module.MatchedExecutor(6., np.ones(16)*.001, .05, .05)
    x=np.zeros(14); x[6]=1; x[13]=6
    p=pack_physical_parameters(TaskParams(mass_kg=6., inertia_B_kg_m2=np.eye(3)))
    y=model.step(x,np.arange(16)/16,p)
    np.testing.assert_allclose(y,x)


def test_warm_rollout_matches_current_mass_and_block_constraints():
    from srb.nmpc.matched_executor import MatchedExecutor
    model=MatchedExecutor(6.,np.ones(16)*.001)
    state=np.zeros(15); state[6]=1.; state[13:]=[11.,5.]
    p=pack_physical_parameters(TaskParams(mass_kg=11.,inertia_B_kg_m2=np.eye(3)))
    duties=np.arange(64).reshape(4,16)/64
    states,blocked=model.warm_rollout(state,duties,p,(2,2))
    np.testing.assert_allclose(blocked[0],blocked[1])
    np.testing.assert_allclose(blocked[2],blocked[3])
    for k in range(4):
        np.testing.assert_allclose(states[k+1],model.step(states[k],blocked[k],p))


def test_empty_fuel_barrier_uses_zero_available_thrust():
    from srb.nmpc.constraints import constraint_margins, koz_terms
    B=np.zeros((6,16)); B[0,0]=1
    p=TaskParams(mean_motion_rad_s=0.,mass_kg=6.,rcs_matrix_B=B)
    x=np.zeros(13); x[0]=.6; x[3]=-.1; x[6]=1
    u=np.zeros(16); u[0]=1
    states=np.tile(x,(2,1))
    m=constraint_margins(states,u[None],p,dynamics_rollout=states,
                        stage_masses=[6,6],stage_fuel=[0,0])
    assert m['koz_ecbf']==pytest.approx(koz_terms(x,np.zeros(16),p)[3])


def test_stationary_com_goal_uses_world_position_and_goal_attitude():
    from srb.nmpc.matched_executor import stationary_com_goal
    from srb.nmpc.quaternion import quaternion_to_rotation
    q=np.array([np.sqrt(.5),0,np.sqrt(.5),0]); com=np.array([.003,.004,.045])
    goal=stationary_com_goal([0,0,.5],q,com)
    np.testing.assert_allclose(goal[:3],np.array([0,0,.5])+quaternion_to_rotation(q)@com)
    np.testing.assert_allclose(goal[6:10],q)
    np.testing.assert_allclose(goal[[3,4,5,10,11,12]],0)
