"""Guard physical jerk units and executed-history continuity in the real OCP."""
from dataclasses import replace
import numpy as np
import pytest
import casadi as ca

from srb.nmpc import TaskParams
from srb.nmpc.acados_expert import AcadosExpert
from srb.nmpc.matched_executor import MatchedExecutor
from srb.nmpc.physical import pack_physical_parameters


@pytest.fixture(scope='module')
def controller(tmp_path_factory):
    B=np.zeros((6,16)); B[0,0]=2.; B[0,1]=-2.
    params=replace(TaskParams(),mean_motion_rad_s=0.,mass_kg=10.,
                   inertia_B_kg_m2=np.eye(3),rcs_matrix_B=B)
    model=MatchedExecutor(5.,np.zeros(16)); model.remaining_fuel=5.
    return AcadosExpert(params,executor_model=model,jerk_weight=1.,
                        code_export_root=tmp_path_factory.mktemp('jerk_solver'),build_on_init=True)


def test_world_jerk_residual_and_propagated_acceleration(controller):
    # Rotated body-X force 2N / 10kg -> world-Y .2m/s². Difference
    # from preceding .15m/s² over .05s, normalized by .1 -> residual10.
    ocp=controller._solver.acados_ocp
    model=ocp.model
    fn=ca.Function('check_jerk',[model.x,model.u,model.p],
                   [model.cost_y_expr[-3:],model.disc_dyn_expr[-3:]])
    x=np.zeros(34); x[0]=-3; x[6]=x[9]=np.sqrt(.5)
    x[29:31]=[10,5]; x[31:34]=[0,.15,0]
    u=np.zeros(16); u[0]=1.
    p=np.r_[np.zeros(16),pack_physical_parameters(controller.task_params)]
    residual, memory=fn(x,u,p)
    np.testing.assert_allclose(np.array(residual).ravel(),[0,10,0],atol=1e-12)
    np.testing.assert_allclose(np.array(memory).ravel(),[0,.2,0],atol=1e-12)
    # Empty fuel makes acceleration zero despite nonzero commanded duty.
    x[30]=0.
    residual,memory=fn(x,u,p)
    np.testing.assert_allclose(np.array(residual).ravel(),[0,-30,0],atol=1e-12)
    np.testing.assert_allclose(np.array(memory).ravel(),0,atol=1e-12)


def test_previous_executed_acceleration_changes_first_control(controller):
    x=np.zeros(13); x[0]=-1.5; x[6]=1.
    ref=np.repeat(x[None],21,axis=0)
    accelerations=[]
    for previous in ([.1,0,0],[-.1,0,0]):
        result=controller.solve(x,reference_states=ref,previous_acceleration_W=previous)
        assert result.accepted, result.failure_reason
        accelerations.append((result.predicted_state[1,3]-x[3])/.05)
    assert accelerations[0]>.05
    assert accelerations[1]<-.05


def test_active_jerk_requires_measured_boundary(controller):
    x=np.zeros(13); x[0]=-1.5; x[6]=1
    with pytest.raises(ValueError,match='previous_acceleration'):
        controller.solve(x)


def test_slew_applies_to_ignition_and_internal_block_boundaries(tmp_path):
    p=replace(TaskParams(),mean_motion_rad_s=0.,duty_rate_per_s=2.)
    model=MatchedExecutor(7.,np.zeros(16)); model.remaining_fuel=5.
    expert=AcadosExpert(p,executor_model=model,code_export_root=tmp_path)
    x=np.zeros(13); x[0]=-3; x[6]=1
    result=expert.solve(x)
    assert result.accepted,result.failure_reason
    changes=np.diff(np.vstack((np.zeros(16),result.predicted_duty)),axis=0)
    assert np.max(np.abs(changes))<=.100001
    np.testing.assert_allclose(result.predicted_duty[:3],np.tile(result.predicted_duty[0],(3,1)),atol=1e-6)
