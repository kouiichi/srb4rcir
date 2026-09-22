"""Fast contract tests for the standalone 20 Hz NMPC components."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from srb.nmpc import (
    AcadosExpert,
    IpoptOracle,
    SafetyAllocator,
    Srb16Adapter,
    SrbAdapter,
    TaskParams,
    build_rcs_matrix,
    build_srb16_matrix,
    build_srb8_matrix,
    load_nmpc_config,
    validate_rcs_geometry,
    validate_srb8_geometry,
    DaggerThresholds,
    DaggerTrigger,
    diagnostic_rows,
    write_episode_hdf5,
)
from srb.nmpc.constraints import koz_terms, validate_solution
from srb.nmpc.dynamics import casadi_rk4_map, cw_acceleration, rk4_step
from srb.nmpc.quaternion import shortest_quaternion_error
from srb.nmpc.types import ExpertResult


def _state(position=(2.0, 0.0, 0.0)) -> np.ndarray:
    return np.array(
        [*position, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def test_global_timing_contract() -> None:
    params = TaskParams()
    assert params.control_period_s == pytest.approx(0.05)
    assert params.horizon_steps == 20
    assert params.horizon_seconds == pytest.approx(1.0)
    assert params.rk4_substeps == 5
    assert params.action_block_starts == (0, 3, 6, 9, 12, 14, 16, 18)


def test_yaml_is_bound_to_taskparams_and_solver_config() -> None:
    config = load_nmpc_config(Path(__file__).parents[1] / "configs/nmpc/default.yaml")
    assert config.task_params.control_period_s == pytest.approx(0.05)
    assert config.task_params.horizon_steps == 20
    assert config.task_params.rk4_substeps == 5
    assert config.task_params.tmax_N == pytest.approx(0.1)
    assert config.solver.production == "acados"
    assert config.solver.qp_solver == "PARTIAL_CONDENSING_HPIPM"
    assert config.solver.nlp_solver_type == "SQP_RTI"
    assert config.resolved_code_export_root(Path(__file__).parents[1]).name == "generated"


def test_yaml_binding_rejects_contract_mismatch() -> None:
    with pytest.raises(ValueError, match="control_frequency_hz"):
        TaskParams.from_mapping(
            {
                "control_frequency_hz": 25,
                "control_period_s": 0.05,
                "horizon_steps": 20,
                "rk4_substeps": 5,
                "rk4_substep_dt_s": 0.01,
                "action_block_lengths": [3, 3, 3, 3, 2, 2, 2, 2],
            }
        )


def test_taskparams_mapping_applies_non_default_values() -> None:
    params = TaskParams.from_mapping(
        {
            "control_period_s": 0.05,
            "horizon_steps": 20,
            "rk4_substeps": 5,
            "rk4_substep_dt_s": 0.01,
            "action_block_lengths": [3, 3, 3, 3, 2, 2, 2, 2],
            "actuator": {
                "channels": 16,
                "tmax_N": 0.2,
                "rho_m": 0.30,
                "delta_m": 0.05,
                "duty_rate_per_s": 10.0,
                "reverse_pair_limit": 0.8,
            },
            "model": {"camera_target_point_T_m": [0.1, 0.2, 0.3]},
        }
    )
    assert params.tmax_N == pytest.approx(0.2)
    assert params.duty_rate_per_s == pytest.approx(10.0)
    assert params.reverse_pair_limit == pytest.approx(0.8)
    np.testing.assert_allclose(params.camera_target_point_T_m, [0.1, 0.2, 0.3])


def test_hcw_acceleration_contract() -> None:
    params = TaskParams()
    state = _state((2.0, -0.5, 0.25))
    state[3:6] = (0.1, -0.2, 0.3)
    expected = np.array(
        [
            3.0 * params.mean_motion_rad_s**2 * state[0]
            + 2.0 * params.mean_motion_rad_s * state[4],
            -2.0 * params.mean_motion_rad_s * state[3],
            -params.mean_motion_rad_s**2 * state[2],
        ]
    )
    np.testing.assert_allclose(
        cw_acceleration(state[:3], state[3:6], params.mean_motion_rad_s), expected
    )


def test_quaternion_sign_equivalence_and_rk4_normalization() -> None:
    params = TaskParams()
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(shortest_quaternion_error(reference, -reference)) == pytest.approx(0.0)
    state = _state()
    state[10:13] = (0.03, -0.02, 0.01)
    next_state = rk4_step(state, np.zeros(16), params)
    assert np.linalg.norm(next_state[6:10]) == pytest.approx(1.0, abs=1.0e-12)


def test_rcs_rank_condition_faults_and_reachable_residual() -> None:
    result = validate_rcs_geometry(raise_on_failure=True)
    assert result["rank"] == 6
    assert result["minimum_fault_rank"] == 6
    assert result["normalized_condition"] <= 1.5
    assert result["maximum_reachable_residual"] < 1.0e-4


def test_allocation_and_exact_executable_wrench() -> None:
    params = TaskParams()
    matrix = build_rcs_matrix()
    duty = np.full(16, 0.05, dtype=np.float64)
    from srb.nmpc.rcs import allocate_wrench

    result = allocate_wrench(matrix @ duty, params)
    assert result.status == "solved"
    np.testing.assert_allclose(result.wrench6_exec, matrix @ result.duty16, atol=1.0e-12)
    assert result.residual < 1.0e-4


def test_safety_filter_hard_ecbf_and_affine_row() -> None:
    params = TaskParams()
    state = _state()
    matrix = build_rcs_matrix()
    duty = np.full(16, 0.05, dtype=np.float64)
    result = SafetyAllocator(params).filter_and_allocate(matrix @ duty, state, np.zeros(16))
    assert result.status in {"solved", "intervened", "residual_exceeded"}
    assert result.constraint_margins["koz_h"] >= 0.0
    assert result.constraint_margins["koz_ecbf"] >= -1.0e-7
    np.testing.assert_allclose(result.wrench6_exec, matrix @ result.duty16, atol=1.0e-12)

    base, coefficient, _ = SafetyAllocator._ecbf_affine_row(state, params, matrix)
    np.testing.assert_allclose(
        base + coefficient @ duty,
        koz_terms(state, duty, params)[3],
        atol=1.0e-10,
    )


def test_safety_filter_marks_inside_koz_as_failed() -> None:
    params = TaskParams()
    result = SafetyAllocator(params).filter_and_allocate(
        np.zeros(6), _state((0.0, 0.0, 0.0)), np.zeros(16)
    )
    assert result.status == "failed"
    assert result.error_message is not None


def test_current_srb_is_explicit_rank5_contrast() -> None:
    result = validate_srb8_geometry(build_srb8_matrix())
    assert result["rank"] == 5
    assert not result["pure_tau_z_reachable"]
    _, _, diagnostics = SrbAdapter().step(np.full(16, 0.1, dtype=np.float64))
    assert diagnostics["rank"] == 5
    assert diagnostics["label_valid"] is False


def test_srb16_adapter_is_full_rank_and_exact() -> None:
    params = TaskParams()
    matrix = build_srb16_matrix(power_N=params.tmax_N)
    np.testing.assert_allclose(matrix, build_rcs_matrix(), atol=1.0e-14)
    result = validate_rcs_geometry(matrix, tmax_N=params.tmax_N, raise_on_failure=True)
    assert result["rank"] == 6
    _, _, diagnostics = Srb16Adapter(task_params=params).step(
        np.full(16, 0.1, dtype=np.float64)
    )
    assert diagnostics["rank"] == 6
    assert diagnostics["label_valid"] is True
    assert diagnostics["sample_type"] == "normal"
    assert diagnostics["projection_residual"] < 1.0e-12


def test_casadi_rk4_jacobian_if_available() -> None:
    ca = pytest.importorskip("casadi")
    params = TaskParams()
    x = ca.SX.sym("x", 13)
    u = ca.SX.sym("u", 16)
    result = casadi_rk4_map(x, u, params)
    jacobian = ca.Function("nmpc_test_jac", [x, u], [ca.jacobian(result, x)])
    value = np.asarray(
        jacobian(_state(), np.zeros(16)),
        dtype=np.float64,
    )
    assert value.shape == (13, 13)
    assert np.all(np.isfinite(value))


@pytest.mark.slow
def test_ipopt_and_acados_objective_agreement_if_configured() -> None:
    pytest.importorskip("casadi")
    if not os.environ.get("ACADOS_SOURCE_DIR"):
        pytest.skip("run through srb_0.0.5_env.sh to enable acados")
    params = TaskParams()
    x0 = _state()
    ipopt = IpoptOracle(max_iterations=40).solve(x0, params)
    acados = AcadosExpert(params, nlp_solver_type="SQP", max_iterations=20).solve(x0)
    assert ipopt.accepted
    assert acados.accepted
    assert abs(ipopt.objective - acados.objective) / max(abs(ipopt.objective), 1.0) < 0.02
    assert acados.constraint_margins["action_block_residual"] < 1.0e-6
    assert acados.constraint_margins["dynamics_residual"] < 1.0e-6


@pytest.mark.slow
def test_acados_fov_constraints_if_configured() -> None:
    pytest.importorskip("casadi")
    if not os.environ.get("ACADOS_SOURCE_DIR"):
        pytest.skip("run through srb_0.0.5_env.sh to enable acados")
    # The hover point and observed target feature are intentionally distinct:
    # the service vehicle approaches the target origin while stopping at a
    # positive camera depth on the +x side.
    params = TaskParams(
        enable_fov=True,
        target_point_T_m=np.array([1.5, 0.0, 0.0]),
        camera_axis_B=np.array([-1.0, 0.0, 0.0]),
    )
    result = AcadosExpert(params, nlp_solver_type="SQP_RTI", max_iterations=1).solve(
        _state(), params
    )
    assert result.accepted
    assert result.constraint_margins["fov_depth"] >= 0.0
    assert result.constraint_margins["fov_cone"] >= 0.0


def test_zero_rollout_quality_gate() -> None:
    params = TaskParams()
    states = [_state()]
    for _ in range(params.horizon_steps):
        states.append(rk4_step(states[-1], np.zeros(16), params))
    duties = np.zeros((params.horizon_steps, 16), dtype=np.float64)
    accepted, margins, reason = validate_solution(np.asarray(states), duties, params)
    assert accepted, reason
    assert margins["dynamics_residual"] < 1.0e-10


def test_hdf5_schema_and_dagger_budget() -> None:
    params = TaskParams()
    state = _state()
    duty = np.zeros((params.horizon_steps, 16), dtype=np.float64)
    states = np.stack([state] * (params.horizon_steps + 1))
    result = ExpertResult(
        status="success",
        solver_status_code=0,
        iterations=1,
        kkt=0.0,
        runtime_s=0.001,
        objective=0.0,
        constraint_margins={"koz_ecbf": 1.0},
        slacks={"recovery": 0.0},
        allocation_residual=0.0,
        warm_start={},
        predicted_state=states,
        predicted_duty=duty,
        predicted_wrench=np.zeros((params.horizon_steps, 6), dtype=np.float64),
        actual_duty=np.zeros(16, dtype=np.float64),
        actual_wrench=np.zeros(6, dtype=np.float64),
        quality_verdict="accepted",
        sample_type="normal",
        action_blocks=np.zeros((8, 16), dtype=np.float64),
    )
    with tempfile.TemporaryDirectory() as directory:
        output = write_episode_hdf5(
            f"{directory}/episode.h5",
            episode_id="test-0",
            rgb=np.zeros((1, 2, 2, 3), dtype=np.uint8),
            observed_state=state[None, :],
            timestamps_s=np.array([0.0]),
            results=[result],
        )
        assert output.exists()
        import h5py

        with h5py.File(output, "r") as handle:
            assert handle["rgb"].shape == (1, 2, 2, 3)
            assert handle["predicted_state"].shape == (1, 21, 13)
            assert handle["action_blocks"].shape == (1, 8, 16)
        rows = diagnostic_rows("test-0", np.array([0.0]), [result])
        assert rows[0]["sample_type"] == "normal"

    thresholds = DaggerThresholds.from_expert_data(
        safety_margins=np.arange(100.0),
        ensemble_uncertainty=np.arange(100.0),
    )
    trigger = DaggerTrigger(thresholds)
    decisions = [
        trigger.decide(safety_margin=-1.0, ensemble_uncertainty=99.0)
        for _ in range(20)
    ]
    assert sum(decision.query for decision in decisions) <= 2
    assert trigger.queried_steps / trigger.visited_steps <= 0.10
