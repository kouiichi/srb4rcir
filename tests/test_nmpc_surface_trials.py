"""Deterministic randomized initial-condition contracts for surface trials."""
import numpy as np

from srb.nmpc.surface_trials import generate_trial_specs, trial_to_json
from scripts.run_nmpc_surface_random_trials import build_replay_command


def test_trial_specs_are_reproducible_and_have_fifteen_meter_separation():
    first = generate_trial_specs(seed=20260918, count=5)
    second = generate_trial_specs(seed=20260918, count=5)
    assert [trial_to_json(x) for x in first] == [trial_to_json(x) for x in second]
    assert len(first) == 5
    for trial in first:
        assert np.all(np.asarray(trial.target_position_m) >= -2.)
        assert np.all(np.asarray(trial.target_position_m) <= 2.)
        assert 14.0 <= trial.initial_distance_m <= 16.0
        assert trial.initial_relative_root_position_m[0] < 0.
        np.testing.assert_allclose(
            np.asarray(trial.target_position_m) + np.asarray(trial.initial_relative_root_position_m),
            trial.initial_actor_world_root_position_m,
        )


def test_trial_specs_change_initial_conditions_without_changing_goal_definition():
    trials = generate_trial_specs(seed=7, count=5)
    assert len({tuple(np.round(x.target_position_m, 8)) for x in trials}) == 5
    assert len({tuple(np.round(x.initial_relative_root_position_m, 8)) for x in trials}) == 5
    for trial in trials:
        assert trial.target_quaternion_wxyz == (1., 0., 0., 0.)
        assert trial.goal_surface_gap_m == .30
        assert trial.front_definition == 'target -X bus face; actor +Z top face'


def test_replay_command_keeps_recorded_n40_k3_surface_contract():
    trial = generate_trial_specs(seed=9, count=1)[0]
    command = build_replay_command(
        trial, '/tmp/trial', '/tmp/plan', steps=1200, seed=9,
        isaac_python='/isaac/python.sh', wrapper='/repo/srb_0.0.5_env.sh',
    )
    assert '--video' in command
    assert '--physics-only' not in command
    assert command[command.index('--horizon-steps') + 1] == '40'
    assert command[command.index('--execute-steps') + 1] == '3'
    assert command[command.index('--surface-stages') + 1:command.index('--surface-stages') + 3] == ['12.0', '120.0']
    assert command[command.index('--target-position') + 1:command.index('--target-position') + 4] == [str(x) for x in trial.target_position_m]
