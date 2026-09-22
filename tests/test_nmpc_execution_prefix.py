"""Regression checks for horizon extension and consumed-prefix warm starts."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from srb.nmpc import TaskParams
from srb.nmpc.acados_expert import _warm_guess
from srb.nmpc.rcs import rcs_matrix_from_task
from scripts import validate_nmpc_replay as replay


def test_closed_loop_cli_defaults_to_40_plan_3_execute(monkeypatch):
    import sys
    captured=[]
    monkeypatch.setattr(sys,'argv',['validate_nmpc_replay.py','replay','--plan','unused',
                                 '--output','unused','--closed-loop'])
    monkeypatch.setattr(replay,'replay',lambda args: captured.append(args))
    replay.main()
    assert captured[0].horizon_steps==40
    assert captured[0].execute_steps==3


def test_open_loop_cli_preserves_saved_plan_semantics(monkeypatch):
    import sys
    captured=[]
    monkeypatch.setattr(sys,'argv',['validate_nmpc_replay.py','replay','--plan','unused','--output','unused'])
    monkeypatch.setattr(replay,'replay',lambda args: captured.append(args))
    replay.main()
    assert captured[0].horizon_steps is None
    assert captured[0].execute_steps==1


def test_explicit_closed_loop_settings_override_defaults(monkeypatch):
    import sys
    captured=[]
    monkeypatch.setattr(sys,'argv',['validate_nmpc_replay.py','replay','--plan','unused',
        '--output','unused','--closed-loop','--horizon-steps','20','--execute-steps','1'])
    monkeypatch.setattr(replay,'replay',lambda args: captured.append(args))
    replay.main()
    assert (captured[0].horizon_steps,captured[0].execute_steps)==(20,1)


def test_extended_horizon_covers_all_intervals():
    params = replace(TaskParams(), horizon_steps=40,
                     action_block_lengths=(3, 3, 3, 3, 2, 2, 2, 2) * 2)
    assert params.action_block_starts[-1] == 38
    assert params.horizon_seconds == 2.0


@pytest.mark.parametrize('blocks', [(3, 0, 17), (3, -1, 18), (3.5, 16.5), (3, 3)])
def test_invalid_blocks_rejected(blocks):
    with pytest.raises(ValueError):
        replace(TaskParams(), action_block_lengths=blocks)


@pytest.mark.parametrize('consumed', [1, 3, 5, 20])
def test_next_solve_starts_at_first_unexecuted_action(consumed):
    assert hasattr(replay, 'warm_after_prefix'), 'multi-step warm start is missing'
    params = TaskParams()
    x = np.zeros((21, 13))
    x[:, 6] = 1
    u = np.repeat((np.arange(20) / 20)[:, None], 16, axis=1)
    result = SimpleNamespace(predicted_state=x, predicted_duty=u)
    previous = u[consumed - 1].copy()
    warm = replay.warm_after_prefix(result, consumed, previous)
    _, next_u = _warm_guess(params, x[0], previous, warm, rcs_matrix_from_task(params))
    np.testing.assert_allclose(next_u[0], min(consumed, 19) / 20)
    np.testing.assert_array_equal(warm['previous_duty'], previous)
    np.testing.assert_array_equal(result.predicted_duty, u)
    if consumed == 1:
        np.testing.assert_array_equal(warm['x'], x)
        np.testing.assert_array_equal(warm['u'], u)


@pytest.mark.parametrize('samples, achieved, duration', [(20, False, .95), (21, True, 1.)])
def test_hover_requires_full_second_between_samples(samples, achieved, duration):
    params = TaskParams()
    states = np.zeros((samples, 13))
    states[:, 0] = -1.5
    states[:, 6] = 1
    result = replay.metrics(states, np.zeros((samples - 1, 16)), params, states)
    assert result['hover_hold_achieved'] == achieved
    assert result['longest_hover_hold_s'] == pytest.approx(duration)
