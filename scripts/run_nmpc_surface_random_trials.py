#!/usr/bin/env python3
"""Run five reproducible randomized NMPC surface-approach recordings.

Each trial randomizes the stationary target's world translation and the
actor's target-relative root start on the target -X side at 14--16 m. The
controller and geometry goal remain the same: actor +Z top to target -X bus
face, with a 0.30 m bus-face gap. The default 120 s approach stage keeps the
15 m displacement within the matched RCS acceleration envelope. Raw replay artifacts and subtitle-annotated
videos are saved under one new experiment directory.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from srb.nmpc.surface_trials import generate_trial_specs, trial_to_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / 'logs/nmpc_validation_20260907/approach_hcw_plan'
DEFAULT_ISAAC_PYTHON = '/home/ubuntu/isaac-sim-4.5/python.sh'


def _triple(values):
    return [str(float(value)) for value in values]


def build_replay_command(trial, output, plan, *, steps, seed, isaac_python,
                         wrapper, turn_s=12., approach_s=120., fill_intensity=600.):
    """Build the exact subprocess argv for one video trial."""
    return [
        'env', '-u', 'CUDA_VISIBLE_DEVICES',
        'CUDA_DEVICE_ORDER=PCI_BUS_ID', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
        'timeout', '-k', '15', '600',
        str(wrapper), '--', str(isaac_python), '-u',
        'scripts/validate_nmpc_replay.py', 'replay',
        '--plan', str(plan), '--output', str(output),
        '--video', '--stable-video-lighting', '--video-fill-intensity', str(float(fill_intensity)),
        '--surface-display-materials', '--device', 'cuda:1', '--closed-loop',
        '--matched-executor', '--surface-approach',
        '--surface-stages', str(float(turn_s)), str(float(approach_s)),
        '--horizon-steps', '40', '--execute-steps', '3', '--steps', str(int(steps)),
        '--seed', str(int(seed)), '--start', *_triple(trial.initial_relative_root_position_m),
        '--target-position', *_triple(trial.target_position_m),
    ]


def build_render_command(folder, output, *, isaac_python, wrapper):
    return [
        str(wrapper), '--', str(isaac_python), '-u',
        'scripts/render_nmpc_surface_eval.py', str(folder), '--output', str(output),
    ]


def _run(command, cwd, log_path):
    with log_path.open('w') as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                   check=False)
    return completed.returncode


def _trial_result(folder, spec, replay_returncode, render_returncode):
    record = trial_to_json(spec)
    record.update({
        'folder': str(folder),
        'replay_returncode': replay_returncode,
        'render_returncode': render_returncode,
        'raw_video': str(folder / 'replay.mp4') if (folder / 'replay.mp4').exists() else None,
        'annotated_video': str(folder / 'surface_approach_eval.mp4') if (folder / 'surface_approach_eval.mp4').exists() else None,
    })
    replay_path = folder / 'replay.json'
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        record['complete_success'] = replay.get('complete_success')
        record['executed_steps'] = replay.get('steps')
        record['requested_steps'] = replay.get('requested_steps')
        record['stopped_reason'] = replay.get('stopped_reason')
        record['max_contact_force_N'] = replay.get('max_contact_force_N')
        record['solver_hot_p50_ms'] = replay.get('solver_hot_p50_ms')
        record['solver_hot_p99_ms'] = replay.get('solver_hot_p99_ms')
        record['solver_replan_deadline_miss_fraction'] = replay.get('solver_replan_deadline_miss_fraction')
        record['surface_approach'] = replay.get('surface_approach')
    else:
        record['complete_success'] = False
        record['executed_steps'] = 0
        record['requested_steps'] = None
        record['stopped_reason'] = 'replay.json was not produced'
    return record


def write_report(path, *, config, trials):
    lines = [
        '# Randomized NMPC surface-approach trials',
        '',
        'Five independent video trials use randomized world translation for the target and randomized actor starts on the target -X side at approximately 15 m.',
        '',
        '## Fixed contract',
        '',
        '- target attitude: identity; stationary target;',
        '- target front: -X bus face without solar-panel extension;',
        '- actor approach face: +Z top face;',
        '- desired bus-face gap: 0.30 m;',
        '- matched free-floating NMPC, N=40, K=3, dt=0.05 s, CPU PhysX, GPU Kit cuda:1;',
        f"- reference staging: {config['turn_seconds']} s quintic turn, {config['approach_seconds']} s quintic approach, then hold;",
        '- no duty-rate, jerk, solver tolerance, or safety-limit changes.',
        '',
        '## Randomization',
        '',
        f"`{json.dumps(config, ensure_ascii=False, sort_keys=True)}`",
        '',
        '| Trial | Initial distance [m] | Target world position [m] | Executed | Complete | Final face gap [m] | Final facing error [deg] | Min mesh plane gap [m] |',
        '|---:|---:|---|---:|---|---:|---:|---:|',
    ]
    for trial in trials:
        surface = trial.get('surface_approach') or {}
        final = surface.get('final') or {}
        lines.append(
            f"| {trial['trial_index'] + 1} | {trial['initial_distance_m']:.4f} | "
            f"[{trial['target_position_m'][0]:.3f}, {trial['target_position_m'][1]:.3f}, {trial['target_position_m'][2]:.3f}] | "
            f"{trial.get('executed_steps', 0)}/{trial.get('requested_steps') or '?'} | "
            f"{trial.get('complete_success', False)} | {final.get('face_gap_m', float('nan')):.6f} | "
            f"{final.get('facing_error_deg', float('nan')):.6f} | {surface.get('min_mesh_plane_gap_m', float('nan')):.6f} |"
        )
    lines.extend([
        '',
        'Each trial directory contains `trial_spec.json`, `command.txt`, `replay.log`, raw `replay.mp4`, annotated `surface_approach_eval.mp4`, `replay.json`, `surface_metrics.npz`, and the controller/physics audit artifacts when the process completed far enough.',
        '',
        'The report records actual results; a failed trial is not converted into a success by post-processing.',
        '',
    ])
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New experiment directory; it must not already exist')
    parser.add_argument('--plan', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--seed', type=int, default=20260918)
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--turn-seconds', type=float, default=12.)
    parser.add_argument('--approach-seconds', type=float, default=120.)
    parser.add_argument('--fill-intensity', type=float, default=600.)
    parser.add_argument('--isaac-python', default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.count <= 0 or args.steps <= 0:
        parser.error('--count and --steps must be positive')
    if args.turn_seconds <= 0 or args.approach_seconds <= 0 or args.fill_intensity < 0:
        parser.error('stage durations must be positive and fill intensity nonnegative')
    if args.output.exists():
        raise FileExistsError(f'output directory already exists: {args.output}')
    if not args.plan.exists():
        raise FileNotFoundError(args.plan)

    specs = generate_trial_specs(args.seed, args.count)
    config = {
        'seed': args.seed, 'count': args.count, 'steps': args.steps,
        'target_position_bounds_m': [-2., 2.],
        'initial_distance_bounds_m': [14., 16.],
        'lateral_offset_bounds_m': [-2., 2.],
        'target_quaternion_wxyz': [1., 0., 0., 0.],
        'turn_seconds': args.turn_seconds, 'approach_seconds': args.approach_seconds,
        'fill_intensity': args.fill_intensity,
        'horizon_steps': 40, 'execute_steps': 3, 'control_period_s': .05,
        'plan': str(args.plan), 'cwd': str(ROOT),
    }
    commands = []
    for spec in specs:
        folder = args.output / f'trial_{spec.trial_index + 1:02d}'
        raw_command = build_replay_command(
            spec, folder, args.plan, steps=args.steps, seed=args.seed + spec.trial_index,
            isaac_python=args.isaac_python, wrapper=ROOT / 'srb_0.0.5_env.sh',
            turn_s=args.turn_seconds, approach_s=args.approach_seconds,
            fill_intensity=args.fill_intensity,
        )
        commands.append({'trial_index': spec.trial_index, 'folder': str(folder),
                         'argv': raw_command, 'command': shlex.join(raw_command)})
    if args.dry_run:
        print(json.dumps({'config': config, 'commands': commands}, indent=2))
        return 0

    args.output.mkdir(parents=True, exist_ok=False)
    invocation = {
        'cwd': str(ROOT),
        'argv': sys.argv,
        'command': shlex.join(sys.argv),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS'),
        'openblas_num_threads': os.environ.get('OPENBLAS_NUM_THREADS'),
    }
    (args.output / 'runner_invocation.json').write_text(json.dumps(invocation, indent=2) + '\n')
    config['runner_command'] = invocation['command']
    (args.output / 'experiment_config.json').write_text(json.dumps(config, indent=2) + '\n')
    results = []
    for spec, command in zip(specs, commands):
        folder = Path(command['folder'])
        staging_log = args.output / f'trial_{spec.trial_index + 1:02d}.replay.log'
        replay_returncode = _run(command['argv'], ROOT, staging_log)
        # validate_nmpc_replay creates its output directory itself and refuses
        # an existing one. Attach runner metadata only after that contract has
        # been honored; preserve a folder even when startup failed early.
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'trial_spec.json').write_text(json.dumps(trial_to_json(spec), indent=2) + '\n')
        (folder / 'command.txt').write_text(command['command'] + '\n')
        staging_log.replace(folder / 'replay.log')
        render_returncode = 1
        if (folder / 'replay.json').exists() and (folder / 'replay.mp4').exists():
            render_command = build_render_command(
                folder, folder / 'surface_approach_eval.mp4',
                isaac_python=args.isaac_python, wrapper=ROOT / 'srb_0.0.5_env.sh',
            )
            (folder / 'render_command.txt').write_text(shlex.join(render_command) + '\n')
            render_returncode = _run(render_command, ROOT, folder / 'render.log')
        results.append(_trial_result(folder, spec, replay_returncode, render_returncode))
        (args.output / 'trials.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(results[-1], indent=2), flush=True)

    summary = {
        'experiment': str(args.output), 'config': config,
        'successful_trials': sum(bool(item.get('complete_success')) for item in results),
        'attempted_trials': len(results), 'trials': results,
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_report(args.output / 'report.md', config=config, trials=results)
    print(json.dumps({'successful_trials': summary['successful_trials'],
                      'attempted_trials': summary['attempted_trials'],
                      'report': str(args.output / 'report.md')}, indent=2), flush=True)
    return 0 if len(results) == args.count else 1


if __name__ == '__main__':
    raise SystemExit(main())
