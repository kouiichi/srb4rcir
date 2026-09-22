#!/usr/bin/env python3
"""Run serial SRB feedback-prefix experiments, audit saved actions, and plot them.

Run via srb_0.0.5_env.sh and Isaac Sim Python. No surrogate dynamics are used
for the measured trajectories; each case launches the existing PhysX replay.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CASES = ((20, 5), (20, 1), (40, 1), (40, 3), (40, 5))


def read_json(path):
    return json.loads(path.read_text())


def audit_run(folder, horizon, prefix, requested, require_complete=False):
    result = read_json(folder / 'replay.json')
    replay = np.load(folder / 'replay.npz')
    pred = np.load(folder / 'predictions.npz')
    steps, accepted = pred['steps'], pred['accepted']
    assert result['horizon_steps'] == horizon
    assert result['execute_steps'] == prefix
    assert result['requested_steps'] == requested
    assert pred['states'].shape == (len(steps), horizon + 1, 13)
    assert pred['duties'].shape == (len(steps), horizon, 16)
    np.testing.assert_array_equal(steps, np.arange(len(steps)) * prefix)
    expected = replay['states'][steps].copy()
    # AcadosExpert._as_state normalizes the float32 PhysX quaternion on entry.
    expected[:, 6:10] /= np.linalg.norm(expected[:, 6:10], axis=1)[:, None]
    np.testing.assert_allclose(pred['states'][:, 0], expected, atol=1e-9, rtol=0)
    for j, start in enumerate(steps):
        count = min(prefix, len(replay['duties']) - start)
        if not accepted[j]:
            assert count == 0, 'rejected plan was executed'
            continue
        np.testing.assert_allclose(replay['duties'][start:start + count],
                                   np.clip(pred['duties'][j, :count], 0, 1), atol=1e-12)
    assert len(replay['states']) == len(replay['duties']) + 1
    assert np.isfinite(replay['states']).all()
    if require_complete:
        assert result['steps'] == requested and result['failure'] is None, result
        assert result['stopped_reason'] is None, result
    physics = read_json(folder / 'physics_audit.json')
    assert physics['env_device'] == 'cpu', physics
    assert physics['physx_mass_tensor_device'] == 'cpu', physics
    return result


def run_case(root, stage, horizon, prefix, steps, resume=False):
    folder = root / stage / f'N{horizon}_K{prefix}'
    logfile = root / stage / f'N{horizon}_K{prefix}.log'
    if resume and (folder / 'replay.json').exists():
        result = audit_run(folder, horizon, prefix, steps, require_complete=stage == 'debug')
        print(f'RESUME audited {stage} N={horizon} K={prefix}', flush=True)
        return result
    if folder.exists() or logfile.exists():
        raise FileExistsError(f'Refusing to overwrite {folder} or {logfile}')
    logfile.parent.mkdir(parents=True, exist_ok=True)
    argv = ['env', '-u', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER=PCI_BUS_ID',
            'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
            str(ROOT / 'srb_0.0.5_env.sh'), '--', '/home/ubuntu/isaac-sim-4.5/python.sh',
            '-u', 'scripts/validate_nmpc_replay.py', 'replay',
            '--plan', 'logs/nmpc_validation_20260907/approach_hcw_plan',
            '--output', str(folder), '--physics-only', '--device', 'cuda:1',
            '--closed-loop', '--align-goal', '--seed', '0', '--steps', str(steps),
            '--horizon-steps', str(horizon), '--execute-steps', str(prefix)]
    command = shlex.join(argv)
    with (root / 'commands.jsonl').open('a') as handle:
        handle.write(json.dumps({'stage': stage, 'cwd': str(ROOT), 'command': command,
                                 'output': str(folder), 'log': str(logfile)}) + '\n')
    print(f'START {stage} N={horizon} K={prefix} steps={steps}', flush=True)
    with logfile.open('w') as handle:
        completed = subprocess.run(argv, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                   timeout=480, check=False)
    if completed.returncode:
        raise RuntimeError(f'Process failed ({completed.returncode}); inspect {logfile}')
    result = audit_run(folder, horizon, prefix, steps, require_complete=stage == 'debug')
    print(f"DONE {stage} N={horizon} K={prefix}: {result['steps']}/{steps}, "
          f"error={result['final_position_error_m']:.6g}m success={result['complete_success']} "
          f"stop={result['stopped_reason']}", flush=True)
    return result


def draw(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    for (horizon, prefix), color in zip(CASES, plt.cm.tab10.colors):
        folder = root / 'full' / f'N{horizon}_K{prefix}'
        result = audit_run(folder, horizon, prefix, 1200)
        data = np.load(folder / 'replay.npz')
        pred = np.load(folder / 'predictions.npz')
        x, u, refs = data['states'], data['duties'], data['refs']
        t = np.arange(len(x)) * .05
        error = np.linalg.norm(x[:, :3] - refs[:, :3], axis=1)
        speed = np.linalg.norm(x[:, 3:6] - refs[:, 3:6], axis=1)
        q = x[:, 6:10] / np.linalg.norm(x[:, 6:10], axis=1)[:, None]
        angle = np.rad2deg(2 * np.arccos(np.clip(np.abs(q[:, 0]), 0, 1)))
        clearance = np.linalg.norm(x[:, :3], axis=1) - .5
        good = (error < .1) & (speed < .05) & (angle < 5)
        qualifying = [i for i in range(max(0, len(good) - 20)) if good[i:i + 21].all()]
        result['first_strict_hover_start_s'] = qualifying[0] * .05 if qualifying else None
        result['first_strict_hover_confirmed_s'] = (qualifying[0] + 20) * .05 if qualifying else None
        result['fuel_used_kg'] = 5 - float(np.asarray(result['fuel_remaining_kg']).ravel()[0])
        result['integrated_position_error_m_s'] = float(np.trapz(error, t))
        result['max_attitude_error_deg'] = float(angle.max())
        result['positive_x_overshoot_m'] = float(max(0, np.max(x[:, 0] + 1.5)))
        rows.append(result)
        label = f'N={horizon}, K={prefix}'
        axes[0, 0].plot(x[:, 0], x[:, 1], color=color, label=label)
        for ax, values in zip((axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]),
                              (error, speed, angle, clearance)):
            ax.plot(t, values, color=color, label=label)
        wall = data['solve_wall_s']
        result['solver_hot_max_ms'] = float(np.max(wall[1:]) * 1000)
        result['solver_replan_deadline_miss_count'] = int(np.sum(wall[1:] > prefix * .05))
        axes[1, 2].plot(pred['steps'][1:] * .05, wall[1:] * 1000,
                        color=color, alpha=.8, linewidth=.8, label=label)
        ax3.plot(x[:, 0], x[:, 1], x[:, 2], color=color, label=label)

        # Per-case plans: full predicted horizons + measured path, with actual
        # executed prefix endpoints to distinguish plans from realized motion.
        single, axs = plt.subplots(2, 2, figsize=(13, 10))
        chosen = np.flatnonzero(pred['accepted'])
        chosen = chosen[::max(1, int(np.ceil(len(chosen) / 60)))]
        for j in chosen:
            xp = pred['states'][j]
            tp = (pred['steps'][j] + np.arange(horizon + 1)) * .05
            axs[0, 0].plot(xp[:, 0], xp[:, 1], color='tab:orange', alpha=.3, linewidth=.8)
            axs[0, 1].plot(xp[:, 0], xp[:, 2], color='tab:orange', alpha=.3, linewidth=.8)
            axs[1, 0].plot(tp, xp[:, 0], color='tab:orange', alpha=.3, linewidth=.8)
            ep = min(prefix, len(x) - 1 - int(pred['steps'][j]))
            if ep > 0:
                axs[0, 0].plot(xp[:ep + 1, 0], xp[:ep + 1, 1], color='tab:green', alpha=.7)
        axs[0, 0].plot([], [], color='tab:orange', label='Predicted horizons (sampled)')
        axs[0, 0].plot([], [], color='tab:green', label='Predicted execution prefixes')
        axs[0, 0].plot(x[:, 0], x[:, 1], color=color, label='Measured SRB trajectory')
        axs[0, 1].plot(x[:, 0], x[:, 2], color=color, label='Measured SRB trajectory')
        axs[1, 0].plot(t, x[:, 0], color=color, label='Measured x')
        axs[1, 0].axhline(-1.5, color='black', linestyle='--', label='Goal x')
        axs[1, 1].plot(t, error, color=color, label='Position error [m]')
        axs[1, 1].plot(t, speed, label='Speed [m/s]')
        axs[1, 1].axhline(.1, color='gray', linestyle='--', label='Position threshold')
        for ax, vertical in ((axs[0, 0], 1), (axs[0, 1], 2)):
            ax.add_patch(plt.Circle((0, 0), .5, color='red', alpha=.12))
            ax.scatter([-1.5], [0], marker='*', s=110, c='black', label='Goal')
            ax.scatter(x[0, 0], x[0, vertical], marker='o', color='black', label='Start')
            ax.set(xlabel='x [m]', ylabel=('y [m]' if vertical == 1 else 'z [m]'), aspect='equal')
        axs[1, 0].set(xlabel='Simulation time [s]', ylabel='x [m]')
        axs[1, 1].set(xlabel='Simulation time [s]')
        for ax in axs.flat:
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        single.suptitle(f'{label}: {result["steps"]}/1200 SRB steps | '
                       f'strict complete success: {result["complete_success"]}')
        single.tight_layout()
        single.savefig(folder / 'planned_and_executed.png', dpi=160)
        single.savefig(folder / 'planned_and_executed.pdf')
        plt.close(single)
    axes[0, 0].add_patch(plt.Circle((0, 0), .5, color='red', alpha=.12))
    axes[0, 0].scatter([-1.5], [0], marker='*', c='black', s=100)
    axes[0, 0].set(xlabel='x [m]', ylabel='y [m]', aspect='equal', title='Measured SRB paths; shaded KOZ')
    for ax, title, ylabel in zip((axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]),
                               ('Position error', 'Relative speed', 'Attitude error', 'KOZ clearance', 'Solve time (first build excluded)'),
                               ('m', 'm/s', 'deg', 'm', 'ms')):
        ax.set(xlabel='Simulation time [s]', ylabel=ylabel, title=title)
    for ax, threshold in ((axes[0, 1], .1), (axes[0, 2], .05), (axes[1, 0], 5), (axes[1, 1], 0), (axes[1, 2], 50)):
        ax.axhline(threshold, color='gray', linestyle='--', linewidth=1)
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('NMPC horizon / execution-prefix comparison | 20 Hz | SRB CPU PhysX / GPU Kit')
    fig.tight_layout()
    fig.savefig(root / 'comparison.png', dpi=170)
    fig.savefig(root / 'comparison.pdf')
    ax3.scatter([-1.5], [0], [0], marker='*', c='black', s=130, label='Goal')
    ax3.scatter([-3], [.2], [.15], c='black', label='Start')
    ax3.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title='Measured SRB trajectories (axis scales differ)')
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(root / 'trajectories_3d.png', dpi=170)
    plt.close('all')
    (root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def write_report(root, rows):
    lines = ['# NMPC horizon and executed-prefix comparison — 2026-09-16', '',
             '## Protocol', '',
             f'- Working directory: `{ROOT}`.',
             '- Serial runs, same seed 0, requested 1200 control steps / 60 s each; dt = 0.05 s.',
             '- Existing measured-state closed-loop SRB rendezvous replay; `--physics-only --closed-loop --align-goal --device cuda:1`.',
             '- CUDA_VISIBLE_DEVICES unset; CUDA_DEVICE_ORDER=PCI_BUS_ID; OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1.',
             '- GPU Kit, CPU environment and CPU PhysX mass tensors; actual per-run device audit in physics_audit.json.',
             '- Initial state: position [-3, 0.2, 0.15] m, unit quaternion, zero linear/angular velocity. Goal [-1.5,0,0] m, unit quaternion.',
             '- Expert unchanged: HCW n=0.001027 rad/s, nominal mass 12 kg, inertia diag[0.20,0.24,0.30], COM zero, 16 one-sided RCS duties, Tmax=0.1 N. SRB remains stationary/free-floating n=0; fuel-dependent real executor retained.',
             '- N=20: 1 s horizon and blocks [3,3,3,3,2,2,2,2]. N=40: 2 s horizon and that pattern repeated twice (16 blocks). RK4 uses five 0.01 s substeps per predicted interval. Costs/constraints unchanged.',
             '- K is the number of consecutive predicted actions executed before replanning; each action lasts 0.05 s. No early corrective replanning or safety filter added; measured collision/KOZ checks remain active every step.',
             '- K>1 warm starts pre-shift by K-1 before the solver\'s existing one-step action shift; actual last executed duty is retained. K=1 preserves historical warm-start behavior.',
             '- Strict hover: position error <0.1 m, speed <0.05 m/s, attitude error <5 deg for at least 1 s. Duration uses sample intervals (21 samples for 1 s). Complete success additionally requires all 1200 steps and no failure/early stop.',
             '- KOZ experiment radius 0.5 m about origin; clearance is center-distance minus radius, not a swept-volume collision certificate.',
             '- Each solve saves full predicted states/actions in predictions.npz; replay.npz stores measured states and actually executed actions. Audits check every executed action against its accepted plan prefix and every plan initial state against the measurement.',
             '', '## Results', '',
             '| N/K | executed | success | final pos m | final speed m/s | final attitude deg | hover s | KOZ clearance m | fuel used kg | P50/P99 ms | >50ms % | >K*50ms % | solves |',
             '|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['horizon_steps']}/{r['execute_steps']} | {r['steps']}/1200 | {r['complete_success']} | "
                     f"{r['final_position_error_m']:.6g} | {r['final_relative_speed_m_s']:.6g} | "
                     f"{r['final_attitude_error_deg']:.5g} | {r['longest_hover_hold_s']:.2f} | "
                     f"{r['min_koz_clearance_m']:.5g} | {r['fuel_used_kg']:.5g} | "
                     f"{r['solver_hot_p50_ms']:.2f}/{r['solver_hot_p99_ms']:.2f} | "
                     f"{100*r['solver_deadline_miss_fraction']:.2f} | {100*r['solver_replan_deadline_miss_fraction']:.2f} | {r['solve_count']} |")
    lines += ['', '## Timing, failures, and limits', '',
              'P50/P99 exclude each run\'s first solve/build. Both the common 50 ms budget and the K×50 ms replanning budget are reported. This synchronous simulation does not inject solve delay into physics and cannot prove real-time performance. Host load may affect timing. One deterministic seed and initial condition are used; these are five configuration comparisons, not statistical robustness tests.', '',
              'The historical HCW-versus-SRB dynamics and nominal-versus-actual mass/inertia/COM/RCS mismatch is intentionally held fixed. Full physical matrices, dry mass, initial mass, fuel parameters and task configuration are recorded in each physics_audit.json; expert parameters are in controller_params.json. No mass matching or task retuning was performed.', '',
              'Move-block durations are preserved; extending N doubles the number of action blocks and changes total running-cost accumulation. Results compare these concrete controller configurations, not horizon length independent of every optimizer effect.', '',
              'Pre-run debugging: the original 20/1 replay completed 20 steps; the extended 20/1 produced identical states/actions for those 20 steps. All five configurations then passed 22-step SRB tests, including partial final prefixes. An audit initially rejected a ~1e-7 quaternion difference: PhysX float32 quaternions are normalized by AcadosExpert on entry. Normalizing the expected audit quaternion eliminated the difference exactly; no controller tolerance was loosened. Core/orbital/prefix tests passed (30 tests); after adding hover-duration edge cases, all 11 prefix/hover tests passed. Existing acados deprecations and the CasADi 3.8 compatibility warning remain.', '',
              'The generic HDF5 demonstration writer still uses its historical 8-block schema; this experiment uses NPZ recordings and does not claim 40-step/16-block HDF5 compatibility.', '']
    for r in rows:
        label = f"N{r['horizon_steps']}_K{r['execute_steps']}"
        lines += [f"- {label}: stop={r['stopped_reason']}; failure={json.dumps(r['failure'])}; max contact={r['max_contact_force_N']} N; first strict hover start={r['first_strict_hover_start_s']} s; x overshoot={r['positive_x_overshoot_m']:.5g} m; integrated position error={r['integrated_position_error_m_s']:.5g} m·s; hot solve cost/control step={r['solver_hot_ms_per_executed_step']:.3f} ms."]
        lines += [f"  Maximum hot solve={r['solver_hot_max_ms']:.3f} ms; replanning-budget misses={r['solver_replan_deadline_miss_count']}/{r['solve_count']-1}."]
    lines += ['', '## Artifacts', '',
              '- comparison.png / comparison.pdf: common axes for all five runs.',
              '- trajectories_3d.png: measured 3D paths (axis scales differ).',
              '- full/N*_K*/planned_and_executed.png and .pdf: separate plots of sampled full predicted horizons, predicted prefixes and measured trajectories. Every prediction is saved even though plots subsample to avoid clutter.',
              '- summary.json: all metrics; full/N*_K*/: raw data, controller and physics audit, invocation, solver records; full/N*_K*.log: complete Kit/solver logs.',
              '- debug/: short SRB runs before the formal experiments; debug_original_20_1/: original script regression reference.',
              '', '## Exact commands', '']
    for line in (root / 'commands.jsonl').read_text().splitlines():
        item = json.loads(line)
        lines += [f"Working directory: `{item['cwd']}`. Log: `{item['log']}`.", '',
                  '```sh', item['command'], '```', '']
    warning_patterns = ('GLFW initialization failed', 'GLInteropContext', 'not tested with acados', 'No device could',
                        'GPU Foundation is not', 'CUDA error', 'bad state', 'Incompatible device', 'Traceback')
    lines += ['## Selected warnings/errors from formal logs', '']
    for horizon, prefix in CASES:
        logfile = root / 'full' / f'N{horizon}_K{prefix}.log'
        selected = sorted(set(line.strip() for line in logfile.read_text(errors='replace').splitlines()
                              if any(p in line for p in warning_patterns)))
        lines += [f'### {logfile.name}', '', '```text', *selected, '```', '']
    (root / 'report.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('debug', 'full', 'report', 'all'), default='all')
    parser.add_argument('--resume', action='store_true', help='Re-audit and retain already completed cases')
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage in ('debug', 'all'):
        # Include a partial last chunk for K=3/5 and at least two feedback updates.
        for horizon, prefix in ((20, 1), (20, 5), (40, 1), (40, 3), (40, 5)):
            run_case(root, 'debug', horizon, prefix, 22, args.resume)
        original = root / 'debug_original_20_1' / 'replay.npz'
        if original.exists():
            old = np.load(original)
            new = np.load(root / 'debug/N20_K1/replay.npz')
            np.testing.assert_allclose(new['states'][:21], old['states'], atol=1e-6, rtol=0)
            np.testing.assert_allclose(new['duties'][:20], old['duties'], atol=1e-6, rtol=0)
            print('Original 20/1 vs extended 20/1 prefix matches within 1e-6', flush=True)
        (root / 'debug_passed.json').write_text(json.dumps({'cases': CASES, 'steps': 22,
            'checks': ['prediction shapes', 'feedback timestamps', 'executed prefix identity',
                       'no execution of rejected plans', 'CPU environment/PhysX', 'no early stops']}, indent=2))
    if args.stage in ('full', 'all'):
        if not (root / 'debug_passed.json').exists():
            raise RuntimeError('All five debug cases must pass first')
        for horizon, prefix in CASES:
            run_case(root, 'full', horizon, prefix, 1200, args.resume)
    if args.stage in ('report', 'full', 'all'):
        write_report(root, draw(root))
        print(f'REPORT {root / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
