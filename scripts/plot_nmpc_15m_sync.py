#!/usr/bin/env python3
"""Plot telemetry from the 15 m synchronous NMPC surface-approach run.

The script uses only recorded data.  It does not re-run SRB or reconstruct
poses.  Acceleration and jerk are finite differences of the measured velocity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(folder: Path) -> dict[str, object]:
    replay = np.load(folder / "replay.npz")
    executor = np.load(folder / "executor.npz")
    surface = np.load(folder / "surface_metrics.npz")
    predictions = np.load(folder / "predictions.npz")
    references = np.load(folder / "solver_references.npz")
    replay_meta = json.loads((folder / "replay.json").read_text())
    audit = json.loads((folder / "physics_audit.json").read_text())
    return {
        "states": np.asarray(replay["states"], dtype=float),
        "refs": np.asarray(replay["refs"], dtype=float),
        "duties": np.asarray(replay["duties"], dtype=float),
        "solve_wall_s": np.asarray(replay["solve_wall_s"], dtype=float),
        "executor": {key: np.asarray(executor[key], dtype=float) for key in executor.files},
        "surface": {key: np.asarray(surface[key], dtype=float) for key in surface.files},
        "solve_steps": np.asarray(predictions["steps"], dtype=int),
        "accepted": np.asarray(predictions["accepted"], dtype=bool),
        "solver_refs": np.asarray(references["states"], dtype=float),
        "replay_meta": replay_meta,
        "audit": audit,
    }


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    """Return a centered finite difference with one-sided endpoints."""

    result = np.empty_like(values)
    result[0] = (values[1] - values[0]) / dt
    result[-1] = (values[-1] - values[-2]) / dt
    result[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    return result


def _phase_background(ax, duration: float, sync_duration: float) -> None:
    ax.axvspan(0.0, sync_duration, color="#4c78a8", alpha=0.055, lw=0)
    ax.axvspan(sync_duration, duration, color="#59a14f", alpha=0.075, lw=0)
    ax.axvline(sync_duration, color="#666666", ls="--", lw=0.9)


def _style_axes(axes, duration: float, sync_duration: float) -> None:
    for ax in np.asarray(axes, dtype=object).ravel():
        _phase_background(ax, duration, sync_duration)
        ax.set_xlim(0.0, duration)
        ax.grid(alpha=0.25, lw=0.7)
        ax.set_xlabel("Simulation time [s]")
        ax.tick_params(labelsize=8)


def _save(fig, output_dir: Path, stem: str) -> list[Path]:
    import matplotlib.pyplot as plt

    paths = [output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def _plot_telemetry(data: dict[str, object], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states = data["states"]
    refs = data["refs"]
    duties = data["duties"]
    executor = data["executor"]
    surface = data["surface"]
    solve_steps = data["solve_steps"]
    solve_wall_s = data["solve_wall_s"]
    solver_refs = data["solver_refs"]
    replay_meta = data["replay_meta"]
    audit = data["audit"]

    dt = float(audit["step_dt_s"])
    duration = float(len(duties) * dt)
    sync_duration = float(replay_meta.get("surface_reference_sync_s") or 120.0)
    time = np.arange(len(states), dtype=float) * dt
    action_time = np.arange(len(duties), dtype=float) * dt + 0.5 * dt
    solve_time = solve_steps * dt

    velocity = states[:, 3:6]
    acceleration = _finite_difference(velocity, dt)
    jerk = _finite_difference(acceleration, dt)
    speed = np.linalg.norm(velocity, axis=1)
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    angular_rate_deg = np.rad2deg(states[:, 10:13])
    angular_rate_norm_deg = np.linalg.norm(angular_rate_deg, axis=1)
    range_to_target = np.linalg.norm(states[:, :3], axis=1)
    position_error = np.linalg.norm(states[:, :3] - refs[0, :3], axis=1)

    actual_force = executor["force_B"].reshape(-1, 3)
    actual_torque = executor["torque_B"].reshape(-1, 3)
    force_norm = np.linalg.norm(actual_force, axis=1)
    torque_norm = np.linalg.norm(actual_torque, axis=1)
    fuel = executor["fuel_after"].reshape(-1)
    mass = executor["mass_after"].reshape(-1)
    previous_duty = np.vstack((np.zeros((1, duties.shape[1])), duties[:-1]))
    duty_rate = np.max(np.abs(duties - previous_duty), axis=1) / dt
    duty_max = np.max(duties, axis=1)
    duty_mean = np.mean(duties, axis=1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 14,
        }
    )
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    series = ["x", "y", "z"]

    # Kinematics: velocity, acceleration, attitude rate, and jerk.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    for i, label in enumerate(series):
        ax.plot(time, velocity[:, i], color=colors[i], lw=1.0, label=f"v{label}")
    ax.plot(time, speed, color="#222222", lw=1.5, label="speed norm")
    ax.set_ylabel("Velocity [m/s]")
    ax.set_title("Measured translational velocity")
    ax.legend(ncol=2, frameon=False)

    ax = axes[0, 1]
    for i, label in enumerate(series):
        ax.plot(time, acceleration[:, i], color=colors[i], lw=1.0, label=f"a{label}")
    ax.plot(time, acceleration_norm, color="#222222", lw=1.5, label="acceleration norm")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.set_title("Finite-difference acceleration from measured velocity")
    ax.legend(ncol=2, frameon=False)

    ax = axes[1, 0]
    for i, label in enumerate(series):
        ax.plot(time, angular_rate_deg[:, i], color=colors[i], lw=1.0, label=f"ω{label}")
    ax.plot(time, angular_rate_norm_deg, color="#222222", lw=1.5, label="rate norm")
    ax.set_ylabel("Angular rate [deg/s]")
    ax.set_title("Measured body angular rate")
    ax.legend(ncol=2, frameon=False)

    ax = axes[1, 1]
    ax.plot(time, jerk_norm, color="#9467bd", lw=1.1, label="jerk norm")
    ax.plot(time, np.abs(acceleration[:, 0]), color="#999999", lw=0.8, alpha=0.8, label="|ax|")
    ax.set_ylabel("Jerk [m/s³]")
    ax.set_title("Derived jerk and longitudinal acceleration")
    ax.legend(frameon=False)
    _style_axes(axes, duration, sync_duration)
    fig.suptitle("15 m synchronous approach: kinematics", x=0.5)
    kinematics_paths = _save(fig, output_dir, "nmpc_15m_sync_kinematics")

    # Approach quality and geometric safety.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time, range_to_target, color="#4c78a8", lw=1.3, label="COM range")
    ax.plot(time, position_error, color="#e45756", lw=1.1, label="position error to goal")
    ax.set_ylabel("Distance [m]")
    ax.set_title("Range and position error")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(time, states[:, 0], color=colors[0], lw=1.0, label="actual x")
    ax.plot(time, states[:, 1], color=colors[1], lw=1.0, label="actual y")
    ax.plot(time, states[:, 2], color=colors[2], lw=1.0, label="actual z")
    if solver_refs.shape[0] == solve_time.shape[0]:
        ax.plot(solve_time, solver_refs[:, 0, 0], "--", color=colors[0], lw=0.9, alpha=0.7, label="reference x")
        ax.plot(solve_time, solver_refs[:, 0, 1], "--", color=colors[1], lw=0.9, alpha=0.7, label="reference y")
        ax.plot(solve_time, solver_refs[:, 0, 2], "--", color=colors[2], lw=0.9, alpha=0.7, label="reference z")
    ax.set_ylabel("Relative COM position [m]")
    ax.set_title("Measured position and synchronous reference")
    ax.legend(ncol=2, frameon=False)

    ax = axes[0, 2]
    face_gap_cm = surface["face_gap_m"] * 100.0
    ax.plot(time, face_gap_cm, color="#54a24b", lw=1.3)
    ax.axhspan(28.0, 32.0, color="#54a24b", alpha=0.14, lw=0)
    ax.axhline(30.0, color="#333333", ls="--", lw=0.9)
    ax.set_ylabel("Face gap [cm]")
    ax.set_title("Target-facing gap (goal: 30 ± 2 cm)")

    ax = axes[1, 0]
    ax.plot(time, surface["lateral_error_m"] * 100.0, color="#f58518", lw=1.3)
    ax.axhline(2.0, color="#e45756", ls="--", lw=0.9, label="2 cm limit")
    ax.set_ylabel("Lateral error [cm]")
    ax.set_title("Alignment with target face")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(time, surface["facing_error_deg"], color="#e45756", lw=1.3)
    ax.axhline(5.0, color="#333333", ls="--", lw=0.9, label="5° limit")
    ax.set_ylabel("Facing error [deg]")
    ax.set_title("Attitude alignment")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    mesh_gap_cm = surface["mesh_plane_gap_m"] * 100.0
    ax.plot(time, mesh_gap_cm, color="#59a14f", lw=1.3)
    ax.axhline(2.0, color="#e45756", ls="--", lw=0.9, label="2 cm guard")
    ax.set_ylabel("Mesh-plane gap [cm]")
    ax.set_title("Full-mesh geometric separation")
    ax.legend(frameon=False)
    _style_axes(axes, duration, sync_duration)
    fig.suptitle("15 m synchronous approach: target tracking and safety", x=0.5)
    approach_paths = _save(fig, output_dir, "nmpc_15m_sync_approach")

    # Actuator, fuel, and computation telemetry.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    ax = axes[0, 0]
    force_time = action_time
    ax.plot(force_time, force_norm, color="#4c78a8", lw=1.2)
    ax.set_ylabel("Force norm [N]")
    ax.set_title("Applied body force")

    ax = axes[0, 1]
    ax.plot(force_time, torque_norm, color="#f58518", lw=1.2)
    ax.set_ylabel("Torque norm [N·m]")
    ax.set_title("Applied body torque")

    ax = axes[0, 2]
    ax.plot(action_time, fuel, color="#54a24b", lw=1.3, label="fuel")
    ax.set_ylabel("Fuel remaining [kg]")
    ax.set_title("Fuel and wet mass")
    ax2 = ax.twinx()
    ax2.plot(action_time, mass, color="#9467bd", lw=1.0, label="mass")
    ax2.set_ylabel("Wet mass [kg]")

    ax = axes[1, 0]
    ax.plot(action_time, duty_max, color="#e45756", lw=1.2, label="max duty")
    ax.plot(action_time, duty_mean, color="#4c78a8", lw=1.0, label="mean duty")
    ax.set_ylabel("Duty [0, 1]")
    ax.set_title("16-channel actuator demand")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(action_time, duty_rate, color="#e45756", lw=1.2)
    ax.axhline(float(replay_meta.get("duty_rate_limit_per_s", 20.0)), color="#333333", ls="--", lw=0.9, label="limit")
    ax.set_ylabel("Maximum duty rate [1/s]")
    ax.set_title("Actuator rate constraint")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    # The first solve includes acados code generation/build and is not part
    # of the repeated hot-loop timing reported by replay.json.  Keep it as a
    # visible annotation, while plotting the repeated solve calls on a useful
    # 0--200 ms scale.
    hot_slice = slice(1, None) if len(solve_wall_s) > 1 else slice(None)
    ax.plot(solve_time[hot_slice], solve_wall_s[hot_slice] * 1000.0, color="#9467bd", lw=1.1)
    if len(solve_wall_s) > 1:
        cold_ms = float(solve_wall_s[0] * 1000.0)
        ax.annotate(
            f"cold build: {cold_ms / 1000.0:.2f} s",
            xy=(solve_time[0], min(190.0, cold_ms)),
            xytext=(8.0, 175.0),
            textcoords="data",
            fontsize=8,
            color="#555555",
            arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 0.8},
        )
    ax.set_ylim(0.0, 200.0)
    ax.axhline(50.0, color="#999999", ls="--", lw=0.9, label="one-step period")
    ax.axhline(150.0, color="#e45756", ls="--", lw=0.9, label="K=3 replan budget")
    ax.set_ylabel("Solve wall time [ms]")
    ax.set_title("NMPC solve time")
    ax.legend(frameon=False)
    _style_axes(axes, duration, sync_duration)
    fig.suptitle("15 m synchronous approach: actuator, fuel, and computation", x=0.5)
    resource_paths = _save(fig, output_dir, "nmpc_15m_sync_resources")

    # A compact summary figure for slides.
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    hot_slice = slice(1, None) if len(solve_wall_s) > 1 else slice(None)
    summary = [
        (axes[0, 0], time, speed, "Speed norm [m/s]", "Speed"),
        (axes[0, 1], time, acceleration_norm, "Acceleration norm [m/s²]", "Acceleration"),
        (axes[0, 2], time, angular_rate_norm_deg, "Angular rate norm [deg/s]", "Angular rate"),
        (axes[1, 0], time, range_to_target, "COM range [m]", "Distance to target"),
        (axes[1, 1], time, face_gap_cm, "Face gap [cm]", "Target face gap"),
        (axes[1, 2], time, surface["facing_error_deg"], "Facing error [deg]", "Attitude alignment"),
        (axes[2, 0], time, mesh_gap_cm, "Mesh-plane gap [cm]", "Geometric separation"),
        (axes[2, 1], action_time, fuel, "Fuel remaining [kg]", "Fuel"),
        (
            axes[2, 2],
            solve_time[hot_slice],
            solve_wall_s[hot_slice] * 1000.0,
            "Hot solve time [ms]",
            "NMPC computation (hot loop)",
        ),
    ]
    for ax, x_values, y_values, ylabel, title in summary:
        ax.plot(x_values, y_values, color="#4c78a8", lw=1.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0, 1].axhline(float(replay_meta.get("max_acceleration_m_s2", np.nan)), color="#999999", ls=":", lw=0.8)
    axes[0, 2].axhline(0.05 * 180.0 / np.pi, color="#e45756", ls="--", lw=0.8)
    axes[1, 1].axhspan(28.0, 32.0, color="#54a24b", alpha=0.14, lw=0)
    axes[1, 2].axhline(5.0, color="#e45756", ls="--", lw=0.8)
    axes[2, 0].axhline(2.0, color="#e45756", ls="--", lw=0.8)
    axes[2, 2].axhline(50.0, color="#999999", ls="--", lw=0.8)
    axes[2, 2].axhline(150.0, color="#e45756", ls="--", lw=0.8)
    axes[2, 2].set_ylim(0.0, 200.0)
    _style_axes(axes, duration, sync_duration)
    fig.suptitle("15 m synchronous NMPC approach: recorded telemetry summary", x=0.5)
    summary_paths = _save(fig, output_dir, "nmpc_15m_sync_summary")
    return kinematics_paths + approach_paths + resource_paths + summary_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        nargs="?",
        default=Path("logs/nmpc_15m_sync_20260922/full"),
        help="formal experiment output folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for generated figures (default: folder)",
    )
    args = parser.parse_args()
    folder = args.folder.resolve()
    output_dir = (args.output_dir or folder).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _plot_telemetry(_load(folder), output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
