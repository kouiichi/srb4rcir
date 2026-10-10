"""Reproducible randomized initial conditions for surface-approach trials."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SurfaceTrialSpec:
    """One target-relative encounter with a stationary identity-pose target."""

    trial_index: int
    target_position_m: tuple[float, float, float]
    initial_relative_root_position_m: tuple[float, float, float]
    initial_actor_world_root_position_m: tuple[float, float, float]
    initial_distance_m: float
    initial_linear_velocity_m_s: tuple[float, float, float] = (0., 0., 0.)
    initial_attitude_rpy_deg: tuple[float, float, float] = (0., 0., 0.)
    initial_angular_velocity_rad_s: tuple[float, float, float] = (0., 0., 0.)
    target_quaternion_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)
    goal_surface_gap_m: float = .30
    front_definition: str = 'target -X bus face; actor +Z top face'


def _validate_bounds(name: str, bounds: tuple[float, float]) -> tuple[float, float]:
    if len(bounds) != 2 or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(f'{name} must contain two finite ordered values')
    return float(bounds[0]), float(bounds[1])


def _sample_vector(rng: np.random.Generator, bounds: tuple[float, float]) -> tuple[float, float, float]:
    low, high = bounds
    return tuple(float(value) for value in rng.uniform(low, high, size=3))


def generate_trial_specs(
    seed: int,
    count: int = 5,
    *,
    target_position_bounds_m: tuple[float, float] = (-2., 2.),
    distance_bounds_m: tuple[float, float] = (14., 16.),
    lateral_offset_bounds_m: tuple[float, float] = (-2., 2.),
    linear_velocity_bounds_m_s: tuple[float, float] = (-.02, .02),
    attitude_rpy_bounds_deg: tuple[float, float] = (-5., 5.),
    angular_velocity_bounds_rad_s: tuple[float, float] = (-.005, .005),
    gap_values_m: tuple[float, ...] = (.5, .3, .1),
) -> list[SurfaceTrialSpec]:
    """Generate reproducible surface-approach initial conditions.

    Target translations are uniform in a 4 m cube. Relative actor distances
    and lateral offsets are sampled from the supplied bounds; the x component
    is solved so the requested distance is exact. The actor stays on the
    target -X side. Small initial velocity, attitude and angular-rate errors
    are randomized independently. Surface gaps are assigned in a repeating
    balanced sequence so all three requested values are represented when the
    dataset count is large enough.
    """
    if count <= 0:
        raise ValueError('count must be positive')
    target_bounds = _validate_bounds('target_position_bounds_m', target_position_bounds_m)
    distance_bounds = _validate_bounds('distance_bounds_m', distance_bounds_m)
    lateral_bounds = _validate_bounds('lateral_offset_bounds_m', lateral_offset_bounds_m)
    linear_bounds = _validate_bounds('linear_velocity_bounds_m_s', linear_velocity_bounds_m_s)
    attitude_bounds = _validate_bounds('attitude_rpy_bounds_deg', attitude_rpy_bounds_deg)
    angular_bounds = _validate_bounds('angular_velocity_bounds_rad_s', angular_velocity_bounds_rad_s)
    if not gap_values_m:
        raise ValueError('gap_values_m must contain at least one value')
    gaps = tuple(float(gap) for gap in gap_values_m)
    if not np.all(np.isfinite(gaps)) or any(gap <= 0. for gap in gaps):
        raise ValueError('gap_values_m must contain positive finite values')
    if distance_bounds[0] <= np.sqrt(2.) * max(abs(lateral_bounds[0]), abs(lateral_bounds[1])):
        raise ValueError('distance lower bound must exceed the maximum lateral offset')
    rng = np.random.default_rng(seed)
    trials = []
    for index in range(count):
        target = rng.uniform(*target_bounds, size=3)
        distance = float(rng.uniform(*distance_bounds))
        lateral = rng.uniform(*lateral_bounds, size=2)
        x_magnitude = float(np.sqrt(distance**2 - np.dot(lateral, lateral)))
        relative = np.array([-x_magnitude, lateral[0], lateral[1]])
        world = target + relative
        trials.append(SurfaceTrialSpec(
            trial_index=index,
            target_position_m=tuple(float(x) for x in target),
            initial_relative_root_position_m=tuple(float(x) for x in relative),
            initial_actor_world_root_position_m=tuple(float(x) for x in world),
            initial_distance_m=float(np.linalg.norm(relative)),
            initial_linear_velocity_m_s=_sample_vector(rng, linear_bounds),
            initial_attitude_rpy_deg=_sample_vector(rng, attitude_bounds),
            initial_angular_velocity_rad_s=_sample_vector(rng, angular_bounds),
            goal_surface_gap_m=gaps[index % len(gaps)],
        ))
    return trials


def trial_to_json(trial: SurfaceTrialSpec) -> dict:
    """Return a JSON-ready trial record used by experiment reports."""
    return asdict(trial)
