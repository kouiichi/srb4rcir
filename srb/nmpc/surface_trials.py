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
    target_quaternion_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)
    goal_surface_gap_m: float = .30
    front_definition: str = 'target -X bus face; actor +Z top face'


def generate_trial_specs(seed: int, count: int = 5) -> list[SurfaceTrialSpec]:
    """Generate target translations and -X-side actor starts around 15 m.

    Target translations are uniform in a 4 m cube. Relative actor distances
    are uniform in [14, 16] m, with independent lateral offsets in [-2, 2] m;
    the x component is solved so the requested distance is exact. Keeping the
    actor on the target -X side makes this a randomized approach experiment,
    rather than mixing in a second choice of docking side.
    """
    if count <= 0:
        raise ValueError('count must be positive')
    rng = np.random.default_rng(seed)
    trials = []
    for index in range(count):
        target = rng.uniform(-2., 2., size=3)
        distance = float(rng.uniform(14., 16.))
        lateral = rng.uniform(-2., 2., size=2)
        x_magnitude = float(np.sqrt(distance**2 - np.dot(lateral, lateral)))
        relative = np.array([-x_magnitude, lateral[0], lateral[1]])
        world = target + relative
        trials.append(SurfaceTrialSpec(
            trial_index=index,
            target_position_m=tuple(float(x) for x in target),
            initial_relative_root_position_m=tuple(float(x) for x in relative),
            initial_actor_world_root_position_m=tuple(float(x) for x in world),
            initial_distance_m=float(np.linalg.norm(relative)),
        ))
    return trials


def trial_to_json(trial: SurfaceTrialSpec) -> dict:
    """Return a JSON-ready trial record used by experiment reports."""
    return asdict(trial)
