"""Budgeted, auditable selective DAgger query logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


def _quantile(values: npt.ArrayLike, probability: float, name: str) -> float:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0 or not np.all(np.isfinite(data)):
        raise ValueError(f"{name} must contain finite values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0,1]")
    return float(np.quantile(data, probability))


@dataclass(frozen=True)
class DaggerThresholds:
    """Thresholds calibrated from normal expert data percentiles."""

    safety_margin_floor: float
    fov_margin_floor: float
    ensemble_uncertainty_ceiling: float
    novelty_ceiling: float
    filter_correction_ceiling: float

    @classmethod
    def from_expert_data(
        cls,
        *,
        safety_margins: npt.ArrayLike,
        fov_margins: npt.ArrayLike | None = None,
        ensemble_uncertainty: npt.ArrayLike | None = None,
        novelty: npt.ArrayLike | None = None,
        filter_correction: npt.ArrayLike | None = None,
        lower_quantile: float = 0.05,
        upper_quantile: float = 0.99,
    ) -> "DaggerThresholds":
        """Calibrate floors/ceilings at expert-data 5/95 percentiles.

        Lower safety floors use the lower 5th percentile (the 95% expert
        interior boundary); uncertainty/novelty/correction ceilings use the
        upper 99th percentile.  The chosen probabilities must be recorded in
        the experiment manifest.
        """

        safety = _quantile(safety_margins, lower_quantile, "safety_margins")
        fov = safety if fov_margins is None else _quantile(fov_margins, lower_quantile, "fov_margins")
        uncertainty = np.inf if ensemble_uncertainty is None else _quantile(ensemble_uncertainty, upper_quantile, "ensemble_uncertainty")
        novelty_value = np.inf if novelty is None else _quantile(novelty, upper_quantile, "novelty")
        correction = np.inf if filter_correction is None else _quantile(filter_correction, upper_quantile, "filter_correction")
        return cls(safety, fov, uncertainty, novelty_value, correction)


@dataclass(frozen=True)
class DaggerDecision:
    """One auditable query decision at a current observation."""

    query: bool
    reasons: tuple[str, ...]
    visited_steps: int
    queried_steps: int
    query_fraction: float
    budget_exhausted: bool


class DaggerTrigger:
    """Trigger selective expert queries without exceeding a 10% step budget."""

    def __init__(
        self,
        thresholds: DaggerThresholds,
        *,
        max_query_fraction: float = 0.10,
    ) -> None:
        if not 0.0 < max_query_fraction <= 0.10:
            raise ValueError("max_query_fraction must be in (0, 0.10]")
        self.thresholds = thresholds
        self.max_query_fraction = float(max_query_fraction)
        self.visited_steps = 0
        self.queried_steps = 0

    def decide(
        self,
        *,
        safety_margin: float,
        fov_margin: float = np.inf,
        ensemble_uncertainty: float = 0.0,
        novelty: float = 0.0,
        filter_correction: float = 0.0,
    ) -> DaggerDecision:
        """Record one visited step and decide whether to query the oracle."""

        values = (safety_margin, ensemble_uncertainty, novelty, filter_correction)
        if not np.all(np.isfinite(values)) or np.isnan(fov_margin) or fov_margin == -np.inf:
            raise ValueError("DAgger trigger metrics must be finite; absent FOV may use +inf")
        self.visited_steps += 1
        reasons: list[str] = []
        if safety_margin <= self.thresholds.safety_margin_floor:
            reasons.append("safety_margin")
        if fov_margin <= self.thresholds.fov_margin_floor:
            reasons.append("fov_margin")
        if ensemble_uncertainty >= self.thresholds.ensemble_uncertainty_ceiling:
            reasons.append("ensemble_uncertainty")
        if novelty >= self.thresholds.novelty_ceiling:
            reasons.append("novelty")
        if filter_correction >= self.thresholds.filter_correction_ceiling:
            reasons.append("filter_correction")
        projected_fraction = (self.queried_steps + 1) / self.visited_steps
        budget_exhausted = projected_fraction > self.max_query_fraction
        query = bool(reasons) and not budget_exhausted
        if query:
            self.queried_steps += 1
        fraction = self.queried_steps / self.visited_steps
        return DaggerDecision(
            query=query,
            reasons=tuple(reasons),
            visited_steps=self.visited_steps,
            queried_steps=self.queried_steps,
            query_fraction=fraction,
            budget_exhausted=budget_exhausted,
        )


def select_visual_action_candidate(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select one candidate by cost/Q value and preserve all alternatives.

    ``candidates`` must contain finite ``cost`` values and an ``action``.  The
    function never averages conflicting wrenches from visually equivalent
    states; it returns the selected candidate plus an audit record containing
    every candidate, their costs, and the conflict flag.
    """

    if not candidates:
        raise ValueError("candidates must be non-empty")
    costs = np.asarray([candidate.get("cost", np.nan) for candidate in candidates], dtype=np.float64)
    if not np.all(np.isfinite(costs)):
        raise ValueError("candidate costs must be finite")
    selected_index = int(np.argmin(costs))
    actions = [np.asarray(candidate["action"], dtype=np.float64) for candidate in candidates]
    if any(not np.all(np.isfinite(action)) for action in actions):
        raise ValueError("candidate actions must be finite")
    conflict = any(not np.allclose(actions[0], action) for action in actions[1:])
    audit = {
        "candidate_count": len(candidates),
        "candidate_costs": costs.tolist(),
        "selected_index": selected_index,
        "action_conflict": conflict,
        "candidates": candidates,
        "resolution": "minimum_cost_no_averaging",
    }
    return candidates[selected_index], audit
