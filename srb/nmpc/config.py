"""Validated YAML loading for the standalone 20 Hz NMPC contract.

The SRB environment uses Isaac-Lab configuration classes, while the NMPC
expert is intentionally a plain NumPy/CasADi component.  This module is the
boundary between the repository YAML files and that standalone component.  A
successful load produces a validated :class:`TaskParams` and a separate
solver configuration; no solver or script is allowed to silently fall back to
``TaskParams()`` after a YAML path has been supplied.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .types import TaskParams


class NmpcConfigError(ValueError):
    """Raised when an NMPC YAML file violates the fixed project contract."""


@dataclass(frozen=True)
class SolverConfig:
    """Numerical solver settings loaded from the NMPC YAML file."""

    oracle: str = "ipopt"
    production: str = "acados"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    nlp_solver_type: str = "SQP_RTI"
    oracle_max_iterations: int = 80
    production_max_iterations: int = 1
    tolerance: float = 1.0e-6
    qp_solver_cond_N: int = 5
    qp_solver_iter_max: int = 100
    code_export_root: Path | None = None


@dataclass(frozen=True)
class NmpcConfig:
    """Complete validated NMPC configuration and its source path."""

    task_params: TaskParams
    solver: SolverConfig
    source_path: Path

    def resolved_code_export_root(self, base_dir: Path | None = None) -> Path | None:
        """Return the configured acados output directory as an absolute path."""

        if self.solver.code_export_root is None:
            return None
        root = self.solver.code_export_root
        if root.is_absolute():
            return root
        return (base_dir or Path.cwd()) / root


_TASK_FIELDS = {field.name for field in fields(TaskParams) if field.init}
_ROOT_KEYS = {
    "control_frequency_hz",
    "control_period_s",
    "horizon_steps",
    "horizon_seconds",
    "rk4_substeps",
    "rk4_substep_dt_s",
    "action_block_lengths",
    "solver",
    "actuator",
    "model",
    "task_params",
    "timing",
    "reference",
    "constraints",
    "limits",
    "costs",
    "camera",
}
_SOLVER_KEYS = {
    "oracle",
    "production",
    "qp_solver",
    "nlp_solver_type",
    "oracle_max_iterations",
    "production_max_iterations",
    "tolerance",
    "qp_solver_cond_N",
    "qp_solver_iter_max",
    "code_export_root",
}
_CONTRACT_MODEL_KEYS = {
    "type",
    "state_dim",
    "quaternion_order",
    "quaternion_frame",
    "angular_rate_frame",
    "target_angular_velocity_episode_constant",
}
_CONTRACT_ACTUATOR_KEYS = {
    "channels",
    "rho_m",
    "delta_m",
    "duty_min",
    "duty_max",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NmpcConfigError(f"{name} must be a mapping")
    return value


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise NmpcConfigError(f"unknown {name} key(s): {', '.join(unknown)}")


def _check_close(actual: Any, expected: float, name: str, atol: float = 1.0e-12) -> None:
    try:
        valid = bool(np.isclose(float(actual), expected, rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise NmpcConfigError(f"{name} must be {expected!r}, got {actual!r}")


def _check_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise NmpcConfigError(f"{name} must be {expected!r}, got {actual!r}")


def _task_values(document: Mapping[str, Any]) -> dict[str, Any]:
    """Extract TaskParams fields and validate non-TaskParams contract keys."""

    _reject_unknown(document, _ROOT_KEYS | _TASK_FIELDS, "root")
    values: dict[str, Any] = {
        key: document[key] for key in _TASK_FIELDS if key in document
    }

    # The current files keep timing fields at the root.  ``timing`` is also
    # accepted so future files can group them without changing the mapping
    # semantics.
    timing: dict[str, Any] = {
        key: document[key]
        for key in (
            "control_period_s",
            "horizon_steps",
            "rk4_substeps",
            "rk4_substep_dt_s",
            "action_block_lengths",
        )
        if key in document
    }
    if "timing" in document:
        timing_section = dict(_mapping(document["timing"], "timing"))
        _reject_unknown(
            timing_section,
            {
                "control_frequency_hz",
                "control_period_s",
                "horizon_steps",
                "horizon_seconds",
                "rk4_substeps",
                "rk4_substep_dt_s",
                "action_block_lengths",
            },
            "timing",
        )
        timing.update(timing_section)
    values.update({key: value for key, value in timing.items() if key in _TASK_FIELDS})

    for section_name in ("task_params", "reference", "constraints", "limits", "costs", "camera"):
        if section_name not in document:
            continue
        section = _mapping(document[section_name], section_name)
        _reject_unknown(section, _TASK_FIELDS, section_name)
        values.update(section)

    actuator = dict(_mapping(document.get("actuator", {}), "actuator"))
    _reject_unknown(
        actuator,
        _CONTRACT_ACTUATOR_KEYS | {"tmax_N", "duty_rate_per_s", "reverse_pair_limit"}
        | _TASK_FIELDS,
        "actuator",
    )
    if "channels" in actuator:
        _check_equal(actuator["channels"], 16, "actuator.channels")
    if "rho_m" in actuator:
        _check_close(actuator["rho_m"], 0.30, "actuator.rho_m")
    if "delta_m" in actuator:
        _check_close(actuator["delta_m"], 0.05, "actuator.delta_m")
    if "duty_min" in actuator:
        _check_close(actuator["duty_min"], 0.0, "actuator.duty_min")
    if "duty_max" in actuator:
        _check_close(actuator["duty_max"], 1.0, "actuator.duty_max")
    values.update({key: actuator[key] for key in _TASK_FIELDS if key in actuator})

    model = dict(_mapping(document.get("model", {}), "model"))
    _reject_unknown(model, _CONTRACT_MODEL_KEYS | _TASK_FIELDS, "model")
    if "type" in model:
        _check_equal(model["type"], "hcw_quaternion_rigid_body", "model.type")
    if "state_dim" in model:
        _check_equal(model["state_dim"], 13, "model.state_dim")
    if "quaternion_order" in model:
        _check_equal(model["quaternion_order"], "scalar_first", "model.quaternion_order")
    if "quaternion_frame" in model:
        _check_equal(model["quaternion_frame"], "q_LB", "model.quaternion_frame")
    if "angular_rate_frame" in model:
        _check_equal(model["angular_rate_frame"], "omega_BI_B", "model.angular_rate_frame")
    if "target_angular_velocity_episode_constant" in model:
        _check_equal(
            model["target_angular_velocity_episode_constant"],
            True,
            "model.target_angular_velocity_episode_constant",
        )
    values.update({key: model[key] for key in _TASK_FIELDS if key in model})

    # These fields are metadata in YAML rather than TaskParams constructor
    # arguments.  They are checked here so a typo or an inconsistent timing
    # value cannot be hidden by the dataclass defaults.
    if "control_frequency_hz" in document:
        _check_close(document["control_frequency_hz"], 20.0, "control_frequency_hz")
    if "timing" in document and "control_frequency_hz" in document["timing"]:
        _check_close(document["timing"]["control_frequency_hz"], 20.0, "timing.control_frequency_hz")
    if "horizon_seconds" in document:
        values_horizon = document["horizon_seconds"]
        values["_horizon_seconds"] = values_horizon
    if "timing" in document and "horizon_seconds" in document["timing"]:
        values["_horizon_seconds"] = document["timing"]["horizon_seconds"]
    if "_horizon_seconds" in values:
        horizon_seconds = values.pop("_horizon_seconds")
        # The expected value is checked after timing overrides have been
        # merged, before TaskParams performs its own fixed-grid validation.
        period = float(values.get("control_period_s", TaskParams().control_period_s))
        steps = int(values.get("horizon_steps", TaskParams().horizon_steps))
        _check_close(horizon_seconds, period * steps, "horizon_seconds")

    return values


def task_params_from_mapping(document: Mapping[str, Any]) -> TaskParams:
    """Create validated TaskParams from one loaded YAML mapping.

    The input may be the full NMPC document or a mapping containing only
    ``TaskParams`` fields.  Timing, actuator, model, shape, unit, and fixed
    contract metadata are validated; omitted physical/reference fields use
    the documented :class:`TaskParams` defaults.
    """

    document = _mapping(document, "NMPC configuration")
    values = _task_values(document)
    if "action_block_lengths" in values:
        values["action_block_lengths"] = tuple(values["action_block_lengths"])
    return TaskParams(**values)


def _solver_config(document: Mapping[str, Any]) -> SolverConfig:
    raw = _mapping(document.get("solver", {}), "solver")
    _reject_unknown(raw, _SOLVER_KEYS, "solver")
    values = dict(raw)
    if "oracle" in values:
        _check_equal(values["oracle"], "ipopt", "solver.oracle")
    if "production" in values:
        _check_equal(values["production"], "acados", "solver.production")
    if "qp_solver" in values:
        _check_equal(values["qp_solver"], "PARTIAL_CONDENSING_HPIPM", "solver.qp_solver")
    if "nlp_solver_type" in values:
        if values["nlp_solver_type"] not in {"SQP", "SQP_RTI"}:
            raise NmpcConfigError("solver.nlp_solver_type must be SQP or SQP_RTI")
    for key in ("oracle_max_iterations", "production_max_iterations", "qp_solver_cond_N", "qp_solver_iter_max"):
        if key in values and (not isinstance(values[key], int) or values[key] <= 0):
            raise NmpcConfigError(f"solver.{key} must be a positive integer")
    if "tolerance" in values:
        if not np.isfinite(float(values["tolerance"])) or float(values["tolerance"]) <= 0.0:
            raise NmpcConfigError("solver.tolerance must be finite and positive")
    if "code_export_root" in values:
        root = Path(values["code_export_root"])
        values["code_export_root"] = root
    return SolverConfig(**values)


def load_nmpc_config(path: str | Path) -> NmpcConfig:
    """Load and validate one NMPC YAML file.

    Args:
        path: YAML file path.  Relative paths are resolved from the current
            process directory before reading and recorded as absolute source
            paths.

    Returns:
        A validated :class:`NmpcConfig`; its ``task_params`` is the only
        parameter object passed to either expert backend.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        NmpcConfigError: for malformed YAML or a contract mismatch.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    try:
        with source_path.open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise NmpcConfigError(f"failed to parse {source_path}: {exc}") from exc
    document = _mapping(document, f"{source_path}")
    timing_document = (
        _mapping(document["timing"], "timing")
        if "timing" in document
        else {}
    )
    required_timing = {
        "control_frequency_hz",
        "control_period_s",
        "horizon_steps",
        "horizon_seconds",
        "rk4_substeps",
        "rk4_substep_dt_s",
        "action_block_lengths",
    }
    missing_timing = sorted(
        key
        for key in required_timing
        if key not in document and key not in timing_document
    )
    missing_sections = sorted(
        key for key in ("solver", "actuator", "model") if key not in document
    )
    if missing_timing or missing_sections:
        missing = [*(f"timing.{key}" for key in missing_timing), *missing_sections]
        raise NmpcConfigError(
            f"{source_path} is missing required NMPC configuration key(s): {', '.join(missing)}"
        )
    try:
        params = task_params_from_mapping(document)
        solver = _solver_config(document)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, NmpcConfigError):
            raise
        raise NmpcConfigError(f"invalid NMPC configuration {source_path}: {exc}") from exc
    return NmpcConfig(task_params=params, solver=solver, source_path=source_path)


def load_task_params(path: str | Path) -> TaskParams:
    """Load only the validated TaskParams portion of an NMPC YAML file."""

    return load_nmpc_config(path).task_params
