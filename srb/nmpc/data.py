"""Auditable HDF5 demonstration and Parquet diagnostic schemas."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .types import AllocationResult, ExpertResult, FloatArray


SAMPLE_TYPES = ("normal", "boundary", "ambiguous", "recovery", "dagger", "failed")


def _finite_array(value: npt.ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite float64 with shape {shape}")
    return np.ascontiguousarray(result)


def _episode_arrays(
    results: Sequence[ExpertResult],
) -> dict[str, np.ndarray]:
    if not results:
        raise ValueError("results must contain at least one ExpertResult")
    try:
        predicted_state = np.stack([result.predicted_state for result in results])
        predicted_duty = np.stack([result.predicted_duty for result in results])
        predicted_wrench = np.stack([result.predicted_wrench for result in results])
        actual_duty = np.stack([result.actual_duty for result in results])
        actual_wrench = np.stack([result.actual_wrench for result in results])
    except ValueError as exc:
        raise ValueError("all ExpertResult trajectory shapes must agree") from exc
    if predicted_state.ndim != 3 or predicted_state.shape[2] != 13:
        raise ValueError("predicted states must stack to (T,N+1,13)")
    if predicted_duty.shape != (len(results), predicted_state.shape[1] - 1, 16):
        raise ValueError("predicted duties must stack to (T,N,16)")
    if predicted_wrench.shape != (len(results), predicted_state.shape[1] - 1, 6):
        raise ValueError("predicted wrenches must stack to (T,N,6)")
    action_blocks = np.full((len(results), 8, 16), np.nan, dtype=np.float64)
    action_blocks_valid = np.zeros(len(results), dtype=np.bool_)
    for index, result in enumerate(results):
        if result.action_blocks is None:
            continue
        blocks = np.asarray(result.action_blocks, dtype=np.float64)
        if blocks.shape != (8, 16) or not np.all(np.isfinite(blocks)):
            raise ValueError("ExpertResult.action_blocks must have shape (8,16) and be finite")
        action_blocks[index] = blocks
        action_blocks_valid[index] = True
    return {
        "predicted_state": np.ascontiguousarray(predicted_state),
        "predicted_duty": np.ascontiguousarray(predicted_duty),
        "predicted_wrench": np.ascontiguousarray(predicted_wrench),
        "actual_duty": _finite_array(actual_duty, (len(results), 16), "actual_duty"),
        "actual_wrench": _finite_array(actual_wrench, (len(results), 6), "actual_wrench"),
        "action_blocks": action_blocks,
        "action_blocks_valid": action_blocks_valid,
    }


def _sample_types(results: Sequence[ExpertResult]) -> np.ndarray:
    values = [str(result.sample_type) for result in results]
    invalid = sorted(set(values).difference(SAMPLE_TYPES))
    if invalid:
        raise ValueError(f"unsupported sample types: {invalid}")
    return np.asarray(values, dtype=object)


def write_episode_hdf5(
    path: str | Path,
    *,
    episode_id: str,
    rgb: npt.ArrayLike | None,
    observed_state: npt.ArrayLike,
    timestamps_s: npt.ArrayLike,
    results: Sequence[ExpertResult],
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write one episode's aligned visual observations and expert outputs.

    ``rgb`` is optional for state-only oracle collection, but when present it
    must be uint8 ``(T,H,W,3)`` and is aligned one-for-one with
    ``observed_state``/``results`` at the global 20 Hz timestamps.  The state
    dataset is float64 ``(T,13)`` in ``[r_L,v_L,q_LB,omega_BI_B]``; prediction
    arrays are privileged labels with shapes ``(T,N+1,13)``, ``(T,N,16)``, and
    ``(T,N,6)``.  Solver diagnostics are intentionally written to Parquet,
    not exposed as student inputs.  Failed/recovery rows remain auditable and
    are never silently converted to normal labels.
    """

    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("episode_id must be a non-empty string")
    states = np.asarray(observed_state, dtype=np.float64)
    times = np.asarray(timestamps_s, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 13 or not np.all(np.isfinite(states)):
        raise ValueError("observed_state must be finite float64 with shape (T,13)")
    if times.shape != (states.shape[0],) or not np.all(np.isfinite(times)):
        raise ValueError("timestamps_s must be finite float64 with shape (T,)")
    if np.any(np.diff(times) < -1.0e-12):
        raise ValueError("timestamps_s must be non-decreasing")
    if len(results) != states.shape[0]:
        raise ValueError("observed_state, timestamps_s, and results must have equal length")
    rgb_array: np.ndarray | None = None
    if rgb is not None:
        rgb_array = np.asarray(rgb)
        if rgb_array.ndim != 4 or rgb_array.shape[0] != states.shape[0] or rgb_array.shape[-1] != 3:
            raise ValueError("rgb must have shape (T,H,W,3) aligned with the episode")
        if rgb_array.dtype != np.uint8:
            raise ValueError("rgb must use uint8 pixels")
        rgb_array = np.ascontiguousarray(rgb_array)
    arrays = _episode_arrays(results)
    sample_types = _sample_types(results)

    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("h5py is required for HDF5 demonstration output") from exc

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.attrs["schema_version"] = "srb-nmpc-v1"
        handle.attrs["episode_id"] = episode_id
        handle.attrs["control_frequency_hz"] = 20.0
        handle.attrs["control_period_s"] = 0.05
        handle.attrs["state_contract"] = "State13=[r_L,v_L,q_LB,omega_BI_B]"
        handle.attrs["student_input_excludes"] = "future_states,future_actions,solver_diagnostics,privileged_reference"
        if metadata:
            for key, value in metadata.items():
                handle.attrs[str(key)] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
        handle.create_dataset("timestamps_s", data=times, dtype="f8")
        handle.create_dataset("observed_state13", data=states, dtype="f8")
        if rgb_array is not None:
            handle.create_dataset(
                "rgb",
                data=rgb_array,
                chunks=(1, *rgb_array.shape[1:]),
                compression="gzip",
                compression_opts=4,
            )
            handle.attrs["rgb_present"] = True
        else:
            handle.attrs["rgb_present"] = False
        for name, value in arrays.items():
            handle.create_dataset(name, data=value, dtype=value.dtype)
        handle.create_dataset(
            "sample_type",
            data=sample_types.astype(h5py.string_dtype(encoding="utf-8")),
        )
    return output


def diagnostic_rows(
    episode_id: str,
    timestamps_s: npt.ArrayLike,
    results: Sequence[ExpertResult],
    filters: Sequence[AllocationResult | None] | None = None,
) -> list[dict[str, Any]]:
    """Convert per-step solver/filter diagnostics to Parquet-ready rows."""

    times = np.asarray(timestamps_s, dtype=np.float64)
    if times.shape != (len(results),):
        raise ValueError("timestamps_s and results must have equal length")
    if filters is not None and len(filters) != len(results):
        raise ValueError("filters and results must have equal length")
    rows: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        allocation = None if filters is None else filters[index]
        row: dict[str, Any] = {
            "episode_id": episode_id,
            "step": index,
            "timestamp_s": float(times[index]),
            "status": result.status,
            "solver_status_code": int(result.solver_status_code),
            "quality_verdict": result.quality_verdict,
            "sample_type": result.sample_type,
            "iterations": int(result.iterations),
            "kkt": float(result.kkt),
            "runtime_s": float(result.runtime_s),
            "objective": float(result.objective),
            "allocation_residual": float(result.allocation_residual),
            "failure_reason": result.failure_reason or "",
            "constraint_margins_json": json.dumps(result.constraint_margins, sort_keys=True),
            "slacks_json": json.dumps(result.slacks, sort_keys=True),
            "filter_status": "",
            "filter_runtime_s": np.nan,
            "filter_residual": np.nan,
            "filter_intervention_norm": np.nan,
            "filter_margins_json": "",
        }
        if allocation is not None:
            row.update(
                {
                    "filter_status": allocation.status,
                    "filter_runtime_s": float(allocation.runtime_s),
                    "filter_residual": float(allocation.residual),
                    "filter_intervention_norm": float(np.linalg.norm(allocation.intervention)),
                    "filter_margins_json": json.dumps(allocation.constraint_margins, sort_keys=True),
                }
            )
        rows.append(row)
    return rows


def write_parquet_index(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write diagnostic rows as Parquet, requiring an explicit engine."""

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("pandas is required for Parquet index output") from exc
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow; run the SRB wrapper with "
            "requirements-nmpc-isolated.txt to install the isolated support"
        ) from exc
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(output, index=False, engine="pyarrow")
    return output
