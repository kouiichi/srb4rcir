"""Explicit contrast adapter for the current SRB 8-thruster Cubesat.

The main expert uses the independent 16-channel full-rank RCS contract.  SRB's
current Cubesat has eight one-sided thrusters and the exact ``ThrustAction``
implementation produces a rank-5 wrench map, so this module never presents an
8-channel projection as a valid 6-D expert label.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from .rcs import build_rcs_matrix, validate_rcs_geometry
from .types import FloatArray, TaskParams


SRB8_OFFSETS_B_M: FloatArray = np.array(
    [
        (-0.05, -0.05, 0.05),
        (-0.05, 0.05, 0.05),
        (0.05, -0.05, 0.05),
        (0.05, 0.05, 0.05),
        (-0.05, -0.05, -0.05),
        (-0.05, 0.05, -0.05),
        (0.05, -0.05, -0.05),
        (0.05, 0.05, -0.05),
    ],
    dtype=np.float64,
)
SRB8_DIRECTIONS_B: FloatArray = np.array(
    [
        (-0.5, -0.5, 1.0),
        (-0.5, 0.5, 1.0),
        (0.5, -0.5, 1.0),
        (0.5, 0.5, 1.0),
        (-0.5, -0.5, -1.0),
        (-0.5, 0.5, -1.0),
        (0.5, -0.5, -1.0),
        (0.5, 0.5, -1.0),
    ],
    dtype=np.float64,
)
SRB8_DIRECTIONS_B /= np.linalg.norm(SRB8_DIRECTIONS_B, axis=1, keepdims=True)


def build_srb8_matrix(
    *,
    power_N: float = 10.0,
    scale: float = 1.0,
    com_B_m: npt.ArrayLike | None = None,
) -> FloatArray:
    """Build the current SRB Cubesat's actual 8-column wrench matrix.

    The column convention follows ``srb/core/action/term/mobile/thrust.py``:
    applied force is ``-scale * power_N * duty * direction`` and torque is
    ``(offset-COM) x force``.  The returned shape is ``(6,8)`` float64 with
    body-frame SI rows ``[F_B [N], tau_B [N m]]``.  This function is a
    contrast/adapter model only; its expected numerical rank is five.
    """

    if not np.isfinite(power_N) or power_N <= 0.0:
        raise ValueError("power_N must be finite and positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    com = np.zeros(3, dtype=np.float64) if com_B_m is None else np.asarray(com_B_m, dtype=np.float64)
    if com.shape != (3,) or not np.all(np.isfinite(com)):
        raise ValueError("com_B_m must be a finite shape-(3,) array")
    forces = -float(scale) * float(power_N) * SRB8_DIRECTIONS_B
    torques = np.cross(SRB8_OFFSETS_B_M - com[None, :], forces)
    return np.ascontiguousarray(np.vstack((forces.T, torques.T)))


def build_srb16_matrix(
    *,
    power_N: float = 0.1,
    scale: float = 1.0,
    com_B_m: npt.ArrayLike | None = None,
) -> FloatArray:
    """Build the full-rank SRB adapter matrix for the 16-thruster variant.

    ``ThrustAction`` applies ``-scale * power * direction``.  The SRB 16-RCS
    asset therefore stores the opposite of each contractual force direction
    in its ``ThrusterCfg.direction``.  After that sign conversion, the
    physical matrix is exactly the canonical contract matrix.  This function
    is intentionally separate from :func:`build_srb8_matrix` so the existing
    eight-thruster contrast remains auditable and unchanged.

    Returns:
        A shape ``(6,16)`` float64 body-frame wrench matrix in SI units.
    """

    if not np.isfinite(power_N) or power_N <= 0.0:
        raise ValueError("power_N must be finite and positive")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return build_rcs_matrix(
        tmax_N=float(power_N) * float(scale),
        com_B_m=com_B_m,
    )


def validate_srb8_geometry(matrix: npt.ArrayLike | None = None) -> dict[str, float | int | bool]:
    """Validate and report the intentionally rank-5 SRB contrast geometry."""

    B = build_srb8_matrix() if matrix is None else np.asarray(matrix, dtype=np.float64)
    if B.shape != (6, 8) or not np.all(np.isfinite(B)):
        raise ValueError("SRB contrast matrix must be a finite shape-(6,8) array")
    singular = np.linalg.svd(B, compute_uv=False)
    rank = int(np.linalg.matrix_rank(B, tol=1.0e-10))
    pure_tau_z = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    tau_z_residual = float(np.linalg.norm(B @ np.linalg.lstsq(B, pure_tau_z, rcond=None)[0] - pure_tau_z))
    return {
        "rank": rank,
        "shape_rows": 6,
        "shape_columns": 8,
        "rank5_expected": rank == 5,
        "pure_tau_z_reachable": bool(tau_z_residual < 1.0e-10),
        "pure_tau_z_residual": tau_z_residual,
        "smallest_nonzero_singular_value": float(singular[4]),
    }


def _as_duty16(value: npt.ArrayLike) -> FloatArray:
    duty = np.asarray(value, dtype=np.float64)
    if duty.shape != (16,) or not np.all(np.isfinite(duty)):
        raise ValueError("duty16 must be a finite float64 array with shape (16,)")
    if np.any(duty < 0.0) or np.any(duty > 1.0):
        raise ValueError("duty16 must lie in [0,1]")
    return np.ascontiguousarray(duty)


class SrbAdapter:
    """Project canonical 16-duty commands into the current SRB Cubesat.

    ``step`` accepts one shape-(16,) float64 canonical duty at the current
    timestamp.  It computes the requested canonical body wrench with the
    full-rank 16-channel matrix, solves a bounded least-squares projection to
    the actual SRB shape-(6,8) matrix, and optionally calls a one-environment
    SRB ``env.step`` with the resulting shape ``(1,8)`` float32 action.  The
    returned diagnostics always contain both wrenches, the projected action,
    residual, rank-5 marker, and ``label_valid=False``.  Thus a projection is
    usable for an explicitly labelled contrast experiment but is never a main
    6-D expert label.

    If ``env`` is ``None`` the method performs a deterministic dry run and
    returns ``observation=None``; this is useful for matrix/projection tests.
    If an environment is supplied it must represent one environment.  Its
    observation and optional privileged state are returned without exposing
    future states or solver diagnostics to the student.
    """

    def __init__(
        self,
        env: Any | None = None,
        *,
        task_params: TaskParams | None = None,
        srb_power_N: float = 10.0,
        srb_scale: float = 1.0,
        residual_tolerance: float = 1.0e-4,
        privileged_state_provider: Callable[[Any], Any] | None = None,
    ) -> None:
        if residual_tolerance <= 0.0 or not np.isfinite(residual_tolerance):
            raise ValueError("residual_tolerance must be finite and positive")
        self.env = env
        self.task_params = task_params or TaskParams()
        self.source_B = build_rcs_matrix(
            tmax_N=self.task_params.tmax_N,
            com_B_m=self.task_params.com_B_m,
        )
        self.srb_B = build_srb8_matrix(
            power_N=srb_power_N,
            scale=srb_scale,
        )
        self.residual_tolerance = float(residual_tolerance)
        self.privileged_state_provider = privileged_state_provider
        geometry = validate_srb8_geometry(self.srb_B)
        if not bool(geometry["rank5_expected"]):
            raise RuntimeError(f"current SRB 8-thruster geometry is not rank 5: {geometry}")
        self.geometry = geometry

    def _project(self, duty16: FloatArray) -> tuple[FloatArray, FloatArray, float]:
        requested = np.ascontiguousarray(self.source_B @ duty16)
        try:
            from scipy.optimize import lsq_linear

            result = lsq_linear(
                self.srb_B,
                requested,
                bounds=(0.0, 1.0),
                lsmr_tol="auto",
                max_iter=200,
            )
            projected = np.asarray(result.x, dtype=np.float64)
            if not result.success or not np.all(np.isfinite(projected)):
                raise RuntimeError(str(result.message))
        except Exception:
            projected = np.clip(np.linalg.pinv(self.srb_B) @ requested, 0.0, 1.0)
        projected = np.ascontiguousarray(projected)
        executed = np.ascontiguousarray(self.srb_B @ projected)
        residual = float(np.linalg.norm(executed - requested))
        return requested, projected, residual

    def _environment_action(self, duty8: FloatArray) -> Any:
        action = duty8[None, :].astype(np.float32, copy=False)
        if self.env is not None and hasattr(self.env, "device"):
            try:
                import torch

                return torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
            except (ImportError, RuntimeError, TypeError):
                pass
        return action

    def _privileged_state(self, info: Any) -> Any:
        if self.privileged_state_provider is not None:
            return self.privileged_state_provider(self.env)
        if isinstance(info, Mapping) and "privileged_state" in info:
            return info["privileged_state"]
        if self.env is not None:
            getter = getattr(self.env, "get_privileged_state", None)
            if callable(getter):
                return getter()
            if hasattr(self.env, "privileged_state"):
                return self.env.privileged_state
        return None

    def step(self, duty16: npt.ArrayLike) -> tuple[Any, Any, dict[str, Any]]:
        """Project and optionally execute one current-time canonical command."""

        canonical_duty = _as_duty16(duty16)
        requested, duty8, residual = self._project(canonical_duty)
        executed = np.ascontiguousarray(self.srb_B @ duty8)
        diagnostics: dict[str, Any] = {
            "requested_duty16": canonical_duty.copy(),
            "projected_duty8": duty8.copy(),
            "requested_wrench6_body": requested.copy(),
            "executed_wrench6_body": executed.copy(),
            "projection_residual": residual,
            "residual_tolerance": self.residual_tolerance,
            "rank": int(self.geometry["rank"]),
            "label_valid": False,
            "sample_type": "contrast",
            "projection_status": "exact_in_subspace" if residual < self.residual_tolerance else "residual_exceeded",
            "executed_in_srb": False,
        }
        if self.env is None:
            return None, None, diagnostics
        if hasattr(self.env, "num_envs") and int(self.env.num_envs) != 1:
            raise ValueError("SrbAdapter.step requires an environment with num_envs=1")
        output = self.env.step(self._environment_action(duty8))
        observation = output[0] if isinstance(output, tuple) else output
        info = output[-1] if isinstance(output, tuple) and output else None
        privileged = self._privileged_state(info)
        diagnostics["executed_in_srb"] = True
        diagnostics["env_step_output_arity"] = len(output) if isinstance(output, tuple) else 1
        diagnostics["env_info"] = info
        return observation, privileged, diagnostics


class Srb16Adapter:
    """Execute canonical duties against the opt-in SRB 16-RCS variant.

    The supplied SRB environment must use ``assets.Cubesat16Rcs`` (or an
    equivalent 16-channel, no-gimbal ``ThrustAction`` configuration) and must
    have ``num_envs=1``.  The adapter compares the canonical matrix with the
    configured SRB matrix before every label is accepted.  A mismatch in
    force scale, center of mass, geometry, or action semantics is reported as
    ``label_valid=False`` and is never silently converted into a main expert
    label.  The old :class:`SrbAdapter` remains the explicit 8-channel,
    rank-5 contrast path.
    """

    def __init__(
        self,
        env: Any | None = None,
        *,
        task_params: TaskParams | None = None,
        srb_power_N: float | None = None,
        srb_scale: float = 1.0,
        residual_tolerance: float = 1.0e-4,
        privileged_state_provider: Callable[[Any], Any] | None = None,
    ) -> None:
        if residual_tolerance <= 0.0 or not np.isfinite(residual_tolerance):
            raise ValueError("residual_tolerance must be finite and positive")
        self.env = env
        self.task_params = task_params or TaskParams()
        self.srb_power_N = (
            self.task_params.tmax_N if srb_power_N is None else float(srb_power_N)
        )
        self.srb_scale = float(srb_scale)
        self.source_B = build_rcs_matrix(
            tmax_N=self.task_params.tmax_N,
            com_B_m=self.task_params.com_B_m,
        )
        self.srb_B = build_srb16_matrix(
            power_N=self.srb_power_N,
            scale=self.srb_scale,
            com_B_m=self.task_params.com_B_m,
        )
        self.residual_tolerance = float(residual_tolerance)
        self.privileged_state_provider = privileged_state_provider
        geometry = validate_rcs_geometry(self.srb_B, tmax_N=self.srb_power_N)
        if not bool(geometry["rank_ok"] and geometry["condition_ok"] and geometry["faults_ok"]):
            raise RuntimeError(f"SRB 16-RCS geometry failed validation: {geometry}")
        self.geometry = geometry

    def _environment_action(self, duty16: FloatArray) -> Any:
        action = duty16[None, :].astype(np.float32, copy=False)
        if self.env is not None and hasattr(self.env, "device"):
            try:
                import torch

                return torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
            except (ImportError, RuntimeError, TypeError):
                pass
        return action

    def _privileged_state(self, info: Any) -> Any:
        if self.privileged_state_provider is not None:
            return self.privileged_state_provider(self.env)
        if isinstance(info, Mapping) and "privileged_state" in info:
            return info["privileged_state"]
        if self.env is not None:
            getter = getattr(self.env, "get_privileged_state", None)
            if callable(getter):
                return getter()
            if hasattr(self.env, "privileged_state"):
                return self.env.privileged_state
        return None

    def step(self, duty16: npt.ArrayLike) -> tuple[Any, Any, dict[str, Any]]:
        """Execute one canonical 16-duty command and return audited diagnostics."""

        canonical_duty = _as_duty16(duty16)
        requested = np.ascontiguousarray(self.source_B @ canonical_duty)
        executed = np.ascontiguousarray(self.srb_B @ canonical_duty)
        residual = float(np.linalg.norm(executed - requested))
        label_valid = residual < self.residual_tolerance
        diagnostics: dict[str, Any] = {
            "requested_duty16": canonical_duty.copy(),
            "executed_duty16": canonical_duty.copy(),
            "requested_wrench6_body": requested.copy(),
            "executed_wrench6_body": executed.copy(),
            "projection_residual": residual,
            "residual_tolerance": self.residual_tolerance,
            "rank": int(self.geometry["rank"]),
            "label_valid": label_valid,
            "sample_type": "normal" if label_valid else "failed",
            "projection_status": "exact" if label_valid else "residual_exceeded",
            "executed_in_srb": False,
        }
        if self.env is None:
            return None, None, diagnostics
        if hasattr(self.env, "num_envs") and int(self.env.num_envs) != 1:
            raise ValueError("Srb16Adapter.step requires an environment with num_envs=1")
        output = self.env.step(self._environment_action(canonical_duty))
        observation = output[0] if isinstance(output, tuple) else output
        info = output[-1] if isinstance(output, tuple) and output else None
        privileged = self._privileged_state(info)
        diagnostics["executed_in_srb"] = True
        diagnostics["env_step_output_arity"] = len(output) if isinstance(output, tuple) else 1
        diagnostics["env_info"] = info
        return observation, privileged, diagnostics
