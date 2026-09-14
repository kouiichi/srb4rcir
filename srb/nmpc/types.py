"""Public data contracts for the SRB NMPC expert.

The simulator and the expert deliberately use different action interfaces.  The
expert optimizes 16 non-negative thruster duties and exposes the corresponding
body wrench only as a derived quantity.  This keeps the training label
executable and makes actuator residuals auditable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

import numpy as np
import numpy.typing as npt


FloatArray: TypeAlias = npt.NDArray[np.float64]

# Runtime values are contiguous float64 NumPy arrays.  The shape and frame
# contracts are documented here because NumPy's type system cannot express the
# fixed dimensions portably on Python 3.10.
State13: TypeAlias = npt.NDArray[np.float64]
"""Shape ``(13,)`` float64 SI state in ``L/B`` coordinates.

The entries are ``[r_L(3) [m], v_L(3) [m/s], q_LB(4) [-],
omega_BI_B(3) [rad/s]]``.  The quaternion is scalar-first and maps body
vectors to LVLH.  A state timestamp is the sample time at which it is
measured; functions reject non-finite values rather than silently repairing
them.
"""

Wrench6Body: TypeAlias = npt.NDArray[np.float64]
"""Shape ``(6,)`` float64 body wrench ``[F_B(3) [N], tau_B(3) [N m]]``."""

RcsDuty16: TypeAlias = npt.NDArray[np.float64]
"""Shape ``(16,)`` float64 one-sided RCS duties in ``[0, 1]``."""


def _array(value: npt.ArrayLike, shape: tuple[int, ...], name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(result)


@dataclass
class TaskParams:
    """Physical, reference, and hard-limit parameters for one episode.

    All lengths, masses, forces, and inertias use SI units.  ``r_L`` and
    ``v_L`` are target-relative LVLH quantities, ``q_LT0`` maps target-body
    vectors to LVLH at the episode start, and target angular velocity is
    expressed in target/body axes.  The target angular velocity is constant
    within an episode by design; the default and pressure ranges are
    experimental assumptions, not measured vehicle specifications.

    The global SRB timing contract is 20 Hz: one control period is 0.05 s,
    the horizon has 20 intervals (1 s), and each interval is integrated with
    five 0.01 s RK4 substeps.  ``action_block_lengths`` is the fixed K=8
    rolling action-block interface; the optimizer still solves every interval.
    """

    control_period_s: float = 0.05
    horizon_steps: int = 20
    rk4_substeps: int = 5
    rk4_substep_dt_s: float = 0.01
    action_block_lengths: tuple[int, ...] = (3, 3, 3, 3, 2, 2, 2, 2)

    mean_motion_rad_s: float = 0.001027
    mass_kg: float = 12.0
    inertia_B_kg_m2: FloatArray = field(
        default_factory=lambda: np.diag([0.20, 0.24, 0.30]).astype(np.float64)
    )
    tmax_N: float = 0.1
    com_B_m: FloatArray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rcs_matrix_B: FloatArray | None = None

    target_point_T_m: FloatArray = field(
        default_factory=lambda: np.array([-1.5, 0.0, 0.0], dtype=np.float64)
    )
    camera_target_point_T_m: FloatArray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    target_q_LT0: FloatArray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    target_omega_TI_T_rad_s: FloatArray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    koz_center_L_m: FloatArray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    koz_radius_m: float = 0.5
    koz_k0_s2: float = 4.0
    koz_k1_s: float = 4.0
    koz_safety_margin_m2_s2: float = 0.0

    velocity_limit_m_s: float = 2.0
    angular_velocity_limit_rad_s: float = 0.5
    attitude_limit_rad: float = np.pi
    duty_rate_per_s: float = 20.0
    reverse_pair_limit: float = 1.0

    enable_fov: bool = False
    camera_axis_B: FloatArray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    fov_half_angle_rad: float = np.deg2rad(45.0)
    min_camera_depth_m: float = 0.05
    max_camera_depth_m: float = 10.0

    position_cost: float = 10.0
    velocity_cost: float = 1.0
    attitude_cost: float = 2.0
    angular_velocity_cost: float = 0.5
    duty_cost: float = 1.0e-3
    duty_rate_cost: float = 1.0e-2
    terminal_cost_scale: float = 10.0

    @classmethod
    def from_mapping(cls, mapping: "Mapping[str, object]") -> "TaskParams":
        """Construct TaskParams through the validated NMPC YAML boundary.

        The mapping may be a complete NMPC configuration or a mapping of
        TaskParams fields.  Keeping this convenience method on the public
        contract makes it difficult for a caller to accidentally parse YAML
        and then continue with an unrelated default ``TaskParams()``.
        """

        from .config import task_params_from_mapping

        params = task_params_from_mapping(mapping)
        if cls is TaskParams:
            return params
        return cls(**params.__dict__)

    @classmethod
    def from_yaml(cls, path: "str | Path") -> "TaskParams":
        """Load a validated TaskParams instance from an NMPC YAML file."""

        from .config import load_task_params

        params = load_task_params(path)
        if cls is TaskParams:
            return params
        return cls(**params.__dict__)

    def __post_init__(self) -> None:
        """Validate and normalize fixed-shape parameters at construction.

        Raises:
            ValueError: if a dimension, unit limit, timing value, quaternion,
                inertia, or bound is invalid.  No value is clipped silently.
        """

        scalar_positive = (
            ("control_period_s", self.control_period_s),
            ("mass_kg", self.mass_kg),
            ("tmax_N", self.tmax_N),
            ("koz_radius_m", self.koz_radius_m),
            ("koz_k0_s2", self.koz_k0_s2),
            ("velocity_limit_m_s", self.velocity_limit_m_s),
            ("angular_velocity_limit_rad_s", self.angular_velocity_limit_rad_s),
            ("duty_rate_per_s", self.duty_rate_per_s),
        )
        for name, value in scalar_positive:
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        if not isinstance(self.horizon_steps, int) or self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        if not isinstance(self.rk4_substeps, int) or self.rk4_substeps <= 0:
            raise ValueError("rk4_substeps must be a positive integer")
        if not np.isclose(
            self.control_period_s,
            self.rk4_substeps * self.rk4_substep_dt_s,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("control_period_s must equal rk4_substeps*rk4_substep_dt_s")
        if not np.isclose(self.control_period_s, 0.05, rtol=0.0, atol=1.0e-12):
            raise ValueError("the SRB NMPC timing contract requires Ts=0.05 s (20 Hz)")
        if self.action_block_lengths != (3, 3, 3, 3, 2, 2, 2, 2):
            raise ValueError("action_block_lengths is fixed to (3,3,3,3,2,2,2,2)")
        if sum(self.action_block_lengths) != self.horizon_steps:
            raise ValueError("action block lengths must cover the prediction horizon")

        self.inertia_B_kg_m2 = _array(self.inertia_B_kg_m2, (3, 3), "inertia_B_kg_m2")
        self.com_B_m = _array(self.com_B_m, (3,), "com_B_m")
        if self.rcs_matrix_B is not None:
            self.rcs_matrix_B = _array(self.rcs_matrix_B, (6, 16), "rcs_matrix_B")
        if not np.isfinite(self.mean_motion_rad_s) or self.mean_motion_rad_s < 0:
            raise ValueError("mean_motion_rad_s must be finite and >= 0")
        self.target_point_T_m = _array(self.target_point_T_m, (3,), "target_point_T_m")
        self.camera_target_point_T_m = _array(
            self.camera_target_point_T_m, (3,), "camera_target_point_T_m"
        )
        self.target_q_LT0 = _array(self.target_q_LT0, (4,), "target_q_LT0")
        self.target_omega_TI_T_rad_s = _array(
            self.target_omega_TI_T_rad_s, (3,), "target_omega_TI_T_rad_s"
        )
        self.koz_center_L_m = _array(self.koz_center_L_m, (3,), "koz_center_L_m")
        self.camera_axis_B = _array(self.camera_axis_B, (3,), "camera_axis_B")

        q_norm = float(np.linalg.norm(self.target_q_LT0))
        if q_norm <= 1.0e-12:
            raise ValueError("target_q_LT0 must be non-zero")
        self.target_q_LT0 = self.target_q_LT0 / q_norm
        if not np.allclose(self.inertia_B_kg_m2, self.inertia_B_kg_m2.T, atol=1.0e-12):
            raise ValueError("inertia_B_kg_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(self.inertia_B_kg_m2)) <= 0.0:
            raise ValueError("inertia_B_kg_m2 must be positive definite")

        axis_norm = float(np.linalg.norm(self.camera_axis_B))
        if axis_norm <= 1.0e-12:
            raise ValueError("camera_axis_B must be non-zero")
        self.camera_axis_B = self.camera_axis_B / axis_norm
        if not (0.0 < self.fov_half_angle_rad < np.pi / 2.0):
            raise ValueError("fov_half_angle_rad must be in (0, pi/2)")
        if not (0.0 < self.min_camera_depth_m < self.max_camera_depth_m):
            raise ValueError("camera depth bounds are invalid")
        if self.reverse_pair_limit < 0.0 or self.reverse_pair_limit > 2.0:
            raise ValueError("reverse_pair_limit must be in [0, 2]")
        if self.koz_safety_margin_m2_s2 < 0.0:
            raise ValueError("koz_safety_margin_m2_s2 must be non-negative")
        if self.attitude_limit_rad <= 0.0 or self.attitude_limit_rad > np.pi:
            raise ValueError("attitude_limit_rad must be in (0, pi]")

        costs = (
            self.position_cost,
            self.velocity_cost,
            self.attitude_cost,
            self.angular_velocity_cost,
            self.duty_cost,
            self.duty_rate_cost,
            self.terminal_cost_scale,
        )
        if not all(np.isfinite(c) and c >= 0.0 for c in costs):
            raise ValueError("all cost weights must be finite and non-negative")

    @property
    def horizon_seconds(self) -> float:
        """Prediction horizon in seconds; fixed to 1 s for the 20 Hz contract."""

        return self.horizon_steps * self.control_period_s

    @property
    def max_duty_delta(self) -> float:
        """Maximum duty change per control interval from the rate limit."""

        return min(1.0, self.duty_rate_per_s * self.control_period_s)

    @property
    def action_block_starts(self) -> tuple[int, ...]:
        """Start indices of the fixed K=8 action blocks on the 20-step grid."""

        starts: list[int] = []
        index = 0
        for length in self.action_block_lengths:
            starts.append(index)
            index += length
        return tuple(starts)

    @property
    def target_omega_BI_B_rad_s(self) -> FloatArray:
        """Episode-constant target angular velocity in the aligned body axes."""

        return self.target_omega_TI_T_rad_s.copy()

    def signature(self) -> tuple[object, ...]:
        """Return a deterministic signature for solver-cache invalidation."""

        return (
            self.control_period_s,
            self.horizon_steps,
            self.rk4_substeps,
            self.rk4_substep_dt_s,
            self.mean_motion_rad_s,
            self.mass_kg,
            self.tmax_N,
            tuple(self.inertia_B_kg_m2.ravel()),
            tuple(self.com_B_m),
            None if self.rcs_matrix_B is None else tuple(self.rcs_matrix_B.ravel()),
            tuple(self.target_point_T_m),
            tuple(self.camera_target_point_T_m),
            tuple(self.target_q_LT0),
            tuple(self.target_omega_TI_T_rad_s),
            tuple(self.koz_center_L_m),
            self.koz_radius_m,
            self.koz_k0_s2,
            self.koz_k1_s,
            self.koz_safety_margin_m2_s2,
            self.velocity_limit_m_s,
            self.angular_velocity_limit_rad_s,
            self.duty_rate_per_s,
            self.reverse_pair_limit,
            self.enable_fov,
            tuple(self.camera_axis_B),
            self.fov_half_angle_rad,
            self.min_camera_depth_m,
            self.max_camera_depth_m,
            self.attitude_limit_rad,
            self.position_cost, self.velocity_cost, self.attitude_cost,
            self.angular_velocity_cost, self.duty_cost, self.duty_rate_cost,
            self.terminal_cost_scale,
        )


@dataclass
class AllocationResult:
    """Auditable result of mapping a body wrench to 16 RCS duties.

    ``wrench6_exec`` is exactly ``B @ duty16`` in body coordinates.  The
    residual is the Euclidean SI-unit difference from the requested wrench;
    callers must reject a result whose residual exceeds their configured
    tolerance.  ``runtime_s`` is measured at the allocation call timestamp.
    """

    status: str
    duty16: FloatArray
    wrench6_exec: FloatArray
    residual: float
    saturated: bool
    intervention: FloatArray
    constraint_margins: dict[str, float]
    runtime_s: float
    solver_status_code: int = 0
    error_message: str | None = None


@dataclass
class ExpertResult:
    """Solver output and independent quality verdict for one expert query.

    Arrays use float64 and are timestamped on the global 20 Hz grid: states
    have shape ``(N+1,13)``, duties ``(N,16)``, and wrenches ``(N,6)``.  States
    are LVLH/body quantities and wrenches are body-frame SI quantities.  The
    first entries of ``actual_duty`` and ``actual_wrench`` are the actions
    intended for execution at the query timestamp.  ``status`` reports the
    numerical solver only; ``quality_verdict`` is computed separately after
    RK4, RCS, and hard-constraint checks.  Failed solves retain an explicit
    failure reason and must not be treated as expert labels.
    """

    status: str
    solver_status_code: int
    iterations: int
    kkt: float
    runtime_s: float
    objective: float
    constraint_margins: dict[str, float]
    slacks: dict[str, float]
    allocation_residual: float
    warm_start: dict[str, FloatArray]
    predicted_state: FloatArray
    predicted_duty: FloatArray
    predicted_wrench: FloatArray
    actual_duty: FloatArray
    actual_wrench: FloatArray
    quality_verdict: str = "unknown"
    failure_reason: str | None = None
    sample_type: str = "normal"
    action_blocks: FloatArray | None = None

    @property
    def accepted(self) -> bool:
        """Whether the independent post-solve quality gate accepted the result."""

        return self.quality_verdict == "accepted"
