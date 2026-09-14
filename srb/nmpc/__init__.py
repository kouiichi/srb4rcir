"""Standalone 20 Hz NMPC expert components for Space Robotics Bench.

Importing this package does not initialize Isaac Sim.  The HCW/RCS plant is
therefore testable in a plain Python process, while the optional solver
classes lazily require CasADi and acados_template only when instantiated.
"""

from .dynamics import (
    casadi_rk4_map,
    continuous_dynamics,
    cw_acceleration,
    reference_at,
    reference_trajectory,
    rk4_step,
)
from .config import (
    NmpcConfig,
    NmpcConfigError,
    SolverConfig,
    load_nmpc_config,
    load_task_params,
    task_params_from_mapping,
)
from .rcs import allocate_wrench, build_rcs_matrix, rcs_matrix_from_task, validate_rcs_geometry
from .acados_expert import AcadosExpert
from .ipopt_oracle import IpoptOracle
from .safety import SafetyAllocator
from .srb_adapter import (
    Srb16Adapter,
    SrbAdapter,
    build_srb16_matrix,
    build_srb8_matrix,
    validate_srb8_geometry,
)
from .data import SAMPLE_TYPES, diagnostic_rows, write_episode_hdf5, write_parquet_index
from .dagger import DaggerDecision, DaggerThresholds, DaggerTrigger, select_visual_action_candidate
from .types import (
    AllocationResult,
    ExpertResult,
    RcsDuty16,
    State13,
    TaskParams,
    Wrench6Body,
)

__all__ = [
    "AllocationResult",
    "AcadosExpert",
    "ExpertResult",
    "NmpcConfig",
    "NmpcConfigError",
    "IpoptOracle",
    "SafetyAllocator",
    "Srb16Adapter",
    "SrbAdapter",
    "SolverConfig",
    "SAMPLE_TYPES",
    "DaggerDecision",
    "DaggerThresholds",
    "DaggerTrigger",
    "RcsDuty16",
    "State13",
    "TaskParams",
    "Wrench6Body",
    "allocate_wrench",
    "build_rcs_matrix",
    "build_srb16_matrix",
    "build_srb8_matrix",
    "casadi_rk4_map",
    "continuous_dynamics",
    "cw_acceleration",
    "rcs_matrix_from_task",
    "reference_at",
    "reference_trajectory",
    "rk4_step",
    "diagnostic_rows",
    "load_nmpc_config",
    "load_task_params",
    "select_visual_action_candidate",
    "task_params_from_mapping",
    "validate_rcs_geometry",
    "validate_srb8_geometry",
    "write_episode_hdf5",
    "write_parquet_index",
]
