"""Runtime physical parameter ABI: mass, J, J inverse, B (C-order)."""
from dataclasses import replace
import numpy as np
from .rcs import rcs_matrix_from_task


def pack_physical_parameters(params):
    return np.r_[params.mass_kg, params.inertia_B_kg_m2.ravel(),
                 np.linalg.inv(params.inertia_B_kg_m2).ravel(), rcs_matrix_from_task(params).ravel()]


def structural_signature(params):
    """Physical parameters vary online; all constraints/costs remain structural."""
    return replace(params, mass_kg=1., inertia_B_kg_m2=np.eye(3), com_B_m=np.zeros(3),
                   tmax_N=1., rcs_matrix_B=None).signature()
