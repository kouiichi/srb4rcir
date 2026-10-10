"""Visual data and material helpers used by SRB experiment recorders."""

from .camera_quality import summarize_camera_frame
from .materials import (
    apply_metallic_film_material,
    apply_satellite_materials,
    metallic_film_parameters,
    solar_panel_parameters,
)

__all__ = [
    "apply_metallic_film_material",
    "apply_satellite_materials",
    "metallic_film_parameters",
    "solar_panel_parameters",
    "summarize_camera_frame",
]
