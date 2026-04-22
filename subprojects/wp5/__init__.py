"""Public API for subprojects.wp5.

This module exposes a small, explicit surface so external code can do:

    from subprojects.wp5 import *

and get the documented names. Wildcard imports and dynamic imports make static
analysis (Pylance/mypy/ruff) fail to detect available names, so we import the
public symbols explicitly and define __all__.
"""

from subprojects.wp5.src.dummy_2 import file_saving
from subprojects.wp5.src.dummy_3 import hihi
from subprojects.wp5.unified_mesh_extraction import (
    MeshExtractionConfig,
    MeshSDFExtractor,
    compute_surface_area,
    extract_sdf_to_temp_ply,
    load_mesh_extraction_config,
)

__all__ = [  # these functions will be imported when calling from subprojects.wp5 import *
    "MeshExtractionConfig",
    "MeshSDFExtractor",
    "compute_surface_area",
    "extract_sdf_to_temp_ply",
    "file_saving",
    "hihi",
    "load_mesh_extraction_config",
]
