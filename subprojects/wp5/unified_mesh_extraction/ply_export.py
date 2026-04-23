"""PLY export helper for unified mesh extraction."""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp

import trimesh

from subprojects.wp5.unified_mesh_extraction.cache import clear_mesh_cache  # noqa: F401 — re-exported
from subprojects.wp5.unified_mesh_extraction.sdf2mesh import (
    MeshExtractionConfig,
    MeshSDFExtractor,
    SDFModel,
)


def extract_sdf_to_temp_ply(
    sdf_model: SDFModel,
    config: MeshExtractionConfig | None = None,
    output_name: str = "mesh_sdf_output.ply",
) -> str:
    """Extract a mesh and write it to a temp PLY. Returns the file path."""
    ext = MeshSDFExtractor(config=config)
    v, f = ext.extract_mesh(sdf_model)
    path = Path(mkdtemp(prefix="mesh_sdf_")) / output_name
    trimesh.Trimesh(vertices=v.detach().cpu().numpy(), faces=f.detach().cpu().numpy(), process=False).export(path)
    return str(path)
