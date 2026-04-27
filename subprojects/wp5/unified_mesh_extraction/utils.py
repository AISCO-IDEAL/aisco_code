"""Non-calculative utilities for mesh extraction: config loading, bbox normalisation, surface-area helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def _model_version(model: Any) -> int:
    """Return an integer that changes whenever `model`'s parameters are mutated.

    Steps:
      1) Grab the first parameter tensor of `model` via `model.parameters()`.
      2) Read its autograd `_version` counter (incremented on every in-place op
         and on `optimizer.step`). Fall back to 0 if `model` has no parameters
         or is not an `nn.Module`.
    Used by the mesh cache to auto-invalidate stale entries across training steps.
    """
    try:
        return int(next(iter(model.parameters()))._version)
    except (StopIteration, AttributeError, TypeError):
        return 0


def _normalise_bbox(bbox: Any) -> list[list[float]]:
    """Coerce a bbox argument into the canonical `[[xlo,xhi],[ylo,yhi],[zlo,zhi]]` form.

    Steps:
      1) If `bbox` is a flat 6-tuple/list of scalars, split it into 3 pairs.
      2) Otherwise assume it is already a 3x2 nested iterable and cast each
         bound to float.
    """
    if len(bbox) == 6 and all(isinstance(v, (int, float)) for v in bbox):
        f = [float(v) for v in bbox]
        return [[f[0], f[1]], [f[2], f[3]], [f[4], f[5]]]
    return [[float(ax[0]), float(ax[1])] for ax in bbox]


@dataclass(frozen=True, slots=True)
class MeshExtractionConfig:
    """Config for SDF mesh extraction. Constructed via `load_mesh_extraction_config`."""

    device: str
    isovalue: float
    bbox: list[list[float]]
    base_resolution: int
    subdivision_factor: int
    max_refine_levels: int
    batch_size: int
    target_num_vertices: int | None = None
    vertex_count_tolerance: float = 0.1


def _default_config_path() -> Path:
    return Path(__file__).with_name("mesh_extraction_config.yaml")


def load_mesh_extraction_config(config_path: Path | None = None) -> MeshExtractionConfig:
    """Load mesh extraction config from YAML.

    Steps:
      1) Resolve the YAML path (default: sibling `mesh_extraction_config.yaml`).
      2) Parse the file with `yaml.safe_load` and reject non-mapping payloads.
      3) Build an immutable `MeshExtractionConfig` dataclass, applying defaults
         for optional fields (`subdivision_factor`, `target_num_vertices`, ...).
    """
    path = config_path if config_path is not None else _default_config_path()
    with path.open("r", encoding="utf-8") as fh:
        p = yaml.safe_load(fh)
    if not isinstance(p, dict):
        raise TypeError(f"Mesh extraction config must be a mapping, got {type(p).__name__}.")
    return MeshExtractionConfig(
        device=str(p.get("device", "cuda")),
        isovalue=float(p.get("isovalue", 0.0)),
        bbox=_normalise_bbox(p["bbox"]),
        base_resolution=int(p["base_resolution"]),
        subdivision_factor=int(p.get("subdivision_factor", 2)),
        max_refine_levels=int(p.get("max_refine_levels", 0)),
        batch_size=int(p.get("batch_size", 65536)),
        target_num_vertices=int(p["target_num_vertices"]) if p.get("target_num_vertices") is not None else None,
        vertex_count_tolerance=float(p.get("vertex_count_tolerance", 0.1)),
    )


def compute_surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Triangle-mesh surface area (differentiable w.r.t. `vertices`).

    Steps:
      1) Short-circuit to 0 for empty meshes.
      2) Gather the 3 vertex positions of every triangle: shape (F, 3, 3).
      3) Compute two edge vectors and their cross product per triangle.
      4) Triangle area = 0.5 * ||cross||; return the sum over all triangles.
    """
    if vertices.numel() == 0 or faces.numel() == 0:
        return torch.tensor(0.0, device=vertices.device if vertices.numel() > 0 else faces.device)
    v = vertices[faces]
    cross = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
    return 0.5 * cross.norm(dim=1).sum()
