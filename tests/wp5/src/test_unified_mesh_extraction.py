from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from skimage import measure

from subprojects.wp5.unified_mesh_extraction import (
    MeshSDFExtractor,
    compute_surface_area,
    load_mesh_extraction_config,
)


class SphereSDF(torch.nn.Module):
    def __init__(self, radius: float) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(points, dim=-1) - self.radius


def _dense_baseline_surface_area(
    sdf_model: torch.nn.Module,
    bbox: list[list[float]],
    resolution: int,
    isovalue: float,
) -> float:
    axes = [np.linspace(bbox[index][0], bbox[index][1], resolution, dtype=np.float32) for index in range(3)]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")
    points = np.stack([grid_x, grid_y, grid_z], axis=-1).reshape(-1, 3)
    with torch.no_grad():
        sdf_values = sdf_model(torch.from_numpy(points)).reshape(resolution, resolution, resolution).cpu().numpy()

    vertices, faces, _, _ = measure.marching_cubes(
        sdf_values,
        level=isovalue,
        spacing=(
            (bbox[0][1] - bbox[0][0]) / float(resolution - 1),
            (bbox[1][1] - bbox[1][0]) / float(resolution - 1),
            (bbox[2][1] - bbox[2][0]) / float(resolution - 1),
        ),
    )
    vertices[:, 0] += bbox[0][0]
    vertices[:, 1] += bbox[1][0]
    vertices[:, 2] += bbox[2][0]
    return compute_surface_area(torch.from_numpy(vertices.astype(np.float32)), torch.from_numpy(faces.astype(np.int64))).item()


def test_load_mesh_extraction_config() -> None:
    config = load_mesh_extraction_config()

    assert config.base_resolution >= 2
    # bbox is intentionally absent from YAML; callers must provide via set_grid()
    assert config.bbox is None


def test_active_cell_extraction_surface_area_stays_close_to_dense_baseline() -> None:
    base_config = load_mesh_extraction_config()
    test_config = replace(
        base_config,
        device="cpu",
        base_resolution=16,
        max_refine_levels=3,
        batch_size=1024,
    )
    # Provide an explicit bbox; the sphere has radius 1500mm so this bbox contains it.
    _test_bbox = [[-2000.0, 2000.0], [-2000.0, 2000.0], [-2000.0, 2000.0]]
    extractor = MeshSDFExtractor(config=test_config)
    extractor.set_grid(bbox=_test_bbox, resolution=test_config.base_resolution)

    model = SphereSDF(radius=1500.0)
    vertices, faces = extractor.extract_mesh(model)

    adaptive_area = compute_surface_area(vertices, faces).item()
    dense_area = _dense_baseline_surface_area(model, _test_bbox, resolution=48, isovalue=test_config.isovalue)

    assert vertices.shape[0] > 0
    assert faces.shape[0] > 0
    assert math.isfinite(adaptive_area)
    assert abs(adaptive_area - dense_area) / dense_area < 0.05


class _LatentSphereSDF(torch.nn.Module):
    """Sphere whose radius is controlled by a registered buffer `_active_z[0]`.

    Mimics ConcatBaseModel: `set_latent_vector` replaces the buffer reference
    (not an in-place op) so `_version` of model parameters is unchanged between
    calls — exactly the scenario that exposed the slot-collision cache bug.
    """

    def __init__(self) -> None:
        super().__init__()
        # Dummy trainable parameter so _model_version() has something to read.
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("_active_z", torch.tensor([1000.0]), persistent=False)

    def set_latent_vector(self, z: torch.Tensor) -> None:
        self._active_z = z

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        radius = self._active_z[0].item()
        return torch.linalg.norm(points, dim=-1) - radius


def test_extract_mesh_differentiates_latent_slots() -> None:
    """Regression: extract_mesh must return distinct meshes for different latent
    vectors even when model parameters (and therefore _version) are unchanged.

    Prior to the fix, the cache key did not include _active_z, so all slots after
    the first returned the same (slot-0) mesh within one optimizer step.
    """
    base_config = load_mesh_extraction_config()
    test_config = replace(base_config, device="cpu", base_resolution=8, max_refine_levels=1, batch_size=512)
    _bbox = [[-2000.0, 2000.0], [-2000.0, 2000.0], [-2000.0, 2000.0]]

    extractor = MeshSDFExtractor(config=test_config)
    extractor.set_grid(bbox=_bbox, resolution=test_config.base_resolution)

    model = _LatentSphereSDF()

    # Slot 0: small sphere (radius 500)
    model.set_latent_vector(torch.tensor([500.0]))
    v0, _ = extractor.extract_mesh(model)
    area0 = compute_surface_area(v0, _).item() if v0.numel() > 0 else 0.0

    # Slot 1: large sphere (radius 1500) — same model, no optimizer.step()
    model.set_latent_vector(torch.tensor([1500.0]))
    v1, _ = extractor.extract_mesh(model)
    area1 = compute_surface_area(v1, _).item() if v1.numel() > 0 else 0.0

    # Areas should differ by ~9x (4π·r² ratio = (1500/500)² = 9)
    assert v0.shape != v1.shape or not torch.allclose(v0, v1), (
        "extract_mesh returned the same mesh for two different latent vectors. "
        "The mesh cache is not differentiating on _active_z."
    )
    assert area1 > area0 * 4, (
        f"Expected area1 >> area0 (large vs small sphere), got area0={area0:.0f}, area1={area1:.0f}. "
        "Cache is not differentiating on _active_z."
    )
