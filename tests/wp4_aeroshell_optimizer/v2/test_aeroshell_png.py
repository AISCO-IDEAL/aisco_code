"""Tests for subprojects/postprocessing_visuals/aeroshell_png.py.

Follows the TDD-first pattern: each test covers one unit of the module.
No GPU, no real checkpoints, no solar pipeline — all heavy I/O is mocked.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import trimesh

# ---------------------------------------------------------------------------
# Helper: import the module under test without triggering CUDA extensions
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).parents[3] / "subprojects" / "postprocessing_visuals" / "aeroshell_png.py"


def _load_aeroshell_png() -> Any:
    """Import aeroshell_png.py via spec so it doesn't need to be a package."""
    spec = importlib.util.spec_from_file_location("aeroshell_png", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# _polar_to_z
# ---------------------------------------------------------------------------


class TestPolarToZ:
    def test_zero_angle_places_r_on_z0(self) -> None:
        mod = _load_aeroshell_png()
        z = mod._polar_to_z(radius=2.0, angle_deg=0.0, z_dim=4)
        assert isinstance(z, torch.Tensor)
        assert z.shape == (4,)
        assert math.isclose(z[0].item(), 2.0, abs_tol=1e-5)
        assert math.isclose(z[1].item(), 0.0, abs_tol=1e-5)

    def test_90_degree_places_r_on_z1(self) -> None:
        mod = _load_aeroshell_png()
        z = mod._polar_to_z(radius=3.0, angle_deg=90.0, z_dim=4)
        assert math.isclose(z[0].item(), 0.0, abs_tol=1e-5)
        assert math.isclose(z[1].item(), 3.0, abs_tol=1e-5)

    def test_higher_dims_are_zero(self) -> None:
        mod = _load_aeroshell_png()
        z = mod._polar_to_z(radius=1.0, angle_deg=45.0, z_dim=8)
        for i in range(2, 8):
            assert z[i].item() == 0.0

    def test_norm_equals_radius(self) -> None:
        mod = _load_aeroshell_png()
        for angle in [0, 30, 90, 135, 180, 270]:
            z = mod._polar_to_z(radius=2.5, angle_deg=float(angle), z_dim=4)
            assert math.isclose(float(torch.norm(z).item()), 2.5, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# _parse_color_rgb / _parse_color_rgba_float
# ---------------------------------------------------------------------------


class TestColorParsers:
    def test_hex_to_rgb(self) -> None:
        mod = _load_aeroshell_png()
        assert mod._parse_color_rgb("#33FFFF") == (51, 255, 255)

    def test_list_to_rgb(self) -> None:
        mod = _load_aeroshell_png()
        assert mod._parse_color_rgb([51, 255, 255]) == (51, 255, 255)

    def test_rgba_float_range(self) -> None:
        mod = _load_aeroshell_png()
        r, g, _b, a = mod._parse_color_rgba_float("#FF0000", alpha=0.5)
        assert math.isclose(r, 1.0, abs_tol=1e-4)
        assert math.isclose(g, 0.0, abs_tol=1e-4)
        assert math.isclose(a, 0.5, abs_tol=1e-4)


# ---------------------------------------------------------------------------
# _get config helper
# ---------------------------------------------------------------------------


class TestGetHelper:
    def test_nested_key(self) -> None:
        mod = _load_aeroshell_png()
        cfg = {"layers": {"aeroshell": {"alpha": 0.8}}}
        assert mod._get(cfg, "layers", "aeroshell", "alpha") == 0.8

    def test_missing_key_returns_default(self) -> None:
        mod = _load_aeroshell_png()
        cfg: dict = {}
        assert mod._get(cfg, "does_not_exist", default=42) == 42

    def test_none_node_returns_default(self) -> None:
        mod = _load_aeroshell_png()
        cfg = {"layers": None}
        assert mod._get(cfg, "layers", "aeroshell", default="x") == "x"


# ---------------------------------------------------------------------------
# compute_camera_pose
# ---------------------------------------------------------------------------


class TestComputeCameraPose:
    def _dummy_vertices(self) -> np.ndarray:
        # Simple unit cube centred at origin
        return np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )

    def test_returns_4x4(self) -> None:
        mod = _load_aeroshell_png()
        pose = mod.compute_camera_pose(self._dummy_vertices(), azim_deg=-60.0, elev_deg=25.0)
        assert pose.shape == (4, 4)

    def test_rotation_is_orthonormal(self) -> None:
        mod = _load_aeroshell_png()
        pose = mod.compute_camera_pose(self._dummy_vertices())
        rot = pose[:3, :3]
        identity = rot.T @ rot
        assert np.allclose(identity, np.eye(3), atol=1e-6)

    def test_camera_faces_centroid(self) -> None:
        """Camera -Z column should point roughly toward mesh centroid."""
        mod = _load_aeroshell_png()
        verts = self._dummy_vertices()
        pose = mod.compute_camera_pose(verts)
        cam_pos = pose[:3, 3]
        cam_fwd = -pose[:3, 2]  # -Z in OpenGL convention = forward
        to_centroid = verts.mean(axis=0) - cam_pos
        to_centroid /= np.linalg.norm(to_centroid)
        dot = float(np.dot(cam_fwd, to_centroid))
        assert dot > 0.99, f"Camera not facing centroid: dot={dot:.4f}"


# ---------------------------------------------------------------------------
# compute_solar_fields  (mocked solar pipeline)
# ---------------------------------------------------------------------------


class TestComputeSolarFields:
    def _make_sphere_mesh(self) -> trimesh.Trimesh:
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=500.0)
        return sphere

    def test_returns_binary_mask_and_shading_tensors(self) -> None:
        mod = _load_aeroshell_png()
        mesh = self._make_sphere_mesh()
        verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        faces = torch.tensor(mesh.faces, dtype=torch.long)
        sun_dir = torch.tensor([0.0, -0.5, 1.0], dtype=torch.float32)

        binary_mask, shading = mod.compute_solar_fields(
            vertices=verts,
            faces=faces,
            sun_dir=sun_dir,
            area_budget_mm2=4_200_000.0,
            effective_irradiance_W_m2=250.0,
            sigmoid_sharpness=10.0,
            soft_mask_selection_threshold=0.5,
        )

        assert binary_mask.shape == (faces.shape[0],), "binary_mask must be (F,)"
        assert shading.shape == (faces.shape[0],), "shading must be (F,)"
        assert binary_mask.dtype in (torch.float32, torch.bool)
        assert float(binary_mask.min()) >= 0.0
        assert float(binary_mask.max()) <= 1.0


# ---------------------------------------------------------------------------
# load_solar_ply_layer  (file-based artifact path)
# ---------------------------------------------------------------------------


class TestLoadSolarPlyLayer:
    def test_missing_file_warns_and_returns_none(self, tmp_path: Path) -> None:
        mod = _load_aeroshell_png()
        with pytest.warns(UserWarning, match="solar"):
            result = mod.load_solar_ply_layer(tmp_path / "nonexistent.ply")
        assert result is None

    def test_valid_ply_returns_trimesh(self, tmp_path: Path) -> None:
        mod = _load_aeroshell_png()
        sphere = trimesh.creation.icosphere(subdivisions=1, radius=100.0)
        ply_path = tmp_path / "test.ply"
        sphere.export(str(ply_path))
        result = mod.load_solar_ply_layer(ply_path)
        assert isinstance(result, trimesh.Trimesh)
