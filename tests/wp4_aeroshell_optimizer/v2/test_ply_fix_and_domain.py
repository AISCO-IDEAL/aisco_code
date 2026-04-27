"""TDD tests for Plan 010: PLY loader transforms and get_points per-axis bounds.

Tests cover:
  - _load_ply_tensors: axis permutation, X centering, Z grounding, scale=100
  - get_points: per-axis bounds produce output in correct mm ranges
  - _sample_points_for_shape ply domain: uses outer_prism cuboid bounds

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_ply_fix_and_domain.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import trimesh

_V2 = "subprojects.wp4_aeroshell_optimizer.v2.src"
PRETRAIN_LATENT = f"{_V2}.main_files.pretraining_latent"
LOSSES_UTILS = f"{_V2}.utils.losses_utils"
PRETRAIN_V1_1 = f"{_V2}.main_files.pretraining_v1_1"


def _import(module: str, name: str) -> Any:
    mod = importlib.import_module(module)
    return getattr(mod, name)


def _make_box_ply(tmp_path: Path) -> str:
    """Create a synthetic PLY file in decimetre units, mimicking the real pretraining shapes.

    Vertex coordinates stored in decimetres (the actual PLY files use this unit).
    After ``ply_scale=100`` (dm→mm) the extents become:
      - PLY_Z (→ AISCO_X after permutation): [-23, 35] dm → extent 5800mm
      - PLY_X (→ AISCO_Y): [5, 25] dm → extent 2000mm (deliberately off-centre to test X-centering)
      - PLY_Y (→ AISCO_Z): [0, 12] dm → extent 1200mm

    After X centering, AISCO_X centre = 0 (PLY_Z centre was 6 dm = 600mm → shifted out).
    After Z grounding, AISCO_Z_min = 100mm (ground clearance).
    """
    # Vertices in decimetres; PLY_X is off-centre [5,25] so X-centering can be verified
    verts = np.array(
        [
            [5, 0, -23],
            [25, 0, -23],
            [25, 12, -23],
            [5, 12, -23],
            [5, 0, 35],
            [25, 0, 35],
            [25, 12, 35],
            [5, 12, 35],
        ],
        dtype="float32",
    )  # stored directly as decimetres — do NOT multiply by a scale factor here

    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype="int64",
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    ply_path = tmp_path / "test_box.ply"
    mesh.export(str(ply_path))
    return str(ply_path)


# ---------------------------------------------------------------------------
# Tests for _load_ply_tensors — expect failures before M2 implementation
# ---------------------------------------------------------------------------


class TestLoadPlyTensors:
    """Verify that _load_ply_tensors applies correct scale, permutation, and grounding."""

    def test_longest_axis_is_x_after_load(self, tmp_path: Path) -> None:
        """After loading at scale=100 with permutation, axis 0 (AISCO X) must be longest."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")
        ply_path = _make_box_ply(tmp_path)
        verts, _faces, _bbox = _load_ply_tensors(ply_path, ply_scale=100.0)
        v = verts.numpy()
        extents = v.max(axis=0) - v.min(axis=0)
        assert extents[0] == pytest.approx(5800.0, abs=1.0), f"X extent should be 5800mm, got {extents[0]:.1f}"
        assert extents[0] >= extents[1], "AISCO X must be >= AISCO Y"
        assert extents[0] >= extents[2], "AISCO X must be >= AISCO Z"

    def test_x_centered_at_origin(self, tmp_path: Path) -> None:
        """X center must be at 0 after loading (car length centered on origin)."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")
        ply_path = _make_box_ply(tmp_path)
        verts, _faces, _bbox = _load_ply_tensors(ply_path, ply_scale=100.0)
        v = verts.numpy()
        x_center = (v[:, 0].max() + v[:, 0].min()) / 2.0
        assert x_center == pytest.approx(0.0, abs=1.0), f"X center should be 0, got {x_center:.1f}"

    def test_z_min_equals_ground_clearance(self, tmp_path: Path) -> None:
        """After Z grounding, Z_min must equal ground_clearance (100mm)."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")
        ply_path = _make_box_ply(tmp_path)
        verts, _faces, _bbox = _load_ply_tensors(ply_path, ply_scale=100.0)
        v = verts.numpy()
        z_min = float(v[:, 2].min())
        assert z_min == pytest.approx(100.0, abs=1.0), f"Z_min should be 100mm (ground clearance), got {z_min:.1f}"

    def test_bbox_extent_is_mm_scale(self, tmp_path: Path) -> None:
        """bbox_extent (diagonal) must be in mm range (hundreds to thousands), not sub-millimetre."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")
        ply_path = _make_box_ply(tmp_path)
        _verts, _faces, bbox_extent = _load_ply_tensors(ply_path, ply_scale=100.0)
        assert bbox_extent > 500.0, f"bbox_extent should be >500mm at scale=100, got {bbox_extent:.1f}"
        assert bbox_extent < 20_000.0, f"bbox_extent should be <20000mm at scale=100, got {bbox_extent:.1f}"

    def test_faces_preserved(self, tmp_path: Path) -> None:
        """Face count and indices must be unchanged by the transform."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")
        ply_path = _make_box_ply(tmp_path)
        _verts, faces, _bbox = _load_ply_tensors(ply_path, ply_scale=100.0)
        assert faces.shape == (12, 3), f"Box has 12 triangles, got {faces.shape}"

    def test_scale_1000_gives_wrong_size(self, tmp_path: Path) -> None:
        """At scale=1000 the PLY Z extent is 58000mm (far outside outer_prism, 10x too large).
        This documents why scale=1000 is wrong and scale=100 is correct."""
        ply_path = _make_box_ply(tmp_path)
        # Raw PLY vertices are in decimetres; PLY_Z: [-23, 35] dm → extent 58 dm
        # At scale=1000 (wrong): 58 dm * 1000 = 58000mm
        # At scale=100 (correct): 58 dm * 100 = 5800mm
        raw_mesh = trimesh.load(str(ply_path))
        v = np.array(raw_mesh.vertices) * 1000.0
        raw_z_extent = float(v[:, 2].max() - v[:, 2].min())
        assert raw_z_extent == pytest.approx(58000.0, abs=1.0), (
            f"At scale=1000, PLY Z extent should be 58000mm, got {raw_z_extent:.1f}"
        )


# ---------------------------------------------------------------------------
# Tests for get_points — expect failures before M5 implementation
# ---------------------------------------------------------------------------


class TestGetPointsPerAxis:
    """Verify that get_points() with per-axis bounds samples a cuboid, not a cube."""

    def test_points_within_x_bounds(self) -> None:
        """All sampled X coordinates must lie within [bbox_lo[0], bbox_hi[0]]."""
        get_points = _import(LOSSES_UTILS, "get_points")
        bbox_lo = [-2900.0, -1150.0, -100.0]
        bbox_hi = [2900.0, 1150.0, 1650.0]
        pts = get_points(num_points=1000, bbox_lo=bbox_lo, bbox_hi=bbox_hi, device="cpu")
        assert pts[:, 0].min().item() >= -2900.0
        assert pts[:, 0].max().item() <= 2900.0

    def test_points_within_y_bounds(self) -> None:
        get_points = _import(LOSSES_UTILS, "get_points")
        bbox_lo = [-2900.0, -1150.0, -100.0]
        bbox_hi = [2900.0, 1150.0, 1650.0]
        pts = get_points(num_points=1000, bbox_lo=bbox_lo, bbox_hi=bbox_hi, device="cpu")
        assert pts[:, 1].min().item() >= -1150.0
        assert pts[:, 1].max().item() <= 1150.0

    def test_points_within_z_bounds(self) -> None:
        get_points = _import(LOSSES_UTILS, "get_points")
        bbox_lo = [-2900.0, -1150.0, -100.0]
        bbox_hi = [2900.0, 1150.0, 1650.0]
        pts = get_points(num_points=1000, bbox_lo=bbox_lo, bbox_hi=bbox_hi, device="cpu")
        assert pts[:, 2].min().item() >= -100.0
        assert pts[:, 2].max().item() <= 1650.0

    def test_output_shape(self) -> None:
        get_points = _import(LOSSES_UTILS, "get_points")
        pts = get_points(num_points=500, bbox_lo=[-1.0, -1.0, -1.0], bbox_hi=[1.0, 1.0, 1.0], device="cpu")
        assert pts.shape == (500, 3)

    def test_requires_grad(self) -> None:
        get_points = _import(LOSSES_UTILS, "get_points")
        pts = get_points(num_points=100, bbox_lo=[-1.0, -1.0, -1.0], bbox_hi=[1.0, 1.0, 1.0], device="cpu")
        assert pts.requires_grad

    def test_asymmetric_bounds_coverage(self) -> None:
        """Sampling with asymmetric Z must not produce Z values outside the asymmetric range."""
        get_points = _import(LOSSES_UTILS, "get_points")
        # Z range is [-100, 1650] — not centred at 0
        bbox_lo = [-2900.0, -1150.0, -100.0]
        bbox_hi = [2900.0, 1150.0, 1650.0]
        pts = get_points(num_points=2000, bbox_lo=bbox_lo, bbox_hi=bbox_hi, device="cpu")
        z_vals = pts[:, 2].detach()
        assert z_vals.min().item() >= -110.0  # small float tolerance
        assert z_vals.max().item() <= 1660.0

    def test_non_cubic_x_wider_than_z(self) -> None:
        """With outer_prism bounds, X range (5800mm) must be wider than Z range (1750mm)."""
        get_points = _import(LOSSES_UTILS, "get_points")
        bbox_lo = [-2900.0, -1150.0, -100.0]
        bbox_hi = [2900.0, 1150.0, 1650.0]
        pts = get_points(num_points=2000, bbox_lo=bbox_lo, bbox_hi=bbox_hi, device="cpu")
        x_range = pts[:, 0].detach().max() - pts[:, 0].detach().min()
        z_range = pts[:, 2].detach().max() - pts[:, 2].detach().min()
        # X range (≈5800mm spread) should be clearly larger than Z range (≈1750mm spread)
        assert x_range.item() > z_range.item() * 1.5, (
            f"Expected X range > 1.5 * Z range (cuboid, not cube). X={x_range:.0f}, Z={z_range:.0f}"
        )
