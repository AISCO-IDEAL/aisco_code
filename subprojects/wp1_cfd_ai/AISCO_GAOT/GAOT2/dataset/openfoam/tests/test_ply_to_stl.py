"""Tests for ply_to_stl.py — geometry conversion module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.ply_to_stl import (
    _GROUND_CLEARANCE_M,
    batch_convert,
    compute_ground_translation,
    convert_single_ply,
    detect_units,
    repair_mesh,
)

# ---------------------------------------------------------------------------
#  Fixtures: synthetic meshes
# ---------------------------------------------------------------------------


@pytest.fixture
def watertight_box_mm() -> trimesh.Trimesh:
    """A watertight box with mm-scale coords (~4000mm long)."""
    return trimesh.creation.box(extents=[4000.0, 1800.0, 1500.0])


@pytest.fixture
def watertight_box_m() -> trimesh.Trimesh:
    """A watertight box already in meters (~4m long)."""
    return trimesh.creation.box(extents=[4.0, 1.8, 1.5])


@pytest.fixture
def non_watertight_mesh() -> trimesh.Trimesh:
    """A mesh with a hole (not watertight)."""
    box = trimesh.creation.box(extents=[3000.0, 1500.0, 1200.0])
    # Remove some faces to break watertightness
    faces = box.faces[:-4]
    return trimesh.Trimesh(vertices=box.vertices, faces=faces)


def _save_ply(mesh: trimesh.Trimesh, path: Path) -> None:
    """Helper: save trimesh as PLY."""
    mesh.export(str(path), file_type="ply")


# ---------------------------------------------------------------------------
#  detect_units
# ---------------------------------------------------------------------------


class TestDetectUnits:
    def test_detects_mm_for_large_extents(self, watertight_box_mm: trimesh.Trimesh) -> None:
        """Max extent > 20 → mm."""
        assert detect_units(watertight_box_mm) == "mm"

    def test_detects_m_for_small_extents(self, watertight_box_m: trimesh.Trimesh) -> None:
        """Max extent <= 20 → m."""
        assert detect_units(watertight_box_m) == "m"

    def test_boundary_at_20(self) -> None:
        """Box with max extent exactly 20 → m (not mm)."""
        box = trimesh.creation.box(extents=[20.0, 10.0, 5.0])
        assert detect_units(box) == "m"

    def test_slightly_above_20(self) -> None:
        """Box with max extent 20.1 → mm."""
        box = trimesh.creation.box(extents=[20.1, 10.0, 5.0])
        assert detect_units(box) == "mm"


# ---------------------------------------------------------------------------
#  repair_mesh
# ---------------------------------------------------------------------------


class TestRepairMesh:
    def test_watertight_mesh_passes_through(self, watertight_box_mm: trimesh.Trimesh) -> None:
        """Already watertight → returned as-is, success=True."""
        repaired, success = repair_mesh(watertight_box_mm)
        assert success is True
        assert repaired.is_watertight

    def test_non_watertight_mesh_repair_attempted(self, non_watertight_mesh: trimesh.Trimesh) -> None:
        """Non-watertight mesh → repair attempted, returns (mesh, bool)."""
        repaired, success = repair_mesh(non_watertight_mesh)
        # success depends on whether trimesh can fix it; just check types
        assert isinstance(repaired, trimesh.Trimesh)
        assert isinstance(success, bool)


# ---------------------------------------------------------------------------
#  convert_single_ply
# ---------------------------------------------------------------------------


class TestConvertSinglePly:
    def test_converts_mm_ply_to_meter_stl(self, watertight_box_mm: trimesh.Trimesh) -> None:
        """PLY in mm → STL in meters with correct bounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "RUN_001.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(watertight_box_mm, ply_path)

            result = convert_single_ply(ply_path, out_dir)

            assert result["run_id"] == "RUN_001"
            assert result["status"] == "converted"
            assert result["detected_unit"] == "mm"
            stl_path = Path(result["stl_path"])
            assert stl_path.exists()
            # Load STL and check bounds are in meters
            stl_mesh = trimesh.load(str(stl_path))
            max_extent = stl_mesh.bounding_box.extents.max()
            assert max_extent < 20.0, f"STL should be in meters, got max extent {max_extent}"

    def test_converts_m_ply_without_scaling(self, watertight_box_m: trimesh.Trimesh) -> None:
        """PLY already in m → STL not rescaled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "RUN_010.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(watertight_box_m, ply_path)

            result = convert_single_ply(ply_path, out_dir)

            assert result["detected_unit"] == "m"
            stl_mesh = trimesh.load(str(Path(result["stl_path"])))
            np.testing.assert_allclose(stl_mesh.bounding_box.extents.max(), 4.0, atol=0.01)

    def test_non_watertight_mesh_converts_with_warning(self) -> None:
        """A non-watertight mesh converts (with warning) rather than being excluded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a degenerate mesh (single triangle — not watertight, not repairable)
            verts = np.array([[0, 0, 0], [100, 0, 0], [50, 100, 0]], dtype=np.float64)
            faces = np.array([[0, 1, 2]])
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            ply_path = Path(tmpdir) / "RUN_999.ply"
            _save_ply(mesh, ply_path)
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()

            result = convert_single_ply(ply_path, out_dir)

            assert result["status"] == "converted"
            assert result["watertight_after_repair"] is False

    def test_geometry_lifted_to_minimum_ground_clearance_when_below_zero(self) -> None:
        """Centered meshes are snapped to the exact ground clearance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Box centered at Z=0: z spans -0.75 to +0.75
            mesh_centered = trimesh.creation.box(extents=[3.0, 1.5, 1.5])
            assert mesh_centered.bounds[0][2] < -0.1, "fixture must have z_min < 0"

            ply_path = Path(tmpdir) / "RUN_001.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(mesh_centered, ply_path)

            result = convert_single_ply(ply_path, out_dir)

            assert result["status"] == "converted"
            stl_mesh = trimesh.load(str(Path(result["stl_path"])))
            z_min_out = float(stl_mesh.bounds[0][2])
            np.testing.assert_allclose(z_min_out, _GROUND_CLEARANCE_M, atol=0.01)

    def test_geometry_below_exact_clearance_is_raised(self, watertight_box_m: trimesh.Trimesh) -> None:
        """Meshes below the target gap are lifted up to the exact clearance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # watertight_box_m is centred at origin so z_min < 0; translate it up
            mesh = watertight_box_m.copy()
            mesh.vertices[:, 2] += 0.8  # now z_min = 0.05 > 0 but still below target
            ply_path = Path(tmpdir) / "RUN_010.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(mesh, ply_path)

            z_min_in = float(mesh.bounds[0][2])
            result = convert_single_ply(ply_path, out_dir)

            stl_mesh = trimesh.load(str(Path(result["stl_path"])))
            z_min_out = float(stl_mesh.bounds[0][2])
            assert 0.0 < z_min_in < _GROUND_CLEARANCE_M, "fixture must start above ground but below the target clearance"
            np.testing.assert_allclose(z_min_out, _GROUND_CLEARANCE_M, atol=0.01)

    def test_geometry_raised_to_exact_ground_clearance_when_below_ground(self) -> None:
        """Meshes intersecting the ground are snapped to the exact clearance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_centered = trimesh.creation.box(extents=[3.0, 1.5, 1.5])
            ply_path = Path(tmpdir) / "RUN_020.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(mesh_centered, ply_path)

            result = convert_single_ply(ply_path, out_dir)

            stl_mesh = trimesh.load(str(Path(result["stl_path"])))
            z_min_out = float(stl_mesh.bounds[0][2])
            np.testing.assert_allclose(z_min_out, _GROUND_CLEARANCE_M, atol=0.01)

    def test_geometry_above_exact_clearance_is_lowered(self, watertight_box_m: trimesh.Trimesh) -> None:
        """Meshes above the target clearance are lowered to exactly the target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh = watertight_box_m.copy()
            mesh.vertices[:, 2] += 1.6  # z_min = 0.85 m
            ply_path = Path(tmpdir) / "RUN_021.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(mesh, ply_path)

            result = convert_single_ply(ply_path, out_dir)

            stl_mesh = trimesh.load(str(Path(result["stl_path"])))
            z_min_out = float(stl_mesh.bounds[0][2])
            np.testing.assert_allclose(z_min_out, _GROUND_CLEARANCE_M, atol=0.01)


class TestComputeGroundTranslation:
    def test_returns_positive_shift_when_mesh_intersects_ground(self) -> None:
        """Meshes below the target clearance should be lifted by the exact shortfall."""
        shift = compute_ground_translation(-0.75)
        np.testing.assert_allclose(shift, 0.85, atol=1e-9)

    def test_returns_negative_shift_when_mesh_above_target(self) -> None:
        """Meshes above the target clearance should be lowered."""
        shift = compute_ground_translation(0.85)
        np.testing.assert_allclose(shift, -0.75, atol=1e-9)

    def test_result_contains_bounds(self, watertight_box_mm: trimesh.Trimesh) -> None:
        """Result dict contains bounding box in meters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ply_path = Path(tmpdir) / "RUN_002.ply"
            out_dir = Path(tmpdir) / "output"
            out_dir.mkdir()
            _save_ply(watertight_box_mm, ply_path)

            result = convert_single_ply(ply_path, out_dir)

            assert "bounds_min" in result
            assert "bounds_max" in result
            assert len(result["bounds_min"]) == 3
            assert len(result["bounds_max"]) == 3


# ---------------------------------------------------------------------------
#  batch_convert
# ---------------------------------------------------------------------------


class TestBatchConvert:
    def test_batch_processes_multiple_plys(self, watertight_box_mm: trimesh.Trimesh) -> None:
        """Batch processes multiple PLYs and writes manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            manifest_path = Path(tmpdir) / "manifest.json"

            # Create 3 PLY files
            for i in range(1, 4):
                _save_ply(watertight_box_mm, input_dir / f"RUN_{i:03d}.ply")

            results = batch_convert(input_dir, output_dir, manifest_path)

            assert len(results) == 3
            assert manifest_path.exists()
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert len(manifest) == 3

    def test_batch_continues_on_failure(self) -> None:
        """Batch does not abort on per-run failures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            input_dir.mkdir()
            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()
            manifest_path = Path(tmpdir) / "manifest.json"

            # One valid, one degenerate
            box = trimesh.creation.box(extents=[3000.0, 1500.0, 1200.0])
            _save_ply(box, input_dir / "RUN_001.ply")

            verts = np.array([[0, 0, 0], [100, 0, 0], [50, 100, 0]], dtype=np.float64)
            faces = np.array([[0, 1, 2]])
            degen = trimesh.Trimesh(vertices=verts, faces=faces)
            _save_ply(degen, input_dir / "RUN_002.ply")

            results = batch_convert(input_dir, output_dir, manifest_path)

            # Both should appear in manifest, no crash
            assert len(results) == 2
            statuses = {r["run_id"]: r["status"] for r in results}
            assert statuses["RUN_001"] == "converted"
            # Non-watertight mesh is now converted with a warning, not excluded
            assert statuses["RUN_002"] == "converted"
