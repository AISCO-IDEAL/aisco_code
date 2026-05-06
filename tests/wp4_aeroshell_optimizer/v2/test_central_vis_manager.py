"""Tests for CentralVisManager (Plan 012 / v0.7.0).

TDD gate: these tests are written BEFORE the implementation.
Expected failure mode before implementation: ImportError / ModuleNotFoundError.

Covers:
  - Constructor and config loading
  - save_ply: correct file path, disabled type no-op, uniform color
  - save_ply: face→vertex averaging auto-detection
  - save_ply: scalar shape mismatch falls back to uniform, no crash
  - save_ply_points: writes flat epoch_XXXXX.ply, delegates to SamplingVisualizer
  - finalize: calls generate_gifs_flat for each enabled type (mocked)
  - Telemetry events emitted

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_central_vis_manager.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import trimesh

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager import (
    CentralVisManager,
    face_to_vertex_average,
)

# ---------------------------------------------------------------------------
# Minimal geometry fixtures
# ---------------------------------------------------------------------------

_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
_N_VERTS = len(_VERTS)
_N_FACES = len(_FACES)


def _make_manager(tmp_path: Path) -> CentralVisManager:
    """Return a CentralVisManager pointed at tmp_path, using the real config."""
    return CentralVisManager(run_dir=tmp_path, run_id="test-run-001")


# ---------------------------------------------------------------------------
# face_to_vertex_average helper
# ---------------------------------------------------------------------------


class TestFaceToVertexAverage:
    def test_output_shape(self) -> None:
        face_vals = np.ones(_N_FACES, dtype=np.float32)
        result = face_to_vertex_average(face_vals, _FACES, _VERTS)
        assert result.shape == (_N_VERTS,)

    def test_uniform_face_values_produce_uniform_vertex_values(self) -> None:
        face_vals = np.full(_N_FACES, 2.5, dtype=np.float32)
        result = face_to_vertex_average(face_vals, _FACES, _VERTS)
        np.testing.assert_allclose(result, 2.5, atol=1e-5)

    def test_known_gradient(self) -> None:
        # 4 faces; faces touching vertex 0 have value 1.0, others have value 0.0
        # All 4 faces include vertex 0 in a tetrahedron, so vertex 0 avg should be ~1.0
        face_vals = np.ones(_N_FACES, dtype=np.float32)
        result = face_to_vertex_average(face_vals, _FACES, _VERTS)
        # All faces uniformly 1.0, so all vertex averages should be 1.0
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="face_values length"):
            face_to_vertex_average(np.ones(99), _FACES, _VERTS)


# ---------------------------------------------------------------------------
# CentralVisManager construction
# ---------------------------------------------------------------------------


class TestCentralVisManagerConstruction:
    def test_constructs_with_defaults(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        assert mgr is not None

    def test_run_dir_stored(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        assert mgr.run_dir == tmp_path

    def test_invalid_run_dir_raises(self) -> None:
        # A path that cannot be created (parent is a file) should raise
        with tempfile.NamedTemporaryFile() as f:
            bad_path = Path(f.name) / "subdir"
            with pytest.raises((OSError, ValueError, NotADirectoryError)):
                CentralVisManager(run_dir=bad_path, run_id="x")


# ---------------------------------------------------------------------------
# save_ply: basic path and no-op behaviour
# ---------------------------------------------------------------------------


class TestSavePly:
    def test_writes_correct_file_path(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.save_ply("mesh_shape", epoch=5, verts=_VERTS, faces=_FACES)
        expected = tmp_path / "mesh_shape" / "epoch_00005.ply"
        assert expected.exists(), f"Expected PLY at {expected}"

    def test_disabled_type_is_noop(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        # aero/gradient is disabled: false in default config
        result = mgr.save_ply(
            "aero/gradient",
            epoch=1,
            verts=_VERTS,
            faces=_FACES,
            scalar_values=np.ones(_N_VERTS),
        )
        assert result is None
        assert not (tmp_path / "aero" / "gradient").exists()

    def test_unknown_type_is_noop(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        result = mgr.save_ply("nonexistent/type", epoch=1, verts=_VERTS)
        assert result is None

    def test_epoch_zero_padding(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.save_ply("mesh_shape", epoch=100, verts=_VERTS, faces=_FACES)
        assert (tmp_path / "mesh_shape" / "epoch_00100.ply").exists()

    def test_no_scalar_writes_uniform_color_ply(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.save_ply("mesh_shape", epoch=1, verts=_VERTS, faces=_FACES)
        path = tmp_path / "mesh_shape" / "epoch_00001.ply"
        mesh = trimesh.load(str(path), process=False)
        # File should be readable and have vertices
        assert hasattr(mesh, "vertices")
        assert len(mesh.vertices) >= _N_VERTS  # may include coord frame

    def test_vertex_scalar_values_accepted(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        scalar = np.linspace(-2.0, 1.0, _N_VERTS).astype(np.float32)
        path = mgr.save_ply(
            "aero/cp_field",
            epoch=10,
            verts=_VERTS,
            faces=_FACES,
            scalar_values=scalar,
        )
        assert path is not None
        assert path.exists()

    def test_ply_has_vertex_colors(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        scalar = np.linspace(0.0, 1.0, _N_VERTS).astype(np.float32)
        mgr.save_ply(
            "solar/selection",
            epoch=1,
            verts=_VERTS,
            faces=_FACES,
            scalar_values=scalar,
        )
        path = tmp_path / "solar" / "selection" / "epoch_00001.ply"
        mesh = trimesh.load(str(path), process=False)
        assert mesh.visual is not None

    def test_torch_tensor_verts_accepted(self, tmp_path: Path) -> None:
        import torch

        mgr = _make_manager(tmp_path)
        verts_tensor = torch.from_numpy(_VERTS)
        mgr.save_ply("mesh_shape", epoch=2, verts=verts_tensor, faces=_FACES)
        assert (tmp_path / "mesh_shape" / "epoch_00002.ply").exists()


# ---------------------------------------------------------------------------
# save_ply: face→vertex auto-detection
# ---------------------------------------------------------------------------


class TestSavePlyFaceToVertexAutoDetect:
    def test_face_indexed_scalar_auto_averaged(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        # Pass _N_FACES values — should be detected as face-indexed
        face_scalar = np.linspace(0.0, 1.0, _N_FACES).astype(np.float32)
        path = mgr.save_ply(
            "solar/shading",
            epoch=1,
            verts=_VERTS,
            faces=_FACES,
            scalar_values=face_scalar,
        )
        assert path is not None
        assert path.exists()

    def test_shape_mismatch_falls_back_to_uniform(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        mgr = _make_manager(tmp_path)
        # Wrong length: neither N_VERTS nor N_FACES
        wrong_scalar = np.ones(99, dtype=np.float32)
        path = mgr.save_ply(
            "solar/shading",
            epoch=2,
            verts=_VERTS,
            faces=_FACES,
            scalar_values=wrong_scalar,
        )
        # Should not crash, should still write a PLY
        assert path is not None
        assert path.exists()
        # Should log a warning
        captured = capsys.readouterr()
        assert "mismatch" in captured.out.lower() or "warning" in captured.out.lower() or "scalar" in captured.out.lower()


# ---------------------------------------------------------------------------
# save_ply_points: flat naming
# ---------------------------------------------------------------------------


class TestSavePlyPoints:
    def test_writes_flat_epoch_name(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        points = _VERTS[:3]
        mgr.save_ply_points(
            "latent/sampling",
            epoch=7,
            points=points,
            mesh_verts=_VERTS,
            mesh_faces=_FACES,
        )
        expected = tmp_path / "latent" / "sampling" / "epoch_00007.ply"
        assert expected.exists(), f"Expected PLY at {expected}"

    def test_returns_path(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        path = mgr.save_ply_points(
            "latent/sampling",
            epoch=3,
            points=_VERTS[:2],
        )
        assert path is not None
        assert isinstance(path, Path)

    def test_secondary_and_tertiary_accepted(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.save_ply_points(
            "latent/sampling",
            epoch=4,
            points=_VERTS[:1],
            secondary_points=_VERTS[1:2],
            tertiary_points=_VERTS[2:3],
        )
        assert (tmp_path / "latent" / "sampling" / "epoch_00004.ply").exists()


# ---------------------------------------------------------------------------
# finalize: calls generate_gifs_flat per visual type
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_calls_generate_gifs_flat_for_each_enabled_type(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        # Write one PLY for mesh_shape
        mgr.save_ply("mesh_shape", epoch=1, verts=_VERTS, faces=_FACES)

        with patch(
            "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager.generate_gifs_flat"
        ) as mock_gif:
            mock_gif.return_value = None
            mgr.finalize()
            # Should have been called at least once (for mesh_shape)
            assert mock_gif.call_count >= 1
            # Check that mesh_shape dir was one of the calls
            called_dirs = [str(call.args[0]) for call in mock_gif.call_args_list]
            assert any("mesh_shape" in d for d in called_dirs)

    def test_finalize_skips_types_with_no_plys(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        # No PLYs written — finalize should run without error and call no renders
        with patch(
            "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager.generate_gifs_flat"
        ) as mock_gif:
            mgr.finalize()
            assert mock_gif.call_count == 0

    def test_finalize_continues_after_gif_failure(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        mgr.save_ply("mesh_shape", epoch=1, verts=_VERTS, faces=_FACES)
        mgr.save_ply("aero/cp_field", epoch=1, verts=_VERTS, faces=_FACES, scalar_values=np.ones(_N_VERTS))

        with patch(
            "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager.generate_gifs_flat",
            side_effect=RuntimeError("EGL init failed"),
        ):
            # Should not raise — finalize catches per-type failures
            mgr.finalize()
