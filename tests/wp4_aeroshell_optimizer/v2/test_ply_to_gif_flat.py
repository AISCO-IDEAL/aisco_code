"""Tests for generate_gifs_flat in ply_to_gif.py (Plan 012 / M3).

TDD gate: written BEFORE implementation.
Expected failure before M3: ImportError for generate_gifs_flat.

Covers:
  - generate_gifs_flat creates animation.gif from epoch_*.ply files
  - Existing generate_gifs() behaviour unchanged (slot pattern)
  - CLI auto-detects flat vs slot pattern

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_ply_to_gif_flat.py -v
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import open3d as o3d
import pytest

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (
    _FLAT_PATTERN,
    _SLOT_PATTERN,
    generate_gifs_flat,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_FACES = np.array([[0, 1, 2]], dtype=np.int32)


def _write_dummy_ply(path: Path) -> None:
    """Write a minimal valid PLY with vertex colours."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(_VERTS)
    mesh.triangles = o3d.utility.Vector3iVector(_FACES)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)


# ---------------------------------------------------------------------------
# Pattern constants
# ---------------------------------------------------------------------------


class TestPatternConstants:
    def test_flat_pattern_matches_epoch_files(self) -> None:
        assert _FLAT_PATTERN.match("epoch_00001.ply") is not None
        assert _FLAT_PATTERN.match("epoch_12345.ply") is not None

    def test_flat_pattern_rejects_slot_files(self) -> None:
        assert _FLAT_PATTERN.match("slot01_mesh_epoch00001.ply") is None

    def test_slot_pattern_still_matches_slot_files(self) -> None:
        assert _SLOT_PATTERN.search("slot01_mesh_epoch00001.ply") is not None

    def test_flat_pattern_captures_epoch_number(self) -> None:
        m = _FLAT_PATTERN.match("epoch_00042.ply")
        assert m is not None
        assert m.group(1) == "00042"


# ---------------------------------------------------------------------------
# generate_gifs_flat: basic behaviour (rendering mocked)
# ---------------------------------------------------------------------------


class TestGenerateGifsFlat:
    def test_raises_when_no_flat_plys(self, tmp_path: Path) -> None:
        # No epoch_*.ply files → ValueError
        with pytest.raises(ValueError, match="epoch_"):
            generate_gifs_flat(tmp_path)

    def test_produces_animation_gif(self, tmp_path: Path) -> None:
        for i in range(3):
            _write_dummy_ply(tmp_path / f"epoch_{i:05d}.ply")

        with (
            patch(
                "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif._render_frame",
                return_value=np.zeros((720, 1280, 3), dtype=np.uint8),
            ),
            patch("subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif.imageio") as mock_iio,
        ):
            generate_gifs_flat(tmp_path)
            assert mock_iio.mimsave.call_count == 1
            gif_path = mock_iio.mimsave.call_args[0][0]
            assert "animation.gif" in str(gif_path)

    def test_custom_output_path(self, tmp_path: Path) -> None:
        ply_dir = tmp_path / "plys"
        out_dir = tmp_path / "gifs"
        _write_dummy_ply(ply_dir / "epoch_00001.ply")

        with (
            patch(
                "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif._render_frame",
                return_value=np.zeros((720, 1280, 3), dtype=np.uint8),
            ),
            patch("subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif.imageio") as mock_iio,
        ):
            generate_gifs_flat(ply_dir, output_path=out_dir / "custom.gif")
            gif_path = mock_iio.mimsave.call_args[0][0]
            assert "custom.gif" in str(gif_path)

    def test_files_sorted_by_epoch(self, tmp_path: Path) -> None:
        # Write 3 PLYs out of order
        for i in [5, 1, 3]:
            _write_dummy_ply(tmp_path / f"epoch_{i:05d}.ply")

        rendered_paths: list[str] = []

        def capture_render(renderer: object, fpath: Path, *args: object, **kwargs: object) -> np.ndarray:
            rendered_paths.append(fpath.name)
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        with (
            patch(
                "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif._render_frame",
                side_effect=capture_render,
            ),
            patch("subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif.imageio"),
        ):
            generate_gifs_flat(tmp_path)

        assert rendered_paths == ["epoch_00001.ply", "epoch_00003.ply", "epoch_00005.ply"]

    def test_max_frames_downsampling(self, tmp_path: Path) -> None:
        # Write 10 PLYs; max_frames=3 should downsample to 3
        for i in range(10):
            _write_dummy_ply(tmp_path / f"epoch_{i:05d}.ply")

        rendered_count = []

        def capture_render(renderer: object, fpath: Path, *args: object, **kwargs: object) -> np.ndarray:
            rendered_count.append(1)
            return np.zeros((720, 1280, 3), dtype=np.uint8)

        with (
            patch(
                "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif._render_frame",
                side_effect=capture_render,
            ),
            patch("subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif.imageio"),
        ):
            generate_gifs_flat(tmp_path, max_frames=3)

        assert len(rendered_count) == 3
