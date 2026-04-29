"""Tests for the combined_ply_export utility.

Tests the export_combined_bbox_mesh_ply function in isolation using
small synthetic meshes — no CadQuery, no kaolin, no GPU required.

Run:  pytest tests/wp4_aeroshell_optimizer/v2/test_combined_ply_export.py -v
"""

from __future__ import annotations

import importlib
import struct
from pathlib import Path
from typing import Any

import pytest
import torch

MODULE = "subprojects.wp4_aeroshell_optimizer.v2.src.utils.combined_ply_export"
V4_MODULE = "subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.mesh_assembly_mode_v4"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _triangle_mesh(offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a minimal single-triangle mesh (3 vertices, 1 face)."""
    verts = torch.tensor(
        [[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
    return verts, faces


def _import_fn() -> Any:
    """Import the function under test."""
    mod = importlib.import_module(MODULE)
    return mod.export_combined_bbox_mesh_ply


# ---------------------------------------------------------------------------
# Tests — red gate: these should FAIL before combined_ply_export.py exists
# ---------------------------------------------------------------------------


class TestExportCombinedBboxMeshPly:
    def test_creates_ply_file(self, tmp_path: Path) -> None:
        """A PLY file is created at the specified output path."""
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        out = tmp_path / "test.ply"
        export_fn(
            inner_verts=inner_v,
            inner_faces=inner_f,
            outer_verts=outer_v,
            outer_faces=outer_f,
            sdf_verts=sdf_v,
            sdf_faces=sdf_f,
            output_path=out,
            inner_color=(1.0, 0.55, 0.0),
            outer_color=(1.0, 0.35, 0.15),
            sdf_color=(0.2, 1.0, 1.0),
        )
        assert out.exists(), "PLY file was not created"

    def test_ply_has_correct_vertex_count(self, tmp_path: Path) -> None:
        """Merged PLY contains inner + outer + sdf vertex counts."""
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        out = tmp_path / "test.ply"
        export_fn(
            inner_verts=inner_v,
            inner_faces=inner_f,
            outer_verts=outer_v,
            outer_faces=outer_f,
            sdf_verts=sdf_v,
            sdf_faces=sdf_f,
            output_path=out,
            inner_color=(1.0, 0.55, 0.0),
            outer_color=(1.0, 0.35, 0.15),
            sdf_color=(0.2, 1.0, 1.0),
        )
        content = out.read_text(errors="replace")
        # PLY ASCII or binary header always contains vertex count
        total_expected = inner_v.shape[0] + outer_v.shape[0] + sdf_v.shape[0]
        assert f"element vertex {total_expected}" in content, (
            f"Expected 'element vertex {total_expected}' in PLY header, got:\n{content[:400]}"
        )

    def test_ply_has_correct_face_count(self, tmp_path: Path) -> None:
        """Merged PLY contains inner + outer + sdf face counts."""
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        out = tmp_path / "test.ply"
        export_fn(
            inner_verts=inner_v,
            inner_faces=inner_f,
            outer_verts=outer_v,
            outer_faces=outer_f,
            sdf_verts=sdf_v,
            sdf_faces=sdf_f,
            output_path=out,
            inner_color=(1.0, 0.55, 0.0),
            outer_color=(1.0, 0.35, 0.15),
            sdf_color=(0.2, 1.0, 1.0),
        )
        content = out.read_text(errors="replace")
        total_expected = inner_f.shape[0] + outer_f.shape[0] + sdf_f.shape[0]
        assert f"element face {total_expected}" in content, (
            f"Expected 'element face {total_expected}' in PLY header, got:\n{content[:400]}"
        )

    def test_raises_on_empty_inner_faces(self, tmp_path: Path) -> None:
        """Raises ValueError when inner mesh has zero faces."""
        export_fn = _import_fn()
        inner_v = torch.zeros((3, 3), dtype=torch.float32)
        inner_f = torch.zeros((0, 3), dtype=torch.int64)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        with pytest.raises(ValueError, match="inner"):
            export_fn(
                inner_verts=inner_v,
                inner_faces=inner_f,
                outer_verts=outer_v,
                outer_faces=outer_f,
                sdf_verts=sdf_v,
                sdf_faces=sdf_f,
                output_path=tmp_path / "test.ply",
                inner_color=(1.0, 0.55, 0.0),
                outer_color=(1.0, 0.35, 0.15),
                sdf_color=(0.2, 1.0, 1.0),
            )

    def test_raises_on_empty_outer_faces(self, tmp_path: Path) -> None:
        """Raises ValueError when outer mesh has zero faces."""
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v = torch.zeros((3, 3), dtype=torch.float32)
        outer_f = torch.zeros((0, 3), dtype=torch.int64)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        with pytest.raises(ValueError, match="outer"):
            export_fn(
                inner_verts=inner_v,
                inner_faces=inner_f,
                outer_verts=outer_v,
                outer_faces=outer_f,
                sdf_verts=sdf_v,
                sdf_faces=sdf_f,
                output_path=tmp_path / "test.ply",
                inner_color=(1.0, 0.55, 0.0),
                outer_color=(1.0, 0.35, 0.15),
                sdf_color=(0.2, 1.0, 1.0),
            )

    def test_raises_on_empty_sdf_faces(self, tmp_path: Path) -> None:
        """Raises ValueError when SDF mesh has zero faces."""
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v = torch.zeros((3, 3), dtype=torch.float32)
        sdf_f = torch.zeros((0, 3), dtype=torch.int64)

        with pytest.raises(ValueError, match="sdf"):
            export_fn(
                inner_verts=inner_v,
                inner_faces=inner_f,
                outer_verts=outer_v,
                outer_faces=outer_f,
                sdf_verts=sdf_v,
                sdf_faces=sdf_f,
                output_path=tmp_path / "test.ply",
                inner_color=(1.0, 0.55, 0.0),
                outer_color=(1.0, 0.35, 0.15),
                sdf_color=(0.2, 1.0, 1.0),
            )

    def test_accepts_gpu_tensors_if_available(self, tmp_path: Path) -> None:
        """Tensors on GPU are accepted (detach+cpu performed internally)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        out = tmp_path / "test.ply"
        export_fn(
            inner_verts=inner_v.cuda(),
            inner_faces=inner_f.cuda(),
            outer_verts=outer_v.cuda(),
            outer_faces=outer_f.cuda(),
            sdf_verts=sdf_v.cuda(),
            sdf_faces=sdf_f.cuda(),
            output_path=out,
            inner_color=(1.0, 0.55, 0.0),
            outer_color=(1.0, 0.35, 0.15),
            sdf_color=(0.2, 1.0, 1.0),
        )
        assert out.exists()

    def test_colors_encoded_as_uint8(self, tmp_path: Path) -> None:
        """Vertex colors are written as 0-255 uint8 values in PLY."""
        export_fn = _import_fn()
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        out = tmp_path / "test.ply"
        export_fn(
            inner_verts=inner_v,
            inner_faces=inner_f,
            outer_verts=outer_v,
            outer_faces=outer_f,
            sdf_verts=sdf_v,
            sdf_faces=sdf_f,
            output_path=out,
            inner_color=(1.0, 0.0, 0.0),  # pure red → (255, 0, 0)
            outer_color=(0.0, 1.0, 0.0),  # pure green
            sdf_color=(0.0, 0.0, 1.0),  # pure blue
        )
        # trimesh writes ASCII or binary — check the text portion
        raw = out.read_bytes()
        header_end = raw.find(b"end_header")
        header = raw[:header_end].decode("ascii", errors="replace")
        assert "red" in header, "Expected 'red' channel in PLY header"
        assert "green" in header, "Expected 'green' channel in PLY header"
        assert "blue" in header, "Expected 'blue' channel in PLY header"
        assert "alpha" in header, "Expected 'alpha' channel in PLY header"

    def test_alpha_encoded_in_vertex_data(self, tmp_path: Path) -> None:
        """Per-component alpha values are written as uint8 (0-255) in the PLY."""
        export_fn = _import_fn()
        # Single triangle per component — easy to count bytes.
        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        out = tmp_path / "alpha_test.ply"
        export_fn(
            inner_verts=inner_v,
            inner_faces=inner_f,
            outer_verts=outer_v,
            outer_faces=outer_f,
            sdf_verts=sdf_v,
            sdf_faces=sdf_f,
            output_path=out,
            inner_color=(1.0, 0.0, 0.0),
            outer_color=(0.0, 1.0, 0.0),
            sdf_color=(0.0, 0.0, 1.0),
            inner_alpha=1.0,  # 255
            outer_alpha=0.2,  # 51
            sdf_alpha=0.5,  # 128
        )

        raw = out.read_bytes()
        header_end = raw.find(b"end_header") + len(b"end_header\n")
        body = raw[header_end:]
        header_text = raw[:header_end].decode("ascii", errors="replace")

        # Determine bytes-per-vertex from the header (x y z r g b a → 3*float32 + 4*uint8 = 16)
        # trimesh binary PLY: float32 x3 + uchar x4 = 16 bytes per vertex
        bytes_per_vertex = 16  # 3*4 + 4*1
        total_verts = inner_v.shape[0] + outer_v.shape[0] + sdf_v.shape[0]  # = 9

        expected_alphas: list[int] = (
            [255] * inner_v.shape[0] + [round(0.2 * 255)] * outer_v.shape[0] + [round(0.5 * 255)] * sdf_v.shape[0]
        )

        if "binary_little_endian" in header_text:
            for i in range(total_verts):
                offset = i * bytes_per_vertex
                alpha_byte = struct.unpack_from("B", body, offset + 15)[0]
                assert alpha_byte == expected_alphas[i], f"Vertex {i}: expected alpha={expected_alphas[i]}, got {alpha_byte}"


# ---------------------------------------------------------------------------
# Tests — MeshAssemblyModeV4 final-flush behaviour
# ---------------------------------------------------------------------------


def _make_v4_instance(config: dict[str, Any]) -> Any:
    """Build a MeshAssemblyModeV4 shell with only the attributes needed by finalize()."""
    mod = importlib.import_module(V4_MODULE)
    instance = mod.MeshAssemblyModeV4.__new__(mod.MeshAssemblyModeV4)
    # Inject only what finalize() and the PLY-flush path read.
    instance.ply_export_enabled = config.get("ply_export_enabled", False)
    instance.ply_export_interval = config.get("ply_export_interval", 1)
    instance._ply_export_count = 0
    instance._last_ply_state = None
    instance.cached_bbox_tensors = {}
    instance.assembly_color = (1.0, 0.55, 0.0)
    instance.outer_color = (1.0, 0.35, 0.15)
    instance.mesh_color = (0.2, 1.0, 1.0)
    instance.outer_alpha = 0.2
    instance.mesh_alpha = 0.5
    instance.frame_storage = []
    instance.make_gif = False
    return instance


def _minimal_v4_config(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "camera_position": [1, 0, 0],
        "look_at": [0, 0, 0],
        "render_mode": "rgb",
        "fps": 10,
        "render_interval": 1,
        "ply_export_enabled": True,
        "ply_export_interval": 9999,  # larger than any epoch count in tests
    }
    cfg.update(overrides)
    return cfg


class TestMeshAssemblyModeV4FinalFlush:
    def test_finalize_writes_ply_when_no_interval_export_fired(self, tmp_path: Path) -> None:
        """finalize() exports a PLY when ply_export_interval > num_epochs."""
        instance = _make_v4_instance(_minimal_v4_config())

        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        mesh_dir = tmp_path / "meshes"
        mesh_dir.mkdir()

        # Simulate what update() does: populate internal caches.
        instance.cached_bbox_tensors[0] = (inner_v, inner_f, outer_v, outer_f)
        instance._last_ply_state = {
            "iteration": 7,
            "assembly_id": 0,
            "vertices": sdf_v,
            "faces": sdf_f,
            "mesh_dir": mesh_dir,
        }
        instance._ply_export_count = 0  # no interval export has fired

        # finalize() should flush the last state.
        instance.finalize(tmp_path / "out")

        ply_files = list(mesh_dir.glob("combined_bb_mesh_epoch*.ply"))
        assert len(ply_files) == 1, f"Expected 1 PLY file, found: {ply_files}"
        assert "0007" in ply_files[0].name

    def test_finalize_skips_flush_when_interval_export_already_fired(self, tmp_path: Path) -> None:
        """finalize() does NOT write a duplicate PLY if an interval export already happened."""
        instance = _make_v4_instance(_minimal_v4_config())

        inner_v, inner_f = _triangle_mesh(0.0)
        outer_v, outer_f = _triangle_mesh(10.0)
        sdf_v, sdf_f = _triangle_mesh(20.0)

        mesh_dir = tmp_path / "meshes"
        mesh_dir.mkdir()

        instance.cached_bbox_tensors[0] = (inner_v, inner_f, outer_v, outer_f)
        instance._last_ply_state = {
            "iteration": 5,
            "assembly_id": 0,
            "vertices": sdf_v,
            "faces": sdf_f,
            "mesh_dir": mesh_dir,
        }
        instance._ply_export_count = 1  # an interval export already fired

        instance.finalize(tmp_path / "out")

        ply_files = list(mesh_dir.glob("combined_bb_mesh_epoch*.ply"))
        assert len(ply_files) == 0, f"Expected no PLY flush, found: {ply_files}"
