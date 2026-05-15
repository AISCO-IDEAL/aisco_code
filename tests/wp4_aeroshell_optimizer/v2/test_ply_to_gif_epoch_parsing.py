from __future__ import annotations

from pathlib import Path

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization import ply_to_gif


class TestExtractEpochFromPath:
    def test_extracts_flat_epoch_filename(self) -> None:
        path = Path("/tmp/mesh_assembly_v4/epoch_00042.ply")
        assert ply_to_gif._extract_epoch_from_path(path) == 42

    def test_extracts_flattened_slotted_epoch_from_parent_dir(self) -> None:
        path = Path("/tmp/mesh_assembly_v4/epoch_00123/slot_01_mesh.ply")
        assert ply_to_gif._extract_epoch_from_path(path) == 123

    def test_extracts_legacy_slotted_epoch_from_parent_dir(self) -> None:
        path = Path("/tmp/mesh_assembly_v4/epoch_00123/slot_01/mesh.ply")
        assert ply_to_gif._extract_epoch_from_path(path) == 123

    def test_extracts_epoch_from_tokenized_filename(self) -> None:
        path = Path("/tmp/mesh_assembly_v4/epoch_00077_slot_03_mesh.ply")
        assert ply_to_gif._extract_epoch_from_path(path) == 77

    def test_returns_none_when_epoch_missing(self) -> None:
        path = Path("/tmp/mesh_assembly_v4/no_epoch_here/slot_01/mesh.ply")
        assert ply_to_gif._extract_epoch_from_path(path) is None


class TestResolveOverlayEpoch:
    def test_falls_back_to_zero_when_unparseable(self, monkeypatch) -> None:
        monkeypatch.setattr(ply_to_gif, "_EPOCH_PARSE_WARNING_EMITTED", [False])
        path = Path("/tmp/unparseable/slot_01/mesh.ply")
        assert ply_to_gif._resolve_overlay_epoch(path) == 0

    def test_emits_parse_warning_only_once(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(ply_to_gif, "_EPOCH_PARSE_WARNING_EMITTED", [False])
        p1 = Path("/tmp/unparseable/a/mesh.ply")
        p2 = Path("/tmp/unparseable/b/mesh.ply")

        assert ply_to_gif._resolve_overlay_epoch(p1) == 0
        assert ply_to_gif._resolve_overlay_epoch(p2) == 0

        captured = capsys.readouterr()
        assert captured.out.count("Could not parse epoch") == 1
