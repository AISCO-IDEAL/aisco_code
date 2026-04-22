"""Tests for run_pipeline.py — orchestrator for OpenFOAM pipeline modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the openfoam package directory is importable
_OPENFOAM_DIR = str(Path(__file__).resolve().parent.parent)
if _OPENFOAM_DIR not in sys.path:
    sys.path.insert(0, _OPENFOAM_DIR)

from generate_cases import generate_all_cases  # noqa: E402
from ply_to_stl import batch_convert  # noqa: E402
from run_pipeline import write_run_list  # noqa: E402


class TestWriteRunList:
    """Tests for write_run_list()."""

    def test_writes_run_ids_to_file(self, tmp_path: Path) -> None:
        """write_run_list creates a file with one run ID per line."""
        out = tmp_path / "run_ids.txt"
        write_run_list([1, 10, 200], out)
        lines = out.read_text().strip().splitlines()
        assert lines == ["001", "010", "200"]

    def test_zero_pads_to_three_digits(self, tmp_path: Path) -> None:
        """Run IDs are zero-padded to 3 digits."""
        out = tmp_path / "run_ids.txt"
        write_run_list([5], out)
        assert out.read_text().strip() == "005"

    def test_empty_list_raises(self, tmp_path: Path) -> None:
        """Empty run_ids list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            write_run_list([], tmp_path / "run_ids.txt")

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        out = tmp_path / "sub" / "deep" / "run_ids.txt"
        write_run_list([42], out)
        assert out.exists()


class TestAppendSafeManifest:
    """Tests for append-safe manifest merge in ply_to_stl and generate_cases."""

    def test_ply_manifest_merge_preserves_existing(self, tmp_path: Path) -> None:
        """batch_convert with existing manifest merges, not overwrites."""
        manifest = tmp_path / "manifest.json"
        existing = [{"run_id": "RUN_001", "status": "converted", "stl_path": "/old/001.stl"}]
        manifest.write_text(json.dumps(existing))

        # batch_convert with no matching PLY files should preserve existing
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        batch_convert(input_dir, output_dir, manifest, run_ids=[999])
        # No RUN_999.ply exists, so result is empty new entries
        # But manifest should still contain RUN_001
        loaded = json.loads(manifest.read_text())
        assert any(e["run_id"] == "RUN_001" for e in loaded)

    def test_cases_manifest_merge_preserves_existing(self, tmp_path: Path) -> None:
        """generate_all_cases with existing manifest merges, not overwrites."""
        cases_manifest = tmp_path / "cases_manifest.json"
        existing = [{"run_id": "RUN_001", "status": "created", "case_path": "/old/RUN_001"}]
        cases_manifest.write_text(json.dumps(existing))

        # Empty geometry manifest
        geo_manifest = tmp_path / "geometry_manifest.json"
        geo_manifest.write_text("[]")

        template_dir = tmp_path / "template"
        template_dir.mkdir()
        output_dir = tmp_path / "cases"

        generate_all_cases(
            manifest_path=geo_manifest,
            template_dir=template_dir,
            output_dir=output_dir,
            run_ids=[999],
            cases_manifest_path=cases_manifest,
        )

        loaded = json.loads(cases_manifest.read_text())
        assert any(e["run_id"] == "RUN_001" for e in loaded)
