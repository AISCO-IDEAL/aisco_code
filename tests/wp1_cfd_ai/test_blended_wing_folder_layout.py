"""Tests for blended-wing folderized output layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset import preprocessing_blended_wing as bw_module


class _DummyTqdm:
    """No-op progress bar replacement for deterministic tests."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc

    def update(self, _value: int) -> None:
        return

    def close(self) -> None:
        return


class _FakeMesh:
    """Minimal mesh object emulating the PyVista API used by the writer."""

    def __init__(self) -> None:
        self.n_points = 3
        self.faces = np.array([3, 0, 1, 2], dtype=np.int64)
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.point_data: dict[str, Any] = {
            "Normals": np.array(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "cp": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        }

    def triangulate(self) -> _FakeMesh:
        return self

    def compute_normals(self, cell_normals: bool, point_normals: bool, auto_orient_normals: bool) -> _FakeMesh:
        assert not cell_normals
        assert point_normals
        assert auto_orient_normals
        return self


def test_process_single_saves_case_subfolder_with_data_pt(monkeypatch, tmp_path: Path) -> None:
    """Each blended-wing sample must be saved to case_XXXX/data.pt."""

    monkeypatch.setattr(bw_module.pv, "read", lambda _path: _FakeMesh())

    case_lookup = {"case_0001": {"CD": 0.12, "CL": 0.01, "M_inf": 0.3}}
    vtk_file = tmp_path / "case_0001.vtk"
    vtk_file.write_text("dummy")

    result = bw_module._process_single(str(vtk_file), case_lookup, str(tmp_path))

    assert result == "case_0001"
    assert (tmp_path / "case_0001" / "data.pt").exists()


def test_process_blended_wing_skips_existing_case_folder_and_orders_case_names(monkeypatch, tmp_path: Path) -> None:
    """Existing case_XXXX/data.pt should be detected and order_.txt should list case names."""

    vtk_dir = tmp_path / "vtk"
    vtk_dir.mkdir(parents=True)
    (vtk_dir / "case_0001.vtk").write_text("dummy")

    existing_case_dir = tmp_path / "case_0001"
    existing_case_dir.mkdir(parents=True)
    torch.save({"ok": True}, existing_case_dir / "data.pt")

    monkeypatch.setattr(
        bw_module,
        "load_case_data",
        lambda _path: {"case_0001": {"CD": 0.12, "CL": 0.01, "M_inf": 0.3}},
    )

    chunk_calls = {"count": 0}

    def _fake_process_chunk(_chunk, _case_lookup, _output_dir, _num_workers):
        chunk_calls["count"] += 1
        return [], []

    monkeypatch.setattr(bw_module, "_process_chunk", _fake_process_chunk)
    monkeypatch.setattr(bw_module, "tqdm", _DummyTqdm)

    bw_module.process_blended_wing(
        vtk_dir=str(vtk_dir),
        case_data_path=str(tmp_path / "case_data.dat"),
        output_dir=str(tmp_path),
        num_workers=1,
    )

    assert chunk_calls["count"] == 0
    assert (tmp_path / "order_.txt").read_text().strip() == "case_0001"
