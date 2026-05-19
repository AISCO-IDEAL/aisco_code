"""TDD test: canonical v0.7.0 metadata fields are written by process_single_run.

Tests that after adding source_dataset support, the saved .pt file contains all
6 canonical fields required by the v0.7.0 preprocessing contract:
  - source_dataset
  - source_case_id
  - preprocess_version
  - flow_conditions
  - geom_name
  - n_surface_faces
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.process_openfoam_pressure import (
    process_single_run,
)

# ---------------------------------------------------------------------------
#  Minimal synthetic geometry for mocking VTK output
# ---------------------------------------------------------------------------

_N_VERTS = 6
_N_FACES = 4

_FAKE_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
        [0.5, -0.5, 0.5],
    ],
    dtype=np.float32,
)
_FAKE_FACES = np.array(
    [[0, 1, 2], [1, 3, 2], [0, 1, 4], [2, 3, 4]],
    dtype=np.int64,
)
_FAKE_PRESSURE = np.zeros(_N_VERTS, dtype=np.float32) + 50.0
_FAKE_WALL_SHEAR = np.zeros((_N_VERTS, 3), dtype=np.float32)

_FAKE_PATCH_DATA = {
    "vertices": _FAKE_VERTICES,
    "faces": _FAKE_FACES,
    "pressure": _FAKE_PRESSURE,
    "wall_shear": _FAKE_WALL_SHEAR,
}

_FAKE_CONVERGENCE = {
    "converged": True,
    "final_residuals": {"p": 1e-5, "Ux": 1e-5, "k": 1e-5, "omega": 1e-5},
}

_FAKE_STL_QUALITY = {
    "n_stl_pts": 1000,
    "stl_median_edge_mm": 5.0,
    "distinct_normal_dirs": 100,
}

_FAKE_PRESSURE_VALIDITY = {
    "p_min": -50.0,
    "p_max": 50.0,
    "p_zero_frac": 0.0,
}


@pytest.fixture()
def fake_case_dir(tmp_path: Path) -> Path:
    """Create a minimal case directory structure that satisfies existence checks."""
    case_dir = tmp_path / "RUN_001"
    case_dir.mkdir()
    return tmp_path


def _mock_extract_patch_vtk(_case_dir: Path, _time_step: int, _patch_name: str = "car_surface") -> dict:
    return _FAKE_PATCH_DATA


def _mock_parse_openfoam_forces(_case_dir: Path) -> tuple[float | None, float | None]:
    return None, 12.5  # (coeff, force_N)


def _mock_parse_cd_stats(_case_dir: Path) -> tuple[float | None, float | None]:
    return 0.3, 0.001


def _mock_parse_convergence_status(_case_dir: Path) -> dict:
    return _FAKE_CONVERGENCE


def _mock_compute_stl_quality(_case_dir: Path) -> dict:
    return _FAKE_STL_QUALITY


def _mock_compute_pressure_validity(_pressure: np.ndarray) -> dict:
    return _FAKE_PRESSURE_VALIDITY


def _mock_parse_coeff_aref(_case_dir: Path) -> float | None:
    return None


@pytest.mark.parametrize("source_dataset", ["openfoam_federico_021", "test_batch"])
def test_canonical_metadata_fields_present(
    fake_case_dir: Path,
    tmp_path: Path,
    source_dataset: str,
) -> None:
    """process_single_run must write all 6 canonical v0.7.0 metadata fields."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    extractor_module = "subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.process_openfoam_pressure"

    with (
        patch(f"{extractor_module}.extract_patch_vtk", side_effect=_mock_extract_patch_vtk),
        patch(f"{extractor_module}.parse_openfoam_forces", side_effect=_mock_parse_openfoam_forces),
        patch(f"{extractor_module}.parse_cd_stats", side_effect=_mock_parse_cd_stats),
        patch(f"{extractor_module}.parse_convergence_status", side_effect=_mock_parse_convergence_status),
        patch(f"{extractor_module}._compute_stl_quality_metrics", side_effect=_mock_compute_stl_quality),
        patch(f"{extractor_module}._compute_pressure_validity", side_effect=_mock_compute_pressure_validity),
        patch(f"{extractor_module}._parse_coeff_aref", side_effect=_mock_parse_coeff_aref),
    ):
        result = process_single_run(
            run_id=1,
            cases_dir=str(fake_case_dir),
            output_dir=str(output_dir),
            source_dataset=source_dataset,
        )

    assert result is not None, "process_single_run returned None (extraction failed)"

    pt_path = output_dir / result / "data.pt"
    assert pt_path.exists(), f"Expected .pt file at {pt_path}"

    data = torch.load(pt_path, weights_only=False)

    # --- 6 canonical fields ---
    assert hasattr(data, "source_dataset"), "data.source_dataset missing"
    assert data.source_dataset == source_dataset, f"Expected source_dataset={source_dataset!r}, got {data.source_dataset!r}"

    assert hasattr(data, "source_case_id"), "data.source_case_id missing"
    assert data.source_case_id == f"{source_dataset}::RUN_001", (
        f"Expected source_case_id={source_dataset}::RUN_001, got {data.source_case_id!r}"
    )

    assert hasattr(data, "preprocess_version"), "data.preprocess_version missing"
    assert data.preprocess_version == "v0.7.0", f"Expected preprocess_version='v0.7.0', got {data.preprocess_version!r}"

    assert hasattr(data, "flow_conditions"), "data.flow_conditions missing"
    assert data.flow_conditions == {}, f"Expected flow_conditions={{}}, got {data.flow_conditions!r}"

    assert hasattr(data, "geom_name"), "data.geom_name missing"
    assert isinstance(data.geom_name, str), f"geom_name must be str, got {type(data.geom_name)}"

    assert hasattr(data, "n_surface_faces"), "data.n_surface_faces missing"
    assert isinstance(data.n_surface_faces, int), f"n_surface_faces must be int, got {type(data.n_surface_faces)}"
    assert data.n_surface_faces == _N_FACES, f"Expected n_surface_faces={_N_FACES}, got {data.n_surface_faces}"


def test_source_dataset_defaults_to_openfoam_federico(
    fake_case_dir: Path,
    tmp_path: Path,
) -> None:
    """Default source_dataset must be 'openfoam_federico' for backward compat."""
    output_dir = tmp_path / "out_default"
    output_dir.mkdir()

    extractor_module = "subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.process_openfoam_pressure"

    with (
        patch(f"{extractor_module}.extract_patch_vtk", side_effect=_mock_extract_patch_vtk),
        patch(f"{extractor_module}.parse_openfoam_forces", side_effect=_mock_parse_openfoam_forces),
        patch(f"{extractor_module}.parse_cd_stats", side_effect=_mock_parse_cd_stats),
        patch(f"{extractor_module}.parse_convergence_status", side_effect=_mock_parse_convergence_status),
        patch(f"{extractor_module}._compute_stl_quality_metrics", side_effect=_mock_compute_stl_quality),
        patch(f"{extractor_module}._compute_pressure_validity", side_effect=_mock_compute_pressure_validity),
        patch(f"{extractor_module}._parse_coeff_aref", side_effect=_mock_parse_coeff_aref),
    ):
        result = process_single_run(
            run_id=1,
            cases_dir=str(fake_case_dir),
            output_dir=str(output_dir),
            # source_dataset not provided → use default
        )

    assert result is not None
    data = torch.load(output_dir / result / "data.pt", weights_only=False)
    assert data.source_dataset == "openfoam_federico"
