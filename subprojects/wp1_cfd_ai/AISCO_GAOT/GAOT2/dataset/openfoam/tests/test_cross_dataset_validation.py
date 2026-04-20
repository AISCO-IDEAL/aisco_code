"""Cross-dataset validation: verify OpenFOAM .pt output is compatible with
DrivAer / Ahmed / Euler-Fluent datasets used for GAOT training.

These are integration tests that require the processed data on disk.
They are skipped when the data directories are not available (e.g. CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
#  Data paths
# ---------------------------------------------------------------------------
_DATA_ROOT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai")

_DATASET_CONFIGS: dict[str, dict[str, str | Path]] = {
    "ahmed": {
        "dir": _DATA_ROOT / "processed_ahmed_cp_full",
        "sample": "ahmed_run_1",
        "layout": "subdir",
    },
    "drivaer": {
        "dir": _DATA_ROOT / "processed_drivaer_cp_full",
        "sample": "drivaer_run_1",
        "layout": "subdir",
    },
    "euler_fluent": {
        "dir": _DATA_ROOT / "processed_euler_cp_full",
        "sample": "euler_run_1",
        "layout": "subdir",
    },
    "euler_openfoam": {
        "dir": _DATA_ROOT / "processed_euler_openfoam_cp_v050",
        "sample": "euler_run_1",
        "layout": "subdir",
    },
    "bwb": {
        "dir": _DATA_ROOT / "processed_blended_wing_cp_full",
        "sample": "case_0000.pt",
        "layout": "flat",
    },
}

# Keys that every GAOT-training-ready dataset must contain.
_GAOT_REQUIRED_KEYS = {
    "pos",
    "x",
    "c",
    "face",
    "wall_shear",
    "pressure",
    "force",
    "cd",
    "v_inf",
    "geom_extent",
    "geom_aspect",
    "conditioning",
    "filename",
    "cp",
}

# Keys required for the minimal schema (BWB-style, inference only).
_MINIMAL_KEYS = {"pos", "x", "c", "cd", "force", "v_inf", "geom_extent", "geom_aspect", "conditioning", "filename"}


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------
def _load_sample(name: str) -> Any:
    """Load one .pt sample from a dataset, returning the raw torch.load result."""
    cfg = _DATASET_CONFIGS[name]
    if cfg["layout"] == "flat":
        pt_path = Path(cfg["dir"]) / str(cfg["sample"])
    else:
        pt_path = Path(cfg["dir"]) / str(cfg["sample"]) / "data.pt"
    return torch.load(pt_path, map_location="cpu", weights_only=False)


def _dataset_available(name: str) -> bool:
    cfg = _DATASET_CONFIGS[name]
    if cfg["layout"] == "flat":
        return (Path(cfg["dir"]) / str(cfg["sample"])).exists()
    return (Path(cfg["dir"]) / str(cfg["sample"]) / "data.pt").exists()


_skip_if_missing = pytest.mark.skipif(
    not _DATA_ROOT.exists(),
    reason="Processed datasets not available (not on Euler HPC)",
)


# ---------------------------------------------------------------------------
#  Parametrised datasets
# ---------------------------------------------------------------------------
# Full-schema datasets — used for schema, tensor-shape, and NaN tests only.
# Ahmed: schema/format tests ONLY -- v_inf=1.0 m/s (Mach 0.003), Re 625x lower
# than Euler OF (25 m/s); Cp distributions are not physically comparable.
_FULL_DATASETS = ["ahmed", "drivaer", "euler_fluent", "euler_openfoam"]

# Datasets in the same incompressible regime as Euler OF (25 m/s, Mach 0.073).
# Ahmed excluded (Mach 0.003). BWB filtered separately via _get_bwb_mach_filtered_samples.
# See plan 019 Amendment A1 for the Mach-scoping rationale.
_PHYSICAL_DATASETS = ["drivaer", "euler_openfoam"]

# All datasets including BWB.
_ALL_DATASETS = [*_FULL_DATASETS, "bwb"]

# Maximum v_inf for BWB Mach filtering: 86 m/s ≈ Mach 0.25.
# Above this, compressibility effects are present in BWB that are absent from
# the incompressible simpleFoam solver used for Euler OF.
_BWB_MAX_V_INF = 86.0


def _available_full_datasets() -> list[str]:
    return [d for d in _FULL_DATASETS if _dataset_available(d)]


def _available_all_datasets() -> list[str]:
    return [d for d in _ALL_DATASETS if _dataset_available(d)]


def _available_physical_datasets() -> list[str]:
    """Return Mach-comparable datasets (DrivAer, Euler-OF) that are on disk."""
    return [d for d in _PHYSICAL_DATASETS if _dataset_available(d)]


def _get_bwb_mach_filtered_samples(
    max_vinf: float = _BWB_MAX_V_INF,
    max_n: int = 10,
) -> list[Path]:
    """Return paths of BWB flat .pt files with v_inf <= max_vinf (default 86 m/s).

    BWB spans Mach 0.05-0.5+; only the subset with v_inf <= 86 m/s (Mach <= 0.25)
    is in the same incompressible regime as the 25 m/s Euler OF simulations.
    Scans up to max_n * 10 files to find max_n qualifying cases.
    """
    bwb_dir = Path(_DATASET_CONFIGS["bwb"]["dir"])
    if not bwb_dir.exists():
        return []
    results: list[Path] = []
    for pt_path in sorted(bwb_dir.glob("*.pt"))[: max_n * 10]:
        if len(results) >= max_n:
            break
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if "v_inf" not in data:
                continue
            v_tensor = data["v_inf"]
            v_val = float(v_tensor.item() if hasattr(v_tensor, "item") else v_tensor)
            if v_val <= max_vinf:
                results.append(pt_path)
        except Exception:  # skip any unreadable .pt file
            continue
    return results


# ---------------------------------------------------------------------------
#  Schema compatibility tests
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestSchemaCompatibility:
    """Verify .pt file schemas are consistent across datasets."""

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_full_schema_has_all_gaot_keys(self, ds: str) -> None:
        """Full-schema datasets must contain all GAOT-required keys."""
        data = _load_sample(ds)
        keys = set(data.keys())
        missing = _GAOT_REQUIRED_KEYS - keys
        assert not missing, f"{ds} missing keys: {sorted(missing)}"

    @pytest.mark.parametrize("ds", _available_all_datasets())
    def test_minimal_schema(self, ds: str) -> None:
        """Every dataset must have at least the minimal key set."""
        data = _load_sample(ds)
        keys = set(data.keys())
        missing = _MINIMAL_KEYS - keys
        assert not missing, f"{ds} missing minimal keys: {sorted(missing)}"


# ---------------------------------------------------------------------------
#  Shape / dtype consistency tests
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestTensorShapes:
    """Verify tensor shapes and dtypes match the canonical layout."""

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_pos_shape(self, ds: str) -> None:
        data = _load_sample(ds)
        assert data["pos"].ndim == 2
        assert data["pos"].shape[1] == 3
        assert data["pos"].dtype == torch.float32

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_x_shape_matches_pos(self, ds: str) -> None:
        data = _load_sample(ds)
        n = data["pos"].shape[0]
        assert data["x"].shape == torch.Size([n, 1])
        assert data["x"].dtype == torch.float32

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_normals_shape(self, ds: str) -> None:
        data = _load_sample(ds)
        n = data["pos"].shape[0]
        assert data["c"].shape == torch.Size([n, 3])

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_face_shape(self, ds: str) -> None:
        data = _load_sample(ds)
        assert data["face"].ndim == 2
        assert data["face"].shape[0] == 3, "Faces must be triangulated [3, F]"
        assert data["face"].dtype == torch.int64

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_wall_shear_shape(self, ds: str) -> None:
        data = _load_sample(ds)
        n = data["pos"].shape[0]
        assert data["wall_shear"].shape == torch.Size([n, 3])

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_pressure_shape(self, ds: str) -> None:
        data = _load_sample(ds)
        n = data["pos"].shape[0]
        assert data["pressure"].shape == torch.Size([n, 1])

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_scalar_shapes(self, ds: str) -> None:
        data = _load_sample(ds)
        assert data["cd"].shape == torch.Size([1])
        assert data["v_inf"].shape == torch.Size([1])
        if not np.isnan(data["force"].item()):
            assert data["force"].shape == torch.Size([1])

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_conditioning_shape(self, ds: str) -> None:
        data = _load_sample(ds)
        assert data["conditioning"].shape == torch.Size([1, 7])
        assert data["geom_extent"].shape == torch.Size([3])
        assert data["geom_aspect"].shape == torch.Size([3])

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_cp_equals_x(self, ds: str) -> None:
        """cp and x should contain identical Cp values."""
        data = _load_sample(ds)
        diff = (data["x"] - data["cp"]).abs().max().item()
        assert diff < 1e-6, f"{ds}: x and cp differ by {diff}"


# ---------------------------------------------------------------------------
#  Physical sanity tests
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestPhysicalSanity:
    """Verify physical plausibility of field values."""

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_no_nan_in_cp(self, ds: str) -> None:
        data = _load_sample(ds)
        cp = data["x"].numpy()
        assert not np.isnan(cp).any(), f"{ds}: {np.isnan(cp).sum()} NaN in Cp"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_no_inf_in_cp(self, ds: str) -> None:
        data = _load_sample(ds)
        cp = data["x"].numpy()
        assert not np.isinf(cp).any(), f"{ds}: {np.isinf(cp).sum()} Inf in Cp"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_no_nan_in_pos(self, ds: str) -> None:
        data = _load_sample(ds)
        pos = data["pos"].numpy()
        assert not np.isnan(pos).any(), f"{ds}: NaN in positions"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_no_nan_in_normals(self, ds: str) -> None:
        data = _load_sample(ds)
        c = data["c"].numpy()
        assert not np.isnan(c).any(), f"{ds}: NaN in normals"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_cp_has_stagnation_point(self, ds: str) -> None:
        """Automotive/aero surfaces should have Cp near 1.0 at stagnation."""
        data = _load_sample(ds)
        cp_max = data["x"].max().item()
        # Relaxed check: Cp_max should be at least 0.5 (some datasets slightly below 1.0)
        assert cp_max > 0.5, f"{ds}: Cp_max={cp_max:.3f} — no stagnation point detected"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_cp_mean_is_negative(self, ds: str) -> None:
        """For bluff bodies, mean Cp is typically negative (suction-dominated)."""
        data = _load_sample(ds)
        cp_mean = data["x"].mean().item()
        assert cp_mean < 0, f"{ds}: mean Cp={cp_mean:.3f} — expected negative"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_face_indices_in_bounds(self, ds: str) -> None:
        """Face vertex indices must be within [0, N-1]."""
        data = _load_sample(ds)
        n = data["pos"].shape[0]
        faces = data["face"]
        assert faces.min().item() >= 0, f"{ds}: negative face index"
        assert faces.max().item() < n, f"{ds}: face index {faces.max()} >= n_pts {n}"

    @pytest.mark.parametrize("ds", _available_full_datasets())
    def test_wall_shear_no_nan(self, ds: str) -> None:
        data = _load_sample(ds)
        ws = data["wall_shear"].numpy()
        assert not np.isnan(ws).any(), f"{ds}: NaN in wall_shear"


# ---------------------------------------------------------------------------
#  OpenFOAM-specific compatibility tests
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestOpenfoamVsEstablished:
    """Verify OpenFOAM Euler data is compatible with the established
    DrivAer/Ahmed datasets that are known to work for GAOT training."""

    def _skip_if_of_missing(self) -> None:
        if not _dataset_available("euler_openfoam"):
            pytest.skip("euler_openfoam data not available")

    def test_of_schema_superset_of_ahmed(self) -> None:
        """OpenFOAM .pt must have at least all keys Ahmed has."""
        self._skip_if_of_missing()
        if not _dataset_available("ahmed"):
            pytest.skip("ahmed data not available")
        ahmed = _load_sample("ahmed")
        of_data = _load_sample("euler_openfoam")
        ahmed_keys = set(ahmed.keys())
        of_keys = set(of_data.keys())
        missing = ahmed_keys - of_keys
        assert not missing, f"OF missing keys present in Ahmed: {sorted(missing)}"

    def test_of_schema_superset_of_drivaer(self) -> None:
        """OpenFOAM .pt must have at least all keys DrivAer has."""
        self._skip_if_of_missing()
        if not _dataset_available("drivaer"):
            pytest.skip("drivaer data not available")
        drivaer = _load_sample("drivaer")
        of_data = _load_sample("euler_openfoam")
        drivaer_keys = set(drivaer.keys())
        of_keys = set(of_data.keys())
        missing = drivaer_keys - of_keys
        assert not missing, f"OF missing keys present in DrivAer: {sorted(missing)}"

    def test_of_cd_positive(self) -> None:
        """OpenFOAM Cd must be positive (drag is always positive for vehicles)."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        cd = data["cd"].item()
        assert cd > 0, f"OF Cd={cd:.4f} should be positive"

    def test_of_cd_in_physical_range(self) -> None:
        """OpenFOAM Cd should be in [0.01, 2.0] for vehicle shapes."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        cd = data["cd"].item()
        assert 0.01 < cd < 2.0, f"OF Cd={cd:.4f} outside plausible range"

    def test_of_v_inf_matches_euler(self) -> None:
        """OpenFOAM Euler v_inf should be 25.0 m/s."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        v = data["v_inf"].item()
        assert abs(v - 25.0) < 0.1, f"OF v_inf={v} expected 25.0"

    def test_of_conditioning_first_element_is_v_inf(self) -> None:
        """conditioning[0,0] should equal v_inf."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        v = data["v_inf"].item()
        cond_v = data["conditioning"][0, 0].item()
        assert abs(cond_v - v) < 0.01, f"conditioning[0,0]={cond_v} != v_inf={v}"


# ---------------------------------------------------------------------------
#  Known data quality issues (format / pipeline bugs, not physics)
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestKnownIssues:
    """Document known data quality issues across datasets."""

    def test_bwb_missing_full_schema(self) -> None:
        """KNOWN ISSUE: BWB data uses a minimal schema without face/pressure/wall_shear/cp."""
        if not _dataset_available("bwb"):
            pytest.skip("bwb data not available")
        data = _load_sample("bwb")
        keys = set(data.keys())
        expected_missing = {"face", "pressure", "wall_shear", "cp"}
        actual_missing = expected_missing - keys
        # If they're still missing, document it; if fixed, update test
        assert actual_missing == expected_missing, (
            f"BWB schema changed — previously missing {expected_missing}, now only missing {actual_missing}. Update this test."
        )


# ---------------------------------------------------------------------------
#  Multi-sample statistical tests (spot-check a few runs)
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestMultiSampleConsistency:
    """Spot-check a few samples per dataset for consistent statistics."""

    @pytest.mark.parametrize(
        "ds,n_check",
        [
            ("ahmed", 5),
            ("drivaer", 5),
        ],
    )
    def test_cd_all_positive(self, ds: str, n_check: int) -> None:
        """Cd should be positive for all checked samples (Ahmed/DrivAer)."""
        if not _dataset_available(ds):
            pytest.skip(f"{ds} data not available")
        ds_dir = Path(_DATASET_CONFIGS[ds]["dir"])
        checked = 0
        for d in sorted(ds_dir.iterdir()):
            pt = d / "data.pt"
            if not pt.exists():
                continue
            data = torch.load(pt, map_location="cpu", weights_only=False)
            cd = data["cd"].item()
            assert cd > 0, f"{ds}/{d.name}: Cd={cd:.4f} should be positive"
            checked += 1
            if checked >= n_check:
                break
        assert checked > 0, f"No samples found for {ds}"


# ---------------------------------------------------------------------------
#  OpenFOAM new-field regression (M4 deliverables)
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestNewOpenfoamFields:
    """Verify M4-introduced convergence fields are present in re-extracted OF .pt files.

    These tests will FAIL until M5 re-extraction runs process_openfoam_pressure.py
    on all 25 cases (expected TDD behaviour).  Once M5 completes they must pass.
    """

    def _skip_if_of_missing(self) -> None:
        if not _dataset_available("euler_openfoam"):
            pytest.skip("euler_openfoam data not available")

    def test_cd_mean50_present(self) -> None:
        """euler_openfoam .pt must contain the cd_mean50 field (added in M4)."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        assert "cd_mean50" in data, "cd_mean50 missing — run M5 re-extraction"

    def test_cd_std50_present(self) -> None:
        """euler_openfoam .pt must contain the cd_std50 field (added in M4)."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        assert "cd_std50" in data, "cd_std50 missing — run M5 re-extraction"

    def test_cd_std50_non_negative(self) -> None:
        """cd_std50 is a standard deviation — must be >= 0."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        if "cd_std50" not in data:
            pytest.skip("cd_std50 not yet in .pt (pre-M5)")
        val = data["cd_std50"].item()
        assert val >= 0.0, f"cd_std50={val:.6f} is negative"

    def test_cd_mean50_in_physical_range(self) -> None:
        """cd_mean50 must be in (0.01, 2.0) — physically plausible drag coefficient."""
        self._skip_if_of_missing()
        data = _load_sample("euler_openfoam")
        if "cd_mean50" not in data:
            pytest.skip("cd_mean50 not yet in .pt (pre-M5)")
        val = data["cd_mean50"].item()
        assert 0.01 < val < 2.0, f"cd_mean50={val:.4f} outside plausible range (0.01, 2.0)"


# ---------------------------------------------------------------------------
#  Mach-compatible physical distribution tests
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestMachCompatibleDistributions:
    """Physical plausibility checks restricted to Mach-comparable datasets.

    Ahmed is excluded (v_inf=1.0 m/s, Mach 0.003; Re 625x lower than OF).
    BWB is included only for cases with v_inf <= 86 m/s (Mach <= 0.25).
    See plan 019 Amendment A1 for the full Mach-scoping rationale.
    """

    @pytest.mark.parametrize("ds", _available_physical_datasets())
    def test_physical_datasets_cp_no_nan(self, ds: str) -> None:
        """No NaN in Cp for Mach-comparable datasets (DrivAer, Euler-OF)."""
        data = _load_sample(ds)
        cp = data["x"].numpy()
        assert not np.isnan(cp).any(), f"{ds}: {np.isnan(cp).sum()} NaN in Cp"

    @pytest.mark.parametrize("ds", _available_physical_datasets())
    def test_physical_datasets_cp_has_stagnation(self, ds: str) -> None:
        """Mach-comparable datasets must have Cp_max > 0.5 (stagnation point present)."""
        data = _load_sample(ds)
        cp_max = data["x"].max().item()
        assert cp_max > 0.5, f"{ds}: Cp_max={cp_max:.3f} — no stagnation point detected"

    @pytest.mark.parametrize("ds", _available_physical_datasets())
    def test_physical_datasets_cp_mean_negative(self, ds: str) -> None:
        """Mean Cp should be negative for suction-dominated automotive aero bodies."""
        data = _load_sample(ds)
        cp_mean = data["x"].mean().item()
        assert cp_mean < 0, f"{ds}: mean Cp={cp_mean:.3f} — expected negative"

    def test_bwb_mach_filter_returns_samples(self) -> None:
        """BWB Mach filter must find at least one case with v_inf <= 86 m/s."""
        bwb_dir = Path(_DATASET_CONFIGS["bwb"]["dir"])
        if not bwb_dir.exists():
            pytest.skip("BWB data not available")
        samples = _get_bwb_mach_filtered_samples(max_n=5)
        assert len(samples) >= 1, f"No BWB cases found with v_inf <= {_BWB_MAX_V_INF} m/s"

    def test_bwb_filtered_cp_no_nan(self) -> None:
        """No NaN in Cp for the first Mach-filtered BWB sample."""
        bwb_dir = Path(_DATASET_CONFIGS["bwb"]["dir"])
        if not bwb_dir.exists():
            pytest.skip("BWB data not available")
        samples = _get_bwb_mach_filtered_samples(max_n=1)
        if not samples:
            pytest.skip("No BWB Mach-filtered samples found")
        data = torch.load(samples[0], map_location="cpu", weights_only=False)
        cp = data["x"].numpy()
        assert not np.isnan(cp).any(), f"BWB filtered: {np.isnan(cp).sum()} NaN in Cp"

    def test_bwb_filtered_cp_has_stagnation(self) -> None:
        """Mach-filtered BWB sample must have Cp_max > 0.5 (stagnation point present)."""
        bwb_dir = Path(_DATASET_CONFIGS["bwb"]["dir"])
        if not bwb_dir.exists():
            pytest.skip("BWB data not available")
        samples = _get_bwb_mach_filtered_samples(max_n=1)
        if not samples:
            pytest.skip("No BWB Mach-filtered samples found")
        data = torch.load(samples[0], map_location="cpu", weights_only=False)
        cp_max = data["x"].max().item()
        assert cp_max > 0.5, f"BWB filtered: Cp_max={cp_max:.3f} — no stagnation point"

    def test_of_and_drivaer_cp_quantiles_compatible(self) -> None:
        """Cp quantile distributions of OF and DrivAer must be loosely compatible.

        Both are incompressible automotive flows (Mach 0.073 and 0.113).
        Quantile-MAE < 1.0 catches sign errors and unit errors without requiring
        tight agreement (geometries differ so exact match is not expected).
        """
        if not _dataset_available("euler_openfoam"):
            pytest.skip("euler_openfoam data not available")
        if not _dataset_available("drivaer"):
            pytest.skip("drivaer data not available")
        of_cp = _load_sample("euler_openfoam")["x"].numpy().flatten()
        driv_cp = _load_sample("drivaer")["x"].numpy().flatten()
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        of_q = np.quantile(of_cp, quantiles)
        driv_q = np.quantile(driv_cp, quantiles)
        mae = float(np.abs(of_q - driv_q).mean())
        assert mae < 1.0, (
            f"OF vs DrivAer quantile-MAE={mae:.4f} >= 1.0 — possible unit or sign error. "
            f"OF Cp range=[{of_cp.min():.3f},{of_cp.max():.3f}], "
            f"DrivAer Cp range=[{driv_cp.min():.3f},{driv_cp.max():.3f}]"
        )


# ---------------------------------------------------------------------------
#  OpenFOAM vs Fluent quantitative comparison (matched runs)
# ---------------------------------------------------------------------------
@_skip_if_missing
class TestOpenfoamVsFluent:
    """Quantitative comparison between OpenFOAM and Euler-Fluent reference data.

    Uses matched runs (same geometry, same boundary conditions).
    Meshes differ in node count, so comparison uses Cp quantile distributions.
    Run 1 is confirmed Reliable from the M2 audit (plan 019).
    """

    def _skip_if_either_missing(self) -> None:
        if not _dataset_available("euler_openfoam"):
            pytest.skip("euler_openfoam data not available")
        if not _dataset_available("euler_fluent"):
            pytest.skip("euler_fluent data not available")

    def test_of_run1_vs_fluent_cp_quantile_mae(self) -> None:
        """OF run 1 Cp quantile distribution must match Fluent run 1 within MAE < 0.6.

        Different mesh topologies (OF ~137k pts, Fluent ~152k pts) prevent per-vertex
        comparison.  7-quantile MAE < 0.6 catches major distribution shifts while
        allowing for RANS solver and mesh resolution differences.
        """
        self._skip_if_either_missing()
        of_cp = _load_sample("euler_openfoam")["x"].numpy().flatten()
        fl_cp = _load_sample("euler_fluent")["x"].numpy().flatten()
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        of_q = np.quantile(of_cp, quantiles)
        fl_q = np.quantile(fl_cp, quantiles)
        mae = float(np.abs(of_q - fl_q).mean())
        assert mae < 0.6, (
            f"OF run1 vs Fluent run1 quantile-MAE={mae:.4f} >= 0.6 — "
            f"OF Cp range=[{of_cp.min():.3f},{of_cp.max():.3f}], "
            f"Fluent Cp range=[{fl_cp.min():.3f},{fl_cp.max():.3f}]"
        )

    def test_fluent_cd_is_dimensional_force_not_coefficient(self) -> None:
        """KNOWN DATA ISSUE: Euler-Fluent cd field stores dimensional drag force (N),
        not the dimensionless drag coefficient.

        OF:     cd approx 0.05-0.5  (dimensionless, from OpenFOAM coefficient.dat)
        Fluent: cd approx -50 to -300 N  (dimensional force, stored with negative sign)

        At 25 m/s for typical solar car geometries this gives:
          OF run 1:     Cd approx 0.10  (dimensionless coefficient)
          Fluent run 1: cd approx -250 N  (implied Cd approx 0.3-0.5 at Q_inf=382.8 Pa)

        Do NOT compare these directly.  The Fluent pipeline must be re-processed
        to store normalised Cd = |force| / (Q_inf * A_ref) before using the cd
        field for model evaluation or side-by-side comparisons with OF.
        """
        self._skip_if_either_missing()
        of_cd = _load_sample("euler_openfoam")["cd"].item()
        fl_cd = _load_sample("euler_fluent")["cd"].item()
        # OF Cd: small positive dimensionless coefficient
        assert 0.01 < of_cd < 2.0, f"OF Cd={of_cd:.4f} not in expected dimensionless range"
        # Fluent cd: large negative dimensional force (Newtons)
        assert fl_cd < -10.0, (
            f"Fluent cd={fl_cd:.4f} — expected a large negative force in Newtons. "
            "If the Fluent pipeline was re-processed to store normalised Cd, update this test."
        )

    def test_fluent_cd_sign_negative_known_issue(self) -> None:
        """KNOWN PIPELINE DATA ISSUE: Euler-Fluent Cd is stored as a negative number.

        This is a sign convention inconsistency in the upstream Fluent export.
        The dragDir=(-1 0 0) convention means raw drag force is negative in x,
        and the Fluent pipeline stored this raw value instead of |Cd|.
        DrivAer, Ahmed, and OpenFOAM all store Cd as positive.

        When the upstream Fluent .pt files are regenerated with correct sign,
        update this test to assert cd > 0.
        """
        if not _dataset_available("euler_fluent"):
            pytest.skip("euler_fluent data not available")
        data = _load_sample("euler_fluent")
        cd = data["cd"].item()
        assert cd < 0, (
            f"Euler-Fluent Cd={cd:.4f} is now positive — sign convention may have been fixed. Update this test to assert cd > 0."
        )

    def test_openfoam_multi_sample_no_nan_cp(self) -> None:
        """Spot-check 5 OpenFOAM samples for NaN in Cp."""
        if not _dataset_available("euler_openfoam"):
            pytest.skip("euler_openfoam data not available")
        ds_dir = Path(_DATASET_CONFIGS["euler_openfoam"]["dir"])
        checked = 0
        for d in sorted(ds_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            pt = d / "data.pt"
            if not pt.exists():
                continue
            data = torch.load(pt, map_location="cpu", weights_only=False)
            cp = data["x"].numpy()
            assert not np.isnan(cp).any(), f"euler_openfoam/{d.name}: NaN in Cp"
            assert not np.isinf(cp).any(), f"euler_openfoam/{d.name}: Inf in Cp"
            checked += 1
            if checked >= 5:
                break
        assert checked > 0, "No OpenFOAM samples found"

    def test_openfoam_multi_sample_cd_positive(self) -> None:
        """Spot-check all OpenFOAM samples for positive Cd."""
        if not _dataset_available("euler_openfoam"):
            pytest.skip("euler_openfoam data not available")
        ds_dir = Path(_DATASET_CONFIGS["euler_openfoam"]["dir"])
        checked = 0
        for d in sorted(ds_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue
            pt = d / "data.pt"
            if not pt.exists():
                continue
            data = torch.load(pt, map_location="cpu", weights_only=False)
            cd = data["cd"].item()
            assert cd > 0, f"euler_openfoam/{d.name}: Cd={cd:.4f} should be positive"
            checked += 1
        assert checked > 0, "No OpenFOAM samples found"
