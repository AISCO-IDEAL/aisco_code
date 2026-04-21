"""Tests for convergence_qa.py — Convergence quality analysis for .pt files.

TDD: These tests are written BEFORE convergence_qa.py exists.
Expected initial failure: ModuleNotFoundError.
"""

import json
from pathlib import Path

import pytest
import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.convergence_qa import (
    Grade,
    Severity,
    classify_run,
    generate_report,
    generate_training_ready,
    run_qa,
)

# ---------------------------------------------------------------------------
#  Fixtures — synthetic .pt data
# ---------------------------------------------------------------------------


def _make_pt_data(
    *,
    final_residuals: list[float] | None = None,
    cd: float = 0.3,
    cd_mean50: float = 0.3,
    cd_std50: float = 0.0005,
    cp_min: float = -3.0,
    cp_max: float = 1.0,
    n_points: int = 1000,
    zero_pressure_frac: float = 0.0,
) -> dict:
    """Create a synthetic .pt-like dict with the fields convergence_qa reads."""
    if final_residuals is None:
        # All residuals well converged: [p, Ux, k, omega]
        final_residuals = [1e-5, 1e-4, 1e-4, 1e-5]

    # Create a pressure field with controlled range
    pressure = torch.linspace(cp_min, cp_max, n_points).unsqueeze(1)
    if zero_pressure_frac > 0:
        n_zeros = int(n_points * zero_pressure_frac)
        pressure[:n_zeros] = 0.0

    return {
        "x": pressure,  # Cp field [N, 1]
        "cd": torch.tensor([cd]),
        "cd_mean50": torch.tensor([cd_mean50]),
        "cd_std50": torch.tensor([cd_std50]),
        "final_residuals": torch.tensor(final_residuals),
        "pos": torch.randn(n_points, 3),
        "face": torch.zeros(3, 0, dtype=torch.long),
    }


@pytest.fixture
def passing_pt(tmp_path: Path) -> Path:
    """A .pt file that should PASS all convergence tests."""
    data = _make_pt_data()
    pt_path = tmp_path / "euler_run_1" / "data.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(data, pt_path)
    return tmp_path


@pytest.fixture
def failing_cd_oscillation(tmp_path: Path) -> Path:
    """A .pt file that should FAIL T5 (Cd CoV > 0.5%)."""
    data = _make_pt_data(cd_std50=0.01, cd_mean50=0.3)  # CoV = 3.3%
    pt_path = tmp_path / "euler_run_1" / "data.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(data, pt_path)
    return tmp_path


@pytest.fixture
def warn_residual(tmp_path: Path) -> Path:
    """A .pt file that should WARN (high Ux residual but no CRITICAL failures)."""
    data = _make_pt_data(final_residuals=[1e-5, 5e-3, 5e-3, 1e-5])
    pt_path = tmp_path / "euler_run_1" / "data.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(data, pt_path)
    return tmp_path


@pytest.fixture
def negative_cd(tmp_path: Path) -> Path:
    """A .pt file that should FAIL T6 (Cd < 0)."""
    data = _make_pt_data(cd=-0.1)
    pt_path = tmp_path / "euler_run_1" / "data.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(data, pt_path)
    return tmp_path


@pytest.fixture
def extreme_cp(tmp_path: Path) -> Path:
    """A .pt file that should FAIL T7 (Cp out of range)."""
    data = _make_pt_data(cp_min=-15.0, cp_max=1.0)
    pt_path = tmp_path / "euler_run_1" / "data.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(data, pt_path)
    return tmp_path


@pytest.fixture
def high_zero_pressure(tmp_path: Path) -> Path:
    """A .pt file that should FAIL T8 (>1% zero pressure)."""
    data = _make_pt_data(zero_pressure_frac=0.05)
    pt_path = tmp_path / "euler_run_1" / "data.pt"
    pt_path.parent.mkdir(parents=True)
    torch.save(data, pt_path)
    return tmp_path


# ---------------------------------------------------------------------------
#  Tests — classify_run
# ---------------------------------------------------------------------------


class TestClassifyRun:
    """Test the per-run convergence classification logic."""

    def test_passing_run(self, passing_pt: Path) -> None:
        """All tests pass → Grade.PASS."""
        result = classify_run(passing_pt / "euler_run_1" / "data.pt")
        assert result.grade == Grade.PASS
        assert all(t.passed for t in result.test_results)

    def test_cd_oscillation_fails(self, failing_cd_oscillation: Path) -> None:
        """T5 (Cd CoV) > 0.5% → Grade.FAIL (CRITICAL severity)."""
        result = classify_run(failing_cd_oscillation / "euler_run_1" / "data.pt")
        assert result.grade == Grade.FAIL
        t5 = next(t for t in result.test_results if t.test_id == "T5")
        assert not t5.passed
        assert t5.severity == Severity.CRITICAL

    def test_warn_on_major_residual(self, warn_residual: Path) -> None:
        """High Ux/k residuals (MAJOR) but no CRITICAL → Grade.WARN."""
        result = classify_run(warn_residual / "euler_run_1" / "data.pt")
        assert result.grade == Grade.WARN

    def test_negative_cd_fails(self, negative_cd: Path) -> None:
        """Cd < 0 → Grade.FAIL (T6 CRITICAL)."""
        result = classify_run(negative_cd / "euler_run_1" / "data.pt")
        assert result.grade == Grade.FAIL
        t6 = next(t for t in result.test_results if t.test_id == "T6")
        assert not t6.passed

    def test_extreme_cp_fails(self, extreme_cp: Path) -> None:
        """Cp out of [-10, 5] → Grade.FAIL (T7 MAJOR... actually this is MAJOR not CRITICAL)."""
        result = classify_run(extreme_cp / "euler_run_1" / "data.pt")
        # T7 is MAJOR severity, so it should be WARN not FAIL (no CRITICAL failure)
        assert result.grade == Grade.WARN

    def test_zero_pressure_fails(self, high_zero_pressure: Path) -> None:
        """Zero pressure > 1% → Grade.FAIL (T8 CRITICAL)."""
        result = classify_run(high_zero_pressure / "euler_run_1" / "data.pt")
        assert result.grade == Grade.FAIL
        t8 = next(t for t in result.test_results if t.test_id == "T8")
        assert not t8.passed


# ---------------------------------------------------------------------------
#  Tests — run_qa (batch)
# ---------------------------------------------------------------------------


class TestRunQA:
    """Test the batch QA runner."""

    def test_processes_all_runs(self, passing_pt: Path) -> None:
        """run_qa should process all euler_run_* directories."""
        results = run_qa(passing_pt)
        assert len(results) == 1
        assert results[0].grade == Grade.PASS

    def test_multiple_runs(self, tmp_path: Path) -> None:
        """run_qa processes multiple runs."""
        for i in range(3):
            data = _make_pt_data()
            pt_path = tmp_path / f"euler_run_{i + 1}" / "data.pt"
            pt_path.parent.mkdir(parents=True)
            torch.save(data, pt_path)

        results = run_qa(tmp_path)
        assert len(results) == 3
        assert all(r.grade == Grade.PASS for r in results)

    def test_run_ids_filter(self, tmp_path: Path) -> None:
        """run_qa with --run-ids only processes specified runs."""
        for i in range(5):
            data = _make_pt_data()
            pt_path = tmp_path / f"euler_run_{i + 1}" / "data.pt"
            pt_path.parent.mkdir(parents=True)
            torch.save(data, pt_path)

        results = run_qa(tmp_path, run_ids=[1, 3])
        assert len(results) == 2


# ---------------------------------------------------------------------------
#  Tests — output generation
# ---------------------------------------------------------------------------


class TestOutputGeneration:
    """Test report and training_ready output generation."""

    def test_generate_report_json(self, passing_pt: Path, tmp_path: Path) -> None:
        """generate_report writes valid JSON with expected structure."""
        results = run_qa(passing_pt)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generate_report(results, output_dir)

        report_path = output_dir / "convergence_report.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert "runs" in report
        assert len(report["runs"]) == 1

    def test_generate_training_ready(self, tmp_path: Path) -> None:
        """training_ready.txt contains PASS + WARN run IDs, excludes FAIL."""
        # Create PASS, WARN, and FAIL runs
        # PASS run
        data_pass = _make_pt_data()
        (tmp_path / "euler_run_1").mkdir()
        torch.save(data_pass, tmp_path / "euler_run_1" / "data.pt")

        # WARN run (high major residual, no critical)
        data_warn = _make_pt_data(final_residuals=[1e-5, 5e-3, 5e-3, 1e-5])
        (tmp_path / "euler_run_2").mkdir()
        torch.save(data_warn, tmp_path / "euler_run_2" / "data.pt")

        # FAIL run (negative Cd)
        data_fail = _make_pt_data(cd=-0.1)
        (tmp_path / "euler_run_3").mkdir()
        torch.save(data_fail, tmp_path / "euler_run_3" / "data.pt")

        results = run_qa(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generate_training_ready(results, output_dir, strict=False)

        ready_path = output_dir / "training_ready.txt"
        assert ready_path.exists()
        ready_ids = ready_path.read_text().strip().split("\n")
        # PASS + WARN should be included, FAIL excluded
        assert len(ready_ids) == 2

    def test_strict_mode_excludes_warn(self, tmp_path: Path) -> None:
        """With strict=True, only PASS runs are in training_ready.txt."""
        # PASS run
        data_pass = _make_pt_data()
        (tmp_path / "euler_run_1").mkdir()
        torch.save(data_pass, tmp_path / "euler_run_1" / "data.pt")

        # WARN run
        data_warn = _make_pt_data(final_residuals=[1e-5, 5e-3, 5e-3, 1e-5])
        (tmp_path / "euler_run_2").mkdir()
        torch.save(data_warn, tmp_path / "euler_run_2" / "data.pt")

        results = run_qa(tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generate_training_ready(results, output_dir, strict=True)

        ready_path = output_dir / "training_ready.txt"
        ready_ids = ready_path.read_text().strip().split("\n")
        assert len(ready_ids) == 1
