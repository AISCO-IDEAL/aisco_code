#!/usr/bin/env python3
"""Convergence quality analysis for OpenFOAM .pt dataset files.

Classifies each run as PASS/WARN/FAIL based on 8 convergence tests
(T1-T8) covering residual convergence, Cd stability, Cp physicality,
and pressure validity.

Output artifacts:
  - convergence_report.json : per-run, per-test results
  - convergence_summary.txt : human-readable grade counts
  - training_ready.txt      : run IDs suitable for training

Usage:
    python convergence_qa.py --dataset-dir /path/to/processed_*
    python convergence_qa.py --dataset-dir /path/to/processed_* --run-ids 1 3 10
    python convergence_qa.py --dataset-dir /path/to/processed_* --strict
    python convergence_qa.py --dataset-dir /path/to/processed_* --verbose
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Default dataset directory
# ---------------------------------------------------------------------------
_DEFAULT_DATASET_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_openfoam_cp_v050")


# ---------------------------------------------------------------------------
#  Enums and data classes
# ---------------------------------------------------------------------------


class Grade(enum.Enum):
    """Overall quality grade for a run."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class Severity(enum.Enum):
    """Test severity level."""

    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


@dataclass
class TestResult:
    """Result of a single convergence test on one run."""

    test_id: str
    name: str
    severity: Severity
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class RunResult:
    """Aggregated QA result for a single run."""

    run_id: str
    pt_path: str
    grade: Grade
    test_results: list[TestResult]


# ---------------------------------------------------------------------------
#  Convergence tests (T1-T8)
# ---------------------------------------------------------------------------

# Default thresholds per architecture doc 047
_THRESHOLDS = {
    "T1_p_residual": 1e-4,
    "T2_ux_residual": 1e-3,
    "T3_k_residual": 1e-3,
    "T4_omega_residual": 1e-4,
    "T5_cd_cov": 0.05,  # 5.0% — runs with CoV >= 5% are excluded
    "T6_cd_positive": 0.0,
    "T7_cp_min": -10.0,
    "T7_cp_max": 5.0,
    "T8_zero_pressure_frac": 0.01,  # 1%
}

# Strict thresholds (--strict flag)
_STRICT_THRESHOLDS = {
    "T1_p_residual": 1e-4,
    "T2_ux_residual": 1e-4,
    "T3_k_residual": 1e-4,
    "T4_omega_residual": 1e-4,
    "T5_cd_cov": 0.005,  # 0.5%
    "T6_cd_positive": 0.0,
    "T7_cp_min": -10.0,
    "T7_cp_max": 5.0,
    "T8_zero_pressure_frac": 0.01,
}


def _run_tests(data: dict, thresholds: dict[str, float]) -> list[TestResult]:
    """Run all 8 convergence tests on a loaded .pt data dict.

    Args:
        data: Dict loaded from a .pt file with keys like
            'final_residuals', 'cd', 'cd_mean50', 'cd_std50', 'x'.
        thresholds: Threshold dict to use.

    Returns:
        List of 8 TestResult objects.
    """
    results: list[TestResult] = []
    residuals = data.get("final_residuals")
    if residuals is not None:
        residuals = residuals.float().cpu()

    # T1: Pressure residual
    p_res = float(residuals[0]) if residuals is not None and len(residuals) > 0 else float("inf")
    t1_thresh = thresholds["T1_p_residual"]
    results.append(
        TestResult(
            test_id="T1",
            name="Pressure residual",
            severity=Severity.CRITICAL,
            passed=p_res < t1_thresh,
            value=p_res,
            threshold=t1_thresh,
            message=f"p residual {p_res:.2e} {'<' if p_res < t1_thresh else '>='} {t1_thresh:.0e}",
        )
    )

    # T2: Velocity residual (Ux)
    ux_res = float(residuals[1]) if residuals is not None and len(residuals) > 1 else float("inf")
    t2_thresh = thresholds["T2_ux_residual"]
    results.append(
        TestResult(
            test_id="T2",
            name="Velocity residual",
            severity=Severity.MAJOR,
            passed=ux_res < t2_thresh,
            value=ux_res,
            threshold=t2_thresh,
            message=f"Ux residual {ux_res:.2e} {'<' if ux_res < t2_thresh else '>='} {t2_thresh:.0e}",
        )
    )

    # T3: k residual
    k_res = float(residuals[2]) if residuals is not None and len(residuals) > 2 else float("inf")
    t3_thresh = thresholds["T3_k_residual"]
    results.append(
        TestResult(
            test_id="T3",
            name="k residual",
            severity=Severity.MAJOR,
            passed=k_res < t3_thresh,
            value=k_res,
            threshold=t3_thresh,
            message=f"k residual {k_res:.2e} {'<' if k_res < t3_thresh else '>='} {t3_thresh:.0e}",
        )
    )

    # T4: omega residual
    omega_res = float(residuals[3]) if residuals is not None and len(residuals) > 3 else float("inf")
    t4_thresh = thresholds["T4_omega_residual"]
    results.append(
        TestResult(
            test_id="T4",
            name="omega residual",
            severity=Severity.MINOR,
            passed=omega_res < t4_thresh,
            value=omega_res,
            threshold=t4_thresh,
            message=f"omega residual {omega_res:.2e} {'<' if omega_res < t4_thresh else '>='} {t4_thresh:.0e}",
        )
    )

    # T5: Cd stability (coefficient of variation)
    cd_mean50 = float(data.get("cd_mean50", torch.tensor([0.0])).item())
    cd_std50 = float(data.get("cd_std50", torch.tensor([0.0])).item())
    cd_cov = abs(cd_std50 / cd_mean50) if abs(cd_mean50) > 1e-12 else float("inf")
    t5_thresh = thresholds["T5_cd_cov"]
    results.append(
        TestResult(
            test_id="T5",
            name="Cd stability (CoV)",
            severity=Severity.CRITICAL,
            passed=cd_cov < t5_thresh,
            value=cd_cov,
            threshold=t5_thresh,
            message=f"Cd CoV {cd_cov:.4f} ({cd_cov * 100:.2f}%) {'<' if cd_cov < t5_thresh else '>='} {t5_thresh * 100:.1f}%",
        )
    )

    # T6: Cd physicality (positive)
    cd_val = float(data.get("cd", torch.tensor([0.0])).item())
    results.append(
        TestResult(
            test_id="T6",
            name="Cd physicality",
            severity=Severity.CRITICAL,
            passed=cd_val > 0,
            value=cd_val,
            threshold=0.0,
            message=f"Cd = {cd_val:.6f} {'> 0' if cd_val > 0 else '<= 0 (NON-PHYSICAL)'}",
        )
    )

    # T7: Cp range
    cp = data.get("x")
    if cp is not None:
        cp = cp.float().cpu()
        cp_min_val = float(cp.min())
        cp_max_val = float(cp.max())
    else:
        cp_min_val = float("-inf")
        cp_max_val = float("inf")

    t7_min = thresholds["T7_cp_min"]
    t7_max = thresholds["T7_cp_max"]
    t7_passed = cp_min_val > t7_min and cp_max_val < t7_max
    results.append(
        TestResult(
            test_id="T7",
            name="Cp range",
            severity=Severity.MAJOR,
            passed=t7_passed,
            value=cp_min_val if cp_min_val <= t7_min else cp_max_val,
            threshold=t7_min if cp_min_val <= t7_min else t7_max,
            message=f"Cp range [{cp_min_val:.2f}, {cp_max_val:.2f}] vs [{t7_min}, {t7_max}]",
        )
    )

    # T8: Zero pressure fraction
    if cp is not None:
        n_zeros = int((cp.abs() < 1e-10).sum().item())
        zero_frac = n_zeros / len(cp)
    else:
        zero_frac = 1.0

    t8_thresh = thresholds["T8_zero_pressure_frac"]
    results.append(
        TestResult(
            test_id="T8",
            name="Zero pressure fraction",
            severity=Severity.CRITICAL,
            passed=zero_frac < t8_thresh,
            value=zero_frac,
            threshold=t8_thresh,
            message=f"Zero-pressure fraction {zero_frac:.4f} ({zero_frac * 100:.2f}%) {'<' if zero_frac < t8_thresh else '>='} {t8_thresh * 100:.0f}%",
        )
    )

    return results


def _compute_grade(test_results: list[TestResult]) -> Grade:
    """Compute overall grade from individual test results.

    Args:
        test_results: List of 8 test results.

    Returns:
        PASS if all pass, FAIL if any CRITICAL fails, WARN otherwise.
    """
    failed_critical = any(not t.passed and t.severity == Severity.CRITICAL for t in test_results)
    if failed_critical:
        return Grade.FAIL

    any_failure = any(not t.passed for t in test_results)
    if any_failure:
        return Grade.WARN

    return Grade.PASS


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def classify_run(
    pt_path: Path,
    strict: bool = False,
) -> RunResult:
    """Classify a single run's convergence quality.

    Args:
        pt_path: Path to the data.pt file.
        strict: Use strict thresholds if True.

    Returns:
        RunResult with grade and per-test details.
    """
    data = torch.load(pt_path, weights_only=False, map_location="cpu")
    thresholds = _STRICT_THRESHOLDS if strict else _THRESHOLDS
    test_results = _run_tests(data, thresholds)
    grade = _compute_grade(test_results)

    # Extract run_id from directory name (e.g., "euler_run_1" → "1")
    run_id = pt_path.parent.name

    return RunResult(
        run_id=run_id,
        pt_path=str(pt_path),
        grade=grade,
        test_results=test_results,
    )


def run_qa(
    dataset_dir: Path,
    run_ids: list[int] | None = None,
    strict: bool = False,
) -> list[RunResult]:
    """Run QA on all euler_run_* directories in a dataset.

    Args:
        dataset_dir: Root directory containing euler_run_*/data.pt subdirs.
        run_ids: Optional filter — only process these run numbers.
        strict: Use strict thresholds.

    Returns:
        List of RunResult objects, sorted by run_id.
    """
    run_dirs = sorted(dataset_dir.glob("euler_run_*"))
    if not run_dirs:
        logger.warning("No euler_run_* directories found in %s", dataset_dir)
        return []

    if run_ids is not None:
        run_id_set = {f"euler_run_{rid}" for rid in run_ids}
        run_dirs = [d for d in run_dirs if d.name in run_id_set]

    results: list[RunResult] = []
    for run_dir in run_dirs:
        pt_path = run_dir / "data.pt"
        if not pt_path.exists():
            logger.warning("No data.pt in %s, skipping", run_dir)
            continue
        try:
            result = classify_run(pt_path, strict=strict)
            results.append(result)
            logger.info("%s: %s", result.run_id, result.grade.value)
        except Exception:
            logger.exception("Error processing %s", run_dir.name)

    return results


def generate_report(
    results: list[RunResult],
    output_dir: Path,
) -> Path:
    """Write convergence_report.json and convergence_summary.txt.

    Args:
        results: List of RunResult objects from run_qa.
        output_dir: Directory to write output files.

    Returns:
        Path to the convergence_report.json file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build JSON report
    runs_dict: dict[str, dict] = {}
    report: dict[str, object] = {
        "total_runs": len(results),
        "pass_count": sum(1 for r in results if r.grade == Grade.PASS),
        "warn_count": sum(1 for r in results if r.grade == Grade.WARN),
        "fail_count": sum(1 for r in results if r.grade == Grade.FAIL),
        "runs": runs_dict,
    }
    for r in results:
        runs_dict[r.run_id] = {
            "grade": r.grade.value,
            "pt_path": r.pt_path,
            "tests": {
                t.test_id: {
                    "name": t.name,
                    "severity": t.severity.value,
                    "passed": t.passed,
                    "value": t.value,
                    "threshold": t.threshold,
                    "message": t.message,
                }
                for t in r.test_results
            },
        }

    report_path = output_dir / "convergence_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Write human-readable summary
    summary_path = output_dir / "convergence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("OpenFOAM Convergence QA Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total runs:  {report['total_runs']}\n")
        f.write(f"PASS:        {report['pass_count']}\n")
        f.write(f"WARN:        {report['warn_count']}\n")
        f.write(f"FAIL:        {report['fail_count']}\n\n")

        # Per-test failure rates
        test_failures: dict[str, int] = {}
        for r in results:
            for t in r.test_results:
                if not t.passed:
                    test_failures[t.test_id] = test_failures.get(t.test_id, 0) + 1

        if test_failures:
            f.write("Per-test failure counts:\n")
            f.writelines(f"  {tid}: {test_failures[tid]} failures\n" for tid in sorted(test_failures))
            f.write("\n")

        # List FAIL runs
        fail_runs = [r for r in results if r.grade == Grade.FAIL]
        if fail_runs:
            f.write("FAIL runs:\n")
            for r in fail_runs:
                failed_tests = [t.test_id for t in r.test_results if not t.passed]
                f.write(f"  {r.run_id}: {', '.join(failed_tests)}\n")

    logger.info(
        "Report written: %s (PASS=%d, WARN=%d, FAIL=%d)",
        report_path,
        report["pass_count"],
        report["warn_count"],
        report["fail_count"],
    )
    return report_path


def generate_training_ready(
    results: list[RunResult],
    output_dir: Path,
    strict: bool = False,
) -> Path:
    """Write training_ready.txt with run IDs suitable for training.

    Args:
        results: List of RunResult objects.
        output_dir: Directory to write training_ready.txt.
        strict: If True, only include PASS runs. If False, include PASS + WARN.

    Returns:
        Path to the training_ready.txt file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if strict:
        ready_runs = [r for r in results if r.grade == Grade.PASS]
    else:
        ready_runs = [r for r in results if r.grade in (Grade.PASS, Grade.WARN)]

    ready_path = output_dir / "training_ready.txt"
    with open(ready_path, "w") as f:
        f.writelines(f"{r.run_id}\n" for r in sorted(ready_runs, key=lambda r: r.run_id))

    logger.info(
        "Training ready: %d runs written to %s (strict=%s)",
        len(ready_runs),
        ready_path,
        strict,
    )
    return ready_path


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for convergence QA."""
    parser = argparse.ArgumentParser(description="Run convergence QA on OpenFOAM .pt dataset files")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=_DEFAULT_DATASET_DIR,
        help="Directory containing euler_run_*/data.pt subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for QA output files (default: same as dataset-dir)",
    )
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs to process",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict thresholds (all residuals < 1e-4, Cd CoV < 0.5%%)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging with per-test details",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output_dir or args.dataset_dir

    results = run_qa(args.dataset_dir, run_ids=args.run_ids, strict=args.strict)
    if not results:
        logger.error(
            "No runs found in %s. Verify the directory contains euler_run_*/data.pt subdirectories.",
            args.dataset_dir,
        )
        return

    generate_report(results, output_dir)
    generate_training_ready(results, output_dir, strict=args.strict)

    # Print summary
    n_pass = sum(1 for r in results if r.grade == Grade.PASS)
    n_warn = sum(1 for r in results if r.grade == Grade.WARN)
    n_fail = sum(1 for r in results if r.grade == Grade.FAIL)
    print(f"\nConvergence QA complete: {len(results)} runs")
    print(f"  PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}")
    print(f"  Reports written to: {output_dir}")


if __name__ == "__main__":
    main()
