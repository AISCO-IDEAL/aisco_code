#!/usr/bin/env python3
# =============================================================================
# EXPERIMENTAL — Not integrated into main pipeline. Manual invocation only.
# =============================================================================
"""RANS vs URANS force coefficient comparison.

Reads time-averaged Cd/Cl from a completed URANS pimpleFoam run and compares
against the RANS simpleFoam baseline.

Usage:
    # single case
    python3 compare_rans_urans.py
    python3 compare_rans_urans.py --run-id RUN_010

    # all cases in CASES_v3
    python3 compare_rans_urans.py --all-cases
    python3 compare_rans_urans.py --all-cases --output results/urans_batch.csv

Output CSV columns:
    run_id, rans_Cd, rans_Cl, urans_Cd_mean, urans_Cl_mean, urans_Cd_std,
    t_sim, n_timesteps, outcome

Outcome classifications:
    CONVERGED    — running mean Cd drift < 2% in final averaging window
    NOT_CONVERGED — drift >= 2%; extend simulation
    DIVERGED     — NaN detected in force output
    INCOMPLETE   — fewer timesteps than planned (SLURM timeout)

Plan reference: agent-output/planning/042-urans-batch-all-geometries.md
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CASES_DIR = pathlib.Path(
    "/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3"
)

# Batch output
RESULTS_DIR = pathlib.Path(__file__).resolve().parent / "results"

# URANS timing constants (shared — all cases restarted from RANS endTime=3000)
T_START_DEFAULT = 3000.0     # latestTime restart
T_TRANSIENT_FRAC = 0.5       # discard first 50% of run as transient
N_PLANNED_STEPS = 5000       # 2.0 s / 4e-4 deltaT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_forcecoeffs_file(case: pathlib.Path) -> pathlib.Path | None:
    """Probe for forceCoeffs data file, preferring the most recent subdirectory.

    OpenFOAM writes postProcessing output to a subdirectory named after the
    function-object start time.  For a URANS restart from latestTime=3000, the
    directory will be ``postProcessing/forceCoeffs/3000/``.  For a run started
    at t=0 (RANS or cold-start URANS) it is ``postProcessing/forceCoeffs/0/``.

    MPI runs typically produce ``coefficient_0.dat``; serial / newer OF versions
    produce ``coefficient.dat``.  ``forceCoeffs.dat`` is the legacy name.

    Returns:
        Path to the first matching file, or None if not found.
    """
    fc_base = case / "postProcessing" / "forceCoeffs"
    if not fc_base.exists():
        return None

    # Collect all subdirectories (time directories), sort descending to prefer latest
    time_dirs = sorted(
        (d for d in fc_base.iterdir() if d.is_dir()),
        key=lambda d: float(d.name) if d.name.replace(".", "", 1).isdigit() else -1,
        reverse=True,
    )

    for time_dir in time_dirs:
        for name in ("coefficient_0.dat", "coefficient.dat", "forceCoeffs.dat"):
            candidate = time_dir / name
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

    return None


def _parse_forcecoeffs(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a forceCoeffs.dat / coefficient_0.dat file.

    Returns:
        t: time array
        cd: drag coefficient array
        cl: lift coefficient array
    """
    times, cds, cls = [], [], []
    with path.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cols = line.split()
            if len(cols) < 5:
                continue
            try:
                t = float(cols[0])
                cd = float(cols[1])
                cl = float(cols[4])  # Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) ...
            except (ValueError, IndexError):
                continue
            # Use row index as proxy time if all time values are identical
            times.append(t)
            cds.append(cd)
            cls.append(cl)

    t_arr = np.array(times)
    # If all time values are identical (e.g. RANS iteration numbers reused),
    # replace with a synthetic monotone sequence for convergence checks.
    if len(t_arr) > 1 and np.all(t_arr == t_arr[0]):
        t_arr = np.arange(len(t_arr), dtype=float)

    return t_arr, np.array(cds), np.array(cls)


def _convergence_drift(cd_window: np.ndarray) -> float:
    """Compute percentage drift of running mean between first and second half.

    Args:
        cd_window: Cd values over the averaging window.

    Returns:
        Drift as a percentage of the mean.
    """
    half = len(cd_window) // 2
    if half == 0:
        return 100.0
    mean_first = np.mean(cd_window[:half])
    mean_second = np.mean(cd_window[half:])
    mean_total = np.mean(cd_window)
    if mean_total == 0:
        return 100.0
    return abs(mean_second - mean_first) / abs(mean_total) * 100.0


# ---------------------------------------------------------------------------
# RANS baseline per case
# ---------------------------------------------------------------------------


def load_rans_values(rans_case: pathlib.Path) -> tuple[float, float]:
    """Load RANS Cd/Cl from the case's forceCoeffs (last timestep).

    Args:
        rans_case: Path to the RANS case directory.

    Returns:
        (cd, cl) tuple from the last RANS iteration.
    """
    fc = _find_forcecoeffs_file(rans_case)
    if fc is not None:
        try:
            t, cd, cl = _parse_forcecoeffs(fc)
            if len(t) > 0:
                return float(cd[-1]), float(cl[-1])
        except Exception as exc:
            print(f"  WARN: Could not parse RANS forceCoeffs ({exc})")
    raise FileNotFoundError(
        f"No readable RANS forceCoeffs found in {rans_case}. "
        "Run simpleFoam first or ensure postProcessing/forceCoeffs/ exists."
    )


# ---------------------------------------------------------------------------
# Single-case analysis
# ---------------------------------------------------------------------------


def analyse_case(  # noqa: PLR0912  # sequential URANS outcome checks require many branches
    run_id: str,
    rans_case: pathlib.Path,
    urans_case: pathlib.Path,
    *,
    verbose: bool = True,
) -> dict[str, object]:
    """Analyse one URANS case against its RANS baseline.

    Args:
        run_id: Human-readable identifier (e.g. "RUN_010").
        rans_case: Path to the RANS case directory.
        urans_case: Path to the URANS case directory.
        verbose: Print progress / summary to stdout.

    Returns:
        Dict with keys matching the CSV column names.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysing {run_id}")
        print(f"  RANS:  {rans_case}")
        print(f"  URANS: {urans_case}")

    # ---- RANS baseline ----
    try:
        rans_cd, rans_cl = load_rans_values(rans_case)
        if verbose:
            print(f"  RANS baseline: Cd={rans_cd:.6f}  Cl={rans_cl:.6f}")
    except FileNotFoundError as exc:
        if verbose:
            print(f"  WARN: {exc}")
        rans_cd, rans_cl = float("nan"), float("nan")

    # ---- URANS forceCoeffs ----
    fc_path = _find_forcecoeffs_file(urans_case)
    if fc_path is None:
        if verbose:
            print(f"  ERROR: No forceCoeffs file in {urans_case}")
        return _row(run_id, rans_cd, rans_cl, float("nan"), float("nan"),
                    float("nan"), 0.0, 0, "INCOMPLETE")

    if verbose:
        print(f"  forceCoeffs: {fc_path}")

    t, cd, cl = _parse_forcecoeffs(fc_path)

    if len(t) == 0:
        return _row(run_id, rans_cd, rans_cl, float("nan"), float("nan"),
                    float("nan"), 0.0, 0, "DIVERGED")

    if np.any(np.isnan(cd)) or np.any(np.isnan(cl)):
        if verbose:
            print("  ERROR: NaN detected — solver diverged.")
        return _row(run_id, rans_cd, rans_cl, float("nan"), float("nan"),
                    float("nan"), float(t[-1]), len(t), "DIVERGED")

    n_steps = len(t)
    t_sim = float(t[-1] - T_START_DEFAULT)

    # Completeness check
    completeness_threshold = 0.80
    if n_steps < completeness_threshold * N_PLANNED_STEPS:
        if verbose:
            pct = n_steps / N_PLANNED_STEPS * 100
            print(f"  WARN: {n_steps}/{N_PLANNED_STEPS} steps ({pct:.0f}%) — INCOMPLETE")
        cd_mean, cl_mean, cd_std = _window_stats(t, cd, cl)
        return _row(run_id, rans_cd, rans_cl, cd_mean, cl_mean, cd_std,
                    t_sim, n_steps, "INCOMPLETE")

    # Time-averaged statistics
    cd_mean, cl_mean, cd_std = _window_stats(t, cd, cl)

    # Convergence check
    t_unique = np.unique(t)
    if len(t_unique) <= 2:
        half = len(cd) // 2
        cd_window = cd[half:]
    else:
        # Use second half of post-transient window
        half = len(cd) // 2
        cd_window = cd[half:]

    drift_pct = _convergence_drift(cd_window)
    outcome = "CONVERGED" if drift_pct < 2.0 else "NOT_CONVERGED"

    if verbose:
        print(f"  Cd_mean={cd_mean:.6f}  Cd_std={cd_std:.6f}  drift={drift_pct:.2f}%  → {outcome}")
        if not math.isnan(rans_cd):
            delta = (cd_mean - rans_cd) / abs(rans_cd) * 100
            print(f"  RANS→URANS Δ Cd = {delta:+.2f}%")

    return _row(run_id, rans_cd, rans_cl, cd_mean, cl_mean, cd_std,
                t_sim, n_steps, outcome)


def _window_stats(
    _t: np.ndarray,
    cd: np.ndarray,
    cl: np.ndarray,
) -> tuple[float, float, float]:
    """Compute time-averaged Cd/Cl stats over the second half of the run.

    Args:
        t: Time array.
        cd: Drag coefficient array.
        cl: Lift coefficient array.

    Returns:
        (cd_mean, cl_mean, cd_std)
    """
    half = len(cd) // 2
    if half == 0:
        return float(np.mean(cd)), float(np.mean(cl)), float(np.std(cd))
    return float(np.mean(cd[half:])), float(np.mean(cl[half:])), float(np.std(cd[half:]))


def _row(
    run_id: str,
    rans_cd: float,
    rans_cl: float,
    cd_mean: float,
    cl_mean: float,
    cd_std: float,
    t_sim: float,
    n_timesteps: int,
    outcome: str,
) -> dict[str, object]:
    """Build a result dict matching the CSV schema.

    Args:
        run_id: Case identifier.
        rans_cd: RANS drag coefficient.
        rans_cl: RANS lift coefficient.
        cd_mean: URANS time-averaged Cd.
        cl_mean: URANS time-averaged Cl.
        cd_std: URANS Cd standard deviation.
        t_sim: Simulated physical time (s).
        n_timesteps: Number of URANS timesteps written.
        outcome: Convergence classification.

    Returns:
        Dict with string-formatted numeric values.
    """
    def _fmt(v: float) -> str:
        return f"{v:.6f}" if not (math.isnan(v) or math.isinf(v)) else "NaN"

    return {
        "run_id": run_id,
        "rans_Cd": _fmt(rans_cd),
        "rans_Cl": _fmt(rans_cl),
        "urans_Cd_mean": _fmt(cd_mean),
        "urans_Cl_mean": _fmt(cl_mean),
        "urans_Cd_std": _fmt(cd_std),
        "t_sim": f"{t_sim:.4f}",
        "n_timesteps": n_timesteps,
        "outcome": outcome,
    }


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


def run_all_cases(output_csv: pathlib.Path) -> int:
    """Run comparison for all *_urans dirs found in CASES_DIR.

    Args:
        output_csv: Path to write the consolidated results CSV.

    Returns:
        0 on success, 1 if any case is DIVERGED or INCOMPLETE.
    """
    urans_dirs = sorted(
        d for d in CASES_DIR.iterdir()
        if d.is_dir() and d.name.endswith("_urans")
    )

    if not urans_dirs:
        print(f"ERROR: No *_urans directories found in {CASES_DIR}", file=sys.stderr)
        return 1

    print(f"Found {len(urans_dirs)} URANS case(s) in {CASES_DIR}")

    rows = []
    for urans_dir in urans_dirs:
        run_id = urans_dir.name[: -len("_urans")]
        rans_dir = CASES_DIR / run_id
        if not rans_dir.exists():
            print(f"  SKIP {run_id}: RANS dir {rans_dir} not found")
            continue
        row = analyse_case(run_id, rans_dir, urans_dir, verbose=True)
        rows.append(row)

    if not rows:
        print("ERROR: No cases could be analysed.", file=sys.stderr)
        return 1

    # Write consolidated CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "rans_Cd", "rans_Cl",
        "urans_Cd_mean", "urans_Cl_mean", "urans_Cd_std",
        "t_sim", "n_timesteps", "outcome",
    ]
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBatch CSV written: {output_csv}")

    # Summary
    outcome_counts: Counter[str] = Counter(str(r["outcome"]) for r in rows)
    print("\n=== Batch Summary ===")
    for outcome, count in sorted(outcome_counts.items()):
        print(f"  {outcome:<15s}: {count}")
    print(f"  {'TOTAL':<15s}: {len(rows)}")

    # Numeric stats for converged/not-converged cases
    numeric_rows = [
        r for r in rows
        if r["urans_Cd_mean"] != "NaN" and r["rans_Cd"] != "NaN"
    ]
    if numeric_rows:
        deltas = [
            (float(str(r["urans_Cd_mean"])) - float(str(r["rans_Cd"])))
            / abs(float(str(r["rans_Cd"]))) * 100
            for r in numeric_rows
        ]
        print(f"\n  Mean RANS→URANS ΔCd: {np.mean(deltas):+.2f}%")
        print(f"  Max  RANS→URANS ΔCd: {max(deltas, key=abs):+.2f}%")

    n_bad = outcome_counts.get("DIVERGED", 0) + outcome_counts.get("INCOMPLETE", 0)
    return 0 if n_bad == 0 else 1


# ---------------------------------------------------------------------------
# Single-case legacy entry point (RUN_001 or --run-id)
# ---------------------------------------------------------------------------


def run_single_case(run_id: str) -> int:
    """Run comparison for one case, writing per-case CSV to postProcessing/.

    Args:
        run_id: Case identifier (e.g. "RUN_001").

    Returns:
        0 on success, 1 on error.
    """
    rans_case = CASES_DIR / run_id
    urans_case = CASES_DIR / f"{run_id}_urans"

    row = analyse_case(run_id, rans_case, urans_case, verbose=True)

    # Per-case CSV in postProcessing/
    out_dir = urans_case / "postProcessing"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison_result.csv"
    fieldnames = [
        "run_id", "rans_Cd", "rans_Cl",
        "urans_Cd_mean", "urans_Cl_mean", "urans_Cd_std",
        "t_sim", "n_timesteps", "outcome",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"\nCSV written: {csv_path}")

    # Print comparison table if both values available
    rans_cd = row["rans_Cd"]
    urans_cd = row["urans_Cd_mean"]
    if rans_cd != "NaN" and urans_cd != "NaN":
        rcd = float(str(rans_cd))
        ucd = float(str(urans_cd))
        ucl = float(str(row["urans_Cl_mean"]))
        rcl = float(str(row["rans_Cl"]))
        ucstd = float(str(row["urans_Cd_std"]))
        delta = (ucd - rcd) / abs(rcd) * 100
        print(f"\n{'':25s} {'RANS':>12s} {'URANS':>12s} {'Δ (%)':>10s}")
        print("-" * 62)
        print(f"{'Cd':25s} {rcd:12.6f} {ucd:12.6f} {delta:+10.2f}%")
        if abs(rcl) > 1e-9:
            delta_cl = (ucl - rcl) / abs(rcl) * 100
            print(f"{'Cl':25s} {rcl:12.6f} {ucl:12.6f} {delta_cl:+10.2f}%")
        print(f"{'Cd_std':25s} {'—':>12s} {ucstd:12.6f}")
        print("-" * 62)

    outcome = str(row["outcome"])
    print(f"\nOutcome: {outcome}")
    return 0 if outcome != "DIVERGED" else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """CLI entry point.

    Returns:
        0 on success, non-zero on error.
    """
    parser = argparse.ArgumentParser(
        description="Compare RANS vs URANS force coefficients"
    )
    parser.add_argument(
        "--run-id",
        default="RUN_001",
        help="Single case to analyse (default: RUN_001)",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Analyse all *_urans directories in CASES_v3",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=RESULTS_DIR / "urans_batch_comparison.csv",
        help="Output CSV path for --all-cases mode",
    )
    args = parser.parse_args()

    if args.all_cases:
        return run_all_cases(args.output)
    return run_single_case(args.run_id)


if __name__ == "__main__":
    sys.exit(main())
