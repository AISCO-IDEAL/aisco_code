#!/usr/bin/env python3
"""OpenFOAM RANS Pipeline Orchestrator.

Provides four modes for managing the CFD pipeline end-to-end:

    full          PLY→STL → generate cases → mesh → solve → extract + QA
    append        Process only new geometries, continuing run IDs from max existing
    rerun         Re-run specific cases (optionally regenerate cases first)
    extract-only  Re-run extraction + convergence QA without mesh/solve

Usage:
    python run_pipeline.py --mode full
    python run_pipeline.py --mode append --new-geometry-dir /path/to/new_plys
    python run_pipeline.py --mode rerun --run-ids 18 95
    python run_pipeline.py --mode extract-only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Default Euler paths
_OPENFOAM_DIR = Path(__file__).resolve().parent
_SLURM_DIR = _OPENFOAM_DIR / "slurm"
_SUBMIT_SCRIPT = _SLURM_DIR / "submit_pipeline.sh"
_EXTRACT_SLURM = _SLURM_DIR / "openfoam_extract_pt.slurm"

_DEFAULT_PLY_INPUT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_geometry")
_DEFAULT_STL_OUTPUT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/GEOMETRY")
_DEFAULT_GEO_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/geometry_manifest.json")
_DEFAULT_CASES_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/CASES_v3")
_DEFAULT_CASES_MANIFEST = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/OPENFOAM/cases_manifest.json")
_DEFAULT_TEMPLATE = _OPENFOAM_DIR / "case_template"
_DEFAULT_DATASET_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_openfoam_cp_v050")

_DEFAULT_MAX_PARALLEL = 20
_DEFAULT_VINF = 25.0


def _unique_run_list_path(cases_dir: Path, mode: str) -> Path:
    """Generate a unique run-list filename using a timestamp.

    Prevents concurrent pipeline submissions from overwriting each other's
    run-list files (Bug 2 from analysis 054).

    Args:
        cases_dir: Parent directory for run-list files.
        mode: Pipeline mode name (full, append, rerun).

    Returns:
        Path like ``cases_dir / run_list_full_20260601T143022.txt``.
    """
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return cases_dir / f"run_list_{mode}_{stamp}.txt"


def write_run_list(run_ids: list[int], output_path: Path) -> Path:
    """Write a run-list file for submit_pipeline.sh consumption.

    Each line is a zero-padded 3-digit run ID (e.g., "001", "010", "200").

    Args:
        run_ids: List of integer run IDs.
        output_path: Path to write the run-list file.

    Returns:
        The output_path for convenience.

    Raises:
        ValueError: If run_ids is empty.
    """
    if not run_ids:
        raise ValueError("run_ids list is empty — nothing to submit")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{rid:03d}" for rid in sorted(run_ids)]
    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Run-list written: %s (%d cases)", output_path, len(lines))
    return output_path


def _get_max_existing_run_id(manifest_path: Path) -> int:
    """Read geometry manifest and return the highest existing run ID number.

    Args:
        manifest_path: Path to geometry_manifest.json.

    Returns:
        Highest run ID found, or 0 if manifest doesn't exist or is empty.
    """
    if not manifest_path.exists():
        return 0
    with open(manifest_path) as f:
        entries = json.load(f)
    if not entries:
        return 0
    max_id = 0
    for entry in entries:
        run_id_str = entry.get("run_id", "")
        # Extract numeric part from "RUN_XXX"
        try:
            num = int(run_id_str.split("_")[1])
            max_id = max(max_id, num)
        except (IndexError, ValueError):
            continue
    return max_id


def _submit_pipeline(
    run_list_path: Path,
    max_parallel: int = _DEFAULT_MAX_PARALLEL,
    no_extract: bool = False,
) -> None:
    """Invoke submit_pipeline.sh with the given run-list.

    Args:
        run_list_path: Path to the run-list file.
        max_parallel: Max simultaneous SLURM array tasks.
        no_extract: If True, skip extraction + QA step.
    """
    cmd = ["bash", str(_SUBMIT_SCRIPT), str(run_list_path), str(max_parallel)]
    if no_extract:
        cmd.append("--no-extract")

    logger.info("Submitting pipeline: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"submit_pipeline.sh failed with exit code {result.returncode}. Check stderr above for details.")


def _submit_extract_only() -> None:
    """Submit only the extraction + QA SLURM job (no mesh/solve)."""
    cmd = ["sbatch", "--parsable", str(_EXTRACT_SLURM)]
    logger.info("Submitting extract-only: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"sbatch extract failed with exit code {result.returncode}. Check stderr above for details.")
    job_id = result.stdout.strip()
    print(f"\n=== Extract-only job submitted: {job_id} ===")
    print("Monitor with:")
    print("  squeue -u $USER")
    print(f"  sacct -j {job_id} --format=JobID,State,ExitCode,Elapsed")


def _invoke_ply_to_stl(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    run_ids: list[int] | None = None,
) -> None:
    """Run ply_to_stl.py as a subprocess.

    Args:
        input_dir: Directory with PLY files.
        output_dir: Directory for STL output.
        manifest_path: Path for geometry manifest JSON.
        run_ids: Optional list of specific run IDs.
    """
    cmd = [
        sys.executable,
        str(_OPENFOAM_DIR / "ply_to_stl.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--verbose",
    ]
    if run_ids:
        cmd.extend(["--run-ids"] + [str(r) for r in run_ids])

    logger.info("Running PLY→STL: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _invoke_generate_cases(
    manifest_path: Path,
    output_dir: Path,
    cases_manifest_path: Path,
    vinf: float = _DEFAULT_VINF,
    run_ids: list[int] | None = None,
) -> None:
    """Run generate_cases.py as a subprocess.

    Args:
        manifest_path: Path to geometry_manifest.json.
        output_dir: Cases output directory.
        cases_manifest_path: Path for cases manifest JSON.
        vinf: Freestream velocity.
        run_ids: Optional list of specific run IDs.
    """
    cmd = [
        sys.executable,
        str(_OPENFOAM_DIR / "generate_cases.py"),
        "--manifest",
        str(manifest_path),
        "--output",
        str(output_dir),
        "--cases-manifest",
        str(cases_manifest_path),
        "--vinf",
        str(vinf),
        "--verbose",
    ]
    if run_ids:
        cmd.extend(["--run-ids"] + [str(r) for r in run_ids])

    logger.info("Running case generation: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_full(args: argparse.Namespace) -> None:
    """Execute --mode full: complete pipeline from PLY→STL to extraction.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Mode: FULL ===")

    # Step 1: PLY→STL
    _invoke_ply_to_stl(
        input_dir=args.ply_input,
        output_dir=args.stl_output,
        manifest_path=args.geo_manifest,
    )

    # Step 2: Generate cases
    _invoke_generate_cases(
        manifest_path=args.geo_manifest,
        output_dir=args.cases_dir,
        cases_manifest_path=args.cases_manifest,
        vinf=args.vinf,
    )

    # Step 3: Determine all run IDs from geometry manifest
    with open(args.geo_manifest) as f:
        geo = json.load(f)
    all_ids = []
    for entry in geo:
        if entry["status"] == "converted":
            num = int(entry["run_id"].split("_")[1])
            all_ids.append(num)

    if not all_ids:
        logger.error("No converted runs found in geometry manifest. Nothing to submit.")
        return

    # Step 4: Write run-list and submit
    run_list = _unique_run_list_path(args.cases_dir, "full")
    write_run_list(all_ids, run_list)

    _submit_pipeline(run_list, max_parallel=args.max_parallel)


def run_append(args: argparse.Namespace) -> None:
    """Execute --mode append: process only new geometries.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Mode: APPEND ===")

    if not args.new_geometry_dir:
        raise ValueError("--new-geometry-dir is required for append mode. Point it to a directory containing new PLY files.")

    new_geo_dir = Path(args.new_geometry_dir)
    if not new_geo_dir.is_dir():
        raise FileNotFoundError(f"New geometry directory not found: {new_geo_dir}")

    # Determine starting ID from existing manifest
    max_existing = _get_max_existing_run_id(args.geo_manifest)
    new_plys = sorted(new_geo_dir.glob("*.ply"))
    if not new_plys:
        logger.error("No PLY files found in %s", new_geo_dir)
        return

    # Assign new run IDs continuing from max existing
    new_ids = list(range(max_existing + 1, max_existing + 1 + len(new_plys)))
    logger.info(
        "Assigning new run IDs %d-%d for %d geometries (continuing from max existing %d)",
        new_ids[0],
        new_ids[-1],
        len(new_plys),
        max_existing,
    )

    # Rename PLY files to RUN_XXX.ply convention in the new geometry dir
    for ply_path, new_id in zip(new_plys, new_ids, strict=True):
        target = new_geo_dir / f"RUN_{new_id:03d}.ply"
        if ply_path != target:
            ply_path.rename(target)
            logger.info("Renamed %s → %s", ply_path.name, target.name)

    # Step 1: PLY→STL for new geometries only
    _invoke_ply_to_stl(
        input_dir=new_geo_dir,
        output_dir=args.stl_output,
        manifest_path=args.geo_manifest,
        run_ids=new_ids,
    )

    # Step 2: Generate cases for new runs only
    _invoke_generate_cases(
        manifest_path=args.geo_manifest,
        output_dir=args.cases_dir,
        cases_manifest_path=args.cases_manifest,
        vinf=args.vinf,
        run_ids=new_ids,
    )

    # Step 3: Write run-list for new IDs only and submit
    run_list = _unique_run_list_path(args.cases_dir, "append")
    write_run_list(new_ids, run_list)
    _submit_pipeline(run_list, max_parallel=args.max_parallel)


def run_rerun(args: argparse.Namespace) -> None:
    """Execute --mode rerun: re-run specific cases.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Mode: RERUN ===")

    if not args.run_ids:
        raise ValueError("--run-ids is required for rerun mode.")

    # Optionally regenerate cases first
    if args.regenerate_cases:
        logger.info("Regenerating cases for run IDs: %s", args.run_ids)
        _invoke_generate_cases(
            manifest_path=args.geo_manifest,
            output_dir=args.cases_dir,
            cases_manifest_path=args.cases_manifest,
            vinf=args.vinf,
            run_ids=args.run_ids,
        )

    # Write run-list and submit
    run_list = _unique_run_list_path(args.cases_dir, "rerun")
    write_run_list(args.run_ids, run_list)
    _submit_pipeline(run_list, max_parallel=args.max_parallel)


def run_extract_only(args: argparse.Namespace) -> None:  # noqa: ARG001
    """Execute --mode extract-only: just extraction + QA.

    Args:
        args: Parsed CLI arguments.
    """
    logger.info("=== Mode: EXTRACT-ONLY ===")
    _submit_extract_only()


def main() -> None:
    """CLI entry point for the pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="OpenFOAM RANS Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --mode full\n"
            "  python run_pipeline.py --mode append --new-geometry-dir /path/to/new\n"
            "  python run_pipeline.py --mode rerun --run-ids 18 95\n"
            "  python run_pipeline.py --mode extract-only\n"
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["full", "append", "rerun", "extract-only"],
        help="Pipeline execution mode",
    )
    parser.add_argument(
        "--run-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific run IDs (required for rerun mode)",
    )
    parser.add_argument(
        "--new-geometry-dir",
        type=str,
        default=None,
        help="Directory with new PLY files (required for append mode)",
    )
    parser.add_argument(
        "--regenerate-cases",
        action="store_true",
        help="Regenerate case directories before rerunning (rerun mode only)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=_DEFAULT_MAX_PARALLEL,
        help=f"Max simultaneous SLURM array tasks (default: {_DEFAULT_MAX_PARALLEL})",
    )
    parser.add_argument(
        "--vinf",
        type=float,
        default=_DEFAULT_VINF,
        help=f"Freestream velocity in m/s (default: {_DEFAULT_VINF})",
    )

    # Path overrides
    parser.add_argument("--ply-input", type=Path, default=_DEFAULT_PLY_INPUT)
    parser.add_argument("--stl-output", type=Path, default=_DEFAULT_STL_OUTPUT)
    parser.add_argument("--geo-manifest", type=Path, default=_DEFAULT_GEO_MANIFEST)
    parser.add_argument("--cases-dir", type=Path, default=_DEFAULT_CASES_DIR)
    parser.add_argument("--cases-manifest", type=Path, default=_DEFAULT_CASES_MANIFEST)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mode_dispatch = {
        "full": run_full,
        "append": run_append,
        "rerun": run_rerun,
        "extract-only": run_extract_only,
    }
    mode_dispatch[args.mode](args)


if __name__ == "__main__":
    main()
