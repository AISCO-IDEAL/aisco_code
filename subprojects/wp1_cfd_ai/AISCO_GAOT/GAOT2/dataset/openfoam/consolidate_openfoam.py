"""Consolidate dirty OpenFOAM .pt files into clean/ and strict/ folders.

After the extraction + QA array jobs complete, this script reads
``convergence_report.json`` from each batch's QA output and copies
passing samples into two consolidated flat directories:

- **clean/**  — samples graded PASS or WARN (suitable for training)
- **strict/** — samples graded PASS only (highest-confidence data)

Each sample is renamed with its batch prefix to avoid collisions:
    dirty/of021/euler_run_5/data.pt
    → clean/of021_euler_run_5/data.pt
    → strict/of021_euler_run_5/data.pt  (only if PASS)

An ``order_.txt`` file is written to both output directories.

Usage:
    python consolidate_openfoam.py [--dirty-dir PATH] [--clean-dir PATH] \\
                                   [--strict-dir PATH] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Defaults (from batch config)
# ---------------------------------------------------------------------------

try:
    from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.openfoam_batch_config import (
        CLEAN_DIR as _DEFAULT_CLEAN,
    )
    from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.openfoam_batch_config import (
        DIRTY_DIR as _DEFAULT_DIRTY,
    )
    from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.openfoam.openfoam_batch_config import (
        STRICT_DIR as _DEFAULT_STRICT,
    )
except ImportError:
    # Fallback when run directly without the package on PYTHONPATH
    _base = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/openfoam_merged")
    _DEFAULT_DIRTY = _base / "dirty"
    _DEFAULT_CLEAN = _base / "clean"
    _DEFAULT_STRICT = _base / "strict"


# ---------------------------------------------------------------------------
#  Core logic
# ---------------------------------------------------------------------------


def _load_qa_report(batch_dir: Path) -> dict[str, str] | None:
    """Return mapping of run_id → grade from convergence_report.json.

    Args:
        batch_dir: Per-batch subdirectory inside dirty/ (e.g. dirty/of021/).

    Returns:
        Dict ``{"euler_run_5": "PASS", ...}`` or None if no report found.
    """
    report_path = batch_dir / "qa" / "convergence_report.json"
    if not report_path.exists():
        logger.warning("No QA report found at %s — batch will be skipped", report_path)
        return None

    with open(report_path) as f:
        report = json.load(f)

    return {run_id: info["grade"] for run_id, info in report.get("runs", {}).items()}


def _copy_run(
    src_run_dir: Path,
    dest_parent: Path,
    dest_name: str,
    dry_run: bool,
) -> None:
    """Copy euler_run_N/data.pt into dest_parent/dest_name/data.pt.

    Args:
        src_run_dir: Source directory (e.g. dirty/of021/euler_run_5/).
        dest_parent: Destination root (clean/ or strict/).
        dest_name: Unique destination folder name (e.g. of021_euler_run_5).
        dry_run: If True, log the action but do not copy.
    """
    src_pt = src_run_dir / "data.pt"
    if not src_pt.exists():
        logger.warning("Expected .pt file missing: %s", src_pt)
        return

    dest_dir = dest_parent / dest_name
    dest_pt = dest_dir / "data.pt"

    if dry_run:
        logger.info("[DRY-RUN] Would copy %s → %s", src_pt, dest_pt)
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_pt, dest_pt)
    logger.debug("Copied %s → %s", src_pt, dest_pt)


def consolidate(
    dirty_dir: Path,
    clean_dir: Path,
    strict_dir: Path,
    dry_run: bool = False,
) -> None:
    """Populate clean/ and strict/ from dirty/ using per-batch QA reports.

    Args:
        dirty_dir: Root of per-batch extraction output.
        clean_dir: Destination for PASS + WARN samples.
        strict_dir: Destination for PASS-only samples.
        dry_run: Log actions without writing any files.
    """
    if not dirty_dir.exists():
        raise FileNotFoundError(f"dirty_dir does not exist: {dirty_dir}. Run the extraction + QA array jobs first.")

    clean_dir.mkdir(parents=True, exist_ok=True)
    strict_dir.mkdir(parents=True, exist_ok=True)

    clean_order: list[str] = []
    strict_order: list[str] = []

    batch_dirs = sorted(dirty_dir.iterdir())
    for batch_dir in batch_dirs:
        if not batch_dir.is_dir():
            continue

        batch_short = batch_dir.name
        grades = _load_qa_report(batch_dir)
        if grades is None:
            continue

        n_pass = n_warn = n_fail = 0
        for run_name, grade in sorted(grades.items()):
            src_run_dir = batch_dir / run_name
            dest_name = f"{batch_short}_{run_name}"

            if grade == "PASS":
                _copy_run(src_run_dir, clean_dir, dest_name, dry_run)
                _copy_run(src_run_dir, strict_dir, dest_name, dry_run)
                clean_order.append(dest_name)
                strict_order.append(dest_name)
                n_pass += 1
            elif grade == "WARN":
                _copy_run(src_run_dir, clean_dir, dest_name, dry_run)
                clean_order.append(dest_name)
                n_warn += 1
            else:  # FAIL
                n_fail += 1

        logger.info(
            "Batch %-10s: PASS=%d WARN=%d FAIL=%d  →  clean+=%d strict+=%d",
            batch_short,
            n_pass,
            n_warn,
            n_fail,
            n_pass + n_warn,
            n_pass,
        )

    # Write order files
    _write_order(clean_dir, sorted(clean_order), dry_run)
    _write_order(strict_dir, sorted(strict_order), dry_run)

    print(f"\nConsolidation complete:\n  clean/  → {len(clean_order)} samples\n  strict/ → {len(strict_order)} samples")


def _write_order(dest_dir: Path, names: list[str], dry_run: bool) -> None:
    """Write order_.txt to dest_dir.

    Args:
        dest_dir: Directory to write into.
        names: Sorted list of run directory names.
        dry_run: Skip writing if True.
    """
    order_path = dest_dir / "order_.txt"
    if dry_run:
        logger.info("[DRY-RUN] Would write %d entries to %s", len(names), order_path)
        return
    with open(order_path, "w") as f:
        f.writelines(f"{name}\n" for name in names)
    logger.info("Wrote %s (%d entries)", order_path, len(names))


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for post-QA consolidation."""
    parser = argparse.ArgumentParser(description="Consolidate dirty OF .pt files into clean/ and strict/ folders")
    parser.add_argument(
        "--dirty-dir",
        type=Path,
        default=_DEFAULT_DIRTY,
        help="Per-batch extraction root (default: %(default)s)",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=_DEFAULT_CLEAN,
        help="Output directory for PASS+WARN samples (default: %(default)s)",
    )
    parser.add_argument(
        "--strict-dir",
        type=Path,
        default=_DEFAULT_STRICT,
        help="Output directory for PASS-only samples (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be copied without touching the filesystem",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    consolidate(
        dirty_dir=args.dirty_dir,
        clean_dir=args.clean_dir,
        strict_dir=args.strict_dir,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
