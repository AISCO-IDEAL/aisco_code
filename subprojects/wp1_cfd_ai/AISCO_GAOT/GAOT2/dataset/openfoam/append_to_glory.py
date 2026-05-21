#!/usr/bin/env python3
"""Append strict_audit runs to GloryToTheGAOT dataset with prefix 6.

Each run directory in strict_audit contains a data.pt (PyG Data).
Files are copied as sample_6XXXXX.pt, name_mapping.json and order_.txt updated.

Usage:
    python3 append_to_glory.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

STRICT_AUDIT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/strict_audit")
GLORY_DIR = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/GloryToTheGAOT")
NAME_MAPPING = GLORY_DIR / "name_mapping.json"
ORDER_FILE = GLORY_DIR / "order_.txt"
PREFIX_START = 600_001  # first sample_6XXXXX id


def load_name_mapping() -> dict:
    """Load existing name_mapping.json."""
    with open(NAME_MAPPING) as f:
        return json.load(f)


def save_name_mapping(mapping: dict, dry_run: bool) -> None:
    """Save updated name_mapping.json."""
    if dry_run:
        print(f"  [dry-run] would write {NAME_MAPPING}")
        return
    with open(NAME_MAPPING, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Updated {NAME_MAPPING}")


def load_order() -> list[str]:
    """Load existing order_.txt as list of sample ids."""
    with open(ORDER_FILE) as f:
        return [line.strip() for line in f if line.strip()]


def save_order(order: list[str], dry_run: bool) -> None:
    """Write order_.txt."""
    if dry_run:
        print(f"  [dry-run] would write {ORDER_FILE} ({len(order)} entries)")
        return
    with open(ORDER_FILE, "w") as f:
        f.write("\n".join(order) + "\n")
    print(f"  Updated {ORDER_FILE} ({len(order)} entries)")


def get_run_dirs() -> list[Path]:
    """Return sorted list of run directories in strict_audit."""
    return sorted(p for p in STRICT_AUDIT.iterdir() if p.is_dir())


def next_sample_id(mapping: dict) -> int:
    """Return next available sample id for prefix 6 (>= PREFIX_START)."""
    existing_6 = [
        int(v.split("_")[1]) for v in mapping["assignments"].values() if isinstance(v, str) and v.startswith("sample_6")
    ]
    if existing_6:
        return max(existing_6) + 1
    return PREFIX_START


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen without copying files")
    args = parser.parse_args()

    run_dirs = get_run_dirs()
    print(f"Found {len(run_dirs)} run directories in strict_audit")

    mapping = load_name_mapping()
    order = load_order()

    assignments: dict = mapping.setdefault("assignments", {})

    # Check for already-appended entries (idempotency)
    already_done = {v for v in assignments.values() if isinstance(v, str) and v.startswith("sample_6")}
    print(f"Already appended as prefix-6: {len(already_done)}")

    next_id = next_sample_id(mapping)
    copied = 0
    skipped = 0

    for run_dir in run_dirs:
        src = run_dir / "data.pt"
        if not src.exists():
            print(f"  WARNING: no data.pt in {run_dir.name}, skipping")
            skipped += 1
            continue

        # Use directory name as the mapping key (source identifier)
        map_key = f"openfoam_federico::{run_dir.name}"

        # Idempotency: skip if already present
        if map_key in assignments:
            skipped += 1
            continue

        sample_name = f"sample_{next_id}"
        dest = GLORY_DIR / f"{sample_name}.pt"
        next_id += 1

        if args.dry_run:
            print(f"  [dry-run] {run_dir.name} -> {sample_name}.pt  (source_dataset -> openfoam_federico)")
        else:
            data = torch.load(src, map_location="cpu", weights_only=False)
            data.source_dataset = "openfoam_federico"
            torch.save(data, dest)

        assignments[map_key] = sample_name
        order.append(sample_name)
        copied += 1

    print(f"\nSummary: {copied} copied, {skipped} skipped")

    if copied > 0:
        save_name_mapping(mapping, args.dry_run)
        save_order(order, args.dry_run)
    else:
        print("Nothing to do.")


if __name__ == "__main__":
    main()
