#!/usr/bin/env python3
"""
Inspect processed dataset for runs with NaN Cp values.
Generates a list of corrupted runs that can be manually deleted.

Usage:
    python inspect_nan_runs.py /cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_cp_downsampled
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def check_run_for_nan(run_dir):
    """
    Check if a run's data.pt has NaN in Cp field (x).
    Returns: (has_nan, num_nan, total_vertices)
    """
    data_path = os.path.join(run_dir, "data.pt")
    if not os.path.exists(data_path):
        return None, 0, 0

    try:
        data = torch.load(data_path)
        if hasattr(data, 'x') and data.x is not None:
            num_nan = torch.isnan(data.x).sum().item()
            total = data.x.numel()
            has_nan = num_nan > 0
            return has_nan, num_nan, total
        else:
            return None, 0, 0
    except Exception as e:
        print(f"  ERROR loading {data_path}: {e}")
        return None, 0, 0


def inspect_dataset(dataset_path):
    """Scan all runs and identify NaN runs."""
    run_dirs = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])

    nan_runs = []
    valid_runs = []
    error_runs = []

    print(f"\nScanning {len(run_dirs)} runs in {dataset_path}...")
    print()

    for run_name in tqdm(run_dirs, desc="Checking runs"):
        run_path = os.path.join(dataset_path, run_name)
        has_nan, num_nan, total = check_run_for_nan(run_path)

        if has_nan is None:
            error_runs.append(run_name)
        elif has_nan:
            nan_runs.append((run_name, num_nan, total))
        else:
            valid_runs.append(run_name)

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total runs:     {len(run_dirs)}")
    print(f"Valid runs:     {len(valid_runs)}")
    print(f"NaN runs:       {len(nan_runs)}")
    print(f"Error runs:     {len(error_runs)}")

    if nan_runs:
        print("\n" + "="*80)
        print("NaN RUNS (to be removed)")
        print("="*80)
        for run_name, num_nan, total in nan_runs:
            pct_nan = 100.0 * num_nan / total if total > 0 else 0
            print(f"  {run_name:30} {num_nan:6d}/{total:6d} NaN ({pct_nan:5.1f}%)")

    if error_runs:
        print("\n" + "="*80)
        print("ERROR RUNS (corrupted/missing data.pt)")
        print("="*80)
        for run_name in error_runs:
            print(f"  {run_name}")

    # Generate removal commands
    all_bad_runs = nan_runs + [(name, None, None) for name in error_runs]
    if all_bad_runs:
        print("\n" + "="*80)
        print("REMOVAL COMMANDS (copy & paste to delete)")
        print("="*80)
        print()
        for run_name, *_ in all_bad_runs:
            run_path = os.path.join(dataset_path, run_name)
            print(f"rm -rf '{run_path}'")

        print("\n# Or bulk delete all bad runs:")
        print("cat << 'EOF' | bash")
        for run_name, *_ in all_bad_runs:
            run_path = os.path.join(dataset_path, run_name)
            print(f"rm -rf '{run_path}'")
        print("EOF")

    # Save manifest of valid runs
    manifest_path = os.path.join(dataset_path, "valid_runs.txt")
    with open(manifest_path, "w") as f:
        for run_name in valid_runs:
            f.write(f"{run_name}\n")
    print(f"\n✓ Valid runs manifest saved to: {manifest_path}")

    return len(valid_runs), len(nan_runs), len(error_runs)


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset for NaN runs.")
    parser.add_argument("dataset_path", help="Path to processed dataset folder")
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print(f"ERROR: {args.dataset_path} is not a directory")
        sys.exit(1)

    valid, nan, error = inspect_dataset(args.dataset_path)

    print("\n" + "="*80)
    if nan == 0 and error == 0:
        print("✓ Dataset is clean! No NaN or corrupt runs found.")
    else:
        print(f"⚠ Found {nan + error} problematic runs. See removal commands above.")
    print("="*80)


if __name__ == "__main__":
    main()
