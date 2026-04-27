#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from blended_wing.process_blended_wing import process_blended_wing
from common_preprocessing import DEFAULT_OUTPUT_DIRS
from euler.process_fluent_pressure import process_all_runs as process_euler_runs
from process_vtp_pressure import process_all as process_vtp_runs


def _resolve_output_dir(output_root: str | None, dataset_name: str) -> str:
    default_dir = DEFAULT_OUTPUT_DIRS[dataset_name]
    if output_root is None:
        return default_dir
    return os.path.join(output_root, os.path.basename(default_dir))


def run_datasets(
    datasets: list[str],
    output_root: str | None,
    euler_workers: int,
    drivaer_workers: int,
    ahmed_workers: int,
    blended_wing_workers: int,
    target_reduction: float = 0.0,
) -> None:
    for dataset_name in datasets:
        output_dir = _resolve_output_dir(output_root, dataset_name)
        print(f"\n=== Processing {dataset_name} into {output_dir} ===")
        if dataset_name == "euler":
            process_euler_runs(output_dir=output_dir, num_workers=euler_workers)
        elif dataset_name == "drivaer":
            process_vtp_runs("drivaer", num_workers=drivaer_workers, output_dir=output_dir, target_reduction=target_reduction)
        elif dataset_name == "ahmed":
            process_vtp_runs("ahmed", num_workers=ahmed_workers, output_dir=output_dir, target_reduction=target_reduction)
        elif dataset_name == "blended_wing":
            process_blended_wing(output_dir=output_dir, num_workers=blended_wing_workers)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CFD data directly into canonical GAOT3 training data.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "euler", "drivaer", "ahmed", "blended_wing"],
        help="Datasets to process. Use 'all' to run all four.",
    )
    parser.add_argument("--output-root", default=None, help="Optional root overriding the default processed_* output paths.")
    parser.add_argument("--euler-workers", type=int, default=4)
    parser.add_argument("--drivaer-workers", type=int, default=2)
    parser.add_argument("--ahmed-workers", type=int, default=2)
    parser.add_argument("--blended-wing-workers", type=int, default=8)
    parser.add_argument("--target-reduction", type=float, default=0.0,
                        help="Mesh decimation target (0.0-1.0). 0.6 = keep 40%% of vertices (drivaer/ahmed only).")
    args = parser.parse_args()

    datasets = ["euler", "drivaer", "ahmed", "blended_wing"] if "all" in args.datasets else args.datasets
    run_datasets(
        datasets=datasets,
        output_root=args.output_root,
        euler_workers=args.euler_workers,
        drivaer_workers=args.drivaer_workers,
        ahmed_workers=args.ahmed_workers,
        blended_wing_workers=args.blended_wing_workers,
        target_reduction=args.target_reduction,
    )


if __name__ == "__main__":
    main()
