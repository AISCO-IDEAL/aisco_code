#!/usr/bin/env python3

import argparse
import os
from typing import Any

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset.preprocessing_ahmed_drivaer import (
    process_all as process_vtp_runs,
)
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset.preprocessing_blended_wing import process_blended_wing
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset.preprocessing_euler import process_all_runs as process_euler_runs

# Single editable preprocessing settings section.
PREPROCESSING_CONFIG: dict[str, Any] = {
    "defaults": {
        # Dataset execution order when --datasets all is used.
        "dataset_order": ["euler", "drivaer", "ahmed", "blended_wing"],
        # Per-dataset CPU workers for the raw-data conversion step.
        "workers": {
            "euler": 4,
            "drivaer": 2,
            "ahmed": 2,
            "blended_wing": 8,
        },
        # Base output directory. Each selected dataset is written to
        # <output_path>/<dataset_name>/.
        "output_path": "/cluster/work/fuge/AISCO/wp1_cfd_ai/all_processed",
        # Mesh decimation ratio for VTP datasets (drivaer/ahmed only).
        "target_reduction": 0.0,
    },
    "datasets": {
        "euler": {
            # Fluent Euler run directory with per-run .cas.h5/.dat.h5 files.
            "results_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/RESULTS",
            # Fluent force report directory with RUN_xxx_force_report.out files.
            "force_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/EULER_OUTPUT",
        },
        "drivaer": {
            # DrivAer raw VTP directory (one run per file).
            "input_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete",
        },
        "ahmed": {
            # Ahmed raw VTP directory (one run per file).
            "input_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/ahmed_complete",
        },
        "blended_wing": {
            # Blended-wing VTK geometry directory.
            "vtk_dir": "/cluster/work/fuge/AISCO/wp1_cfd_ai/blended_wing/train/vtk",
            # Blended-wing metadata file with aerodynamic conditions per case.
            "case_data_path": "/cluster/work/fuge/AISCO/wp1_cfd_ai/blended_wing/train/case_data_solar.dat",
        },
    },
}

RUNNER_DEFAULTS = PREPROCESSING_CONFIG["defaults"]
RUNNER_DATASET_ORDER = list(RUNNER_DEFAULTS["dataset_order"])
RUNNER_DEFAULT_WORKERS = dict(RUNNER_DEFAULTS["workers"])
RUNNER_DEFAULT_OUTPUT_PATH = str(RUNNER_DEFAULTS["output_path"])
RUNNER_DEFAULT_TARGET_REDUCTION = float(RUNNER_DEFAULTS["target_reduction"])


def _resolve_output_dir(output_root: str | None, dataset_name: str) -> str:
    base_output_path = RUNNER_DEFAULT_OUTPUT_PATH if output_root is None else output_root
    return os.path.join(base_output_path, dataset_name)


def run_datasets(
    datasets: list[str],
    output_root: str | None,
    euler_workers: int,
    drivaer_workers: int,
    ahmed_workers: int,
    blended_wing_workers: int,
    target_reduction: float = 0.0,
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for dataset_name in datasets:
        dataset_cfg: dict[str, Any] = PREPROCESSING_CONFIG["datasets"][dataset_name]
        output_dir = _resolve_output_dir(output_root, dataset_name)
        print(f"\n=== Processing {dataset_name} into {output_dir} ===")
        if dataset_name == "euler":
            processed = process_euler_runs(
                results_dir=str(dataset_cfg["results_dir"]),
                force_dir=str(dataset_cfg["force_dir"]),
                output_dir=output_dir,
                num_workers=euler_workers,
            )
            summary[dataset_name] = len(processed)
        elif dataset_name == "drivaer":
            processed = process_vtp_runs(
                "drivaer",
                input_dir=str(dataset_cfg["input_dir"]),
                num_workers=drivaer_workers,
                output_dir=output_dir,
                target_reduction=target_reduction,
            )
            summary[dataset_name] = len(processed)
        elif dataset_name == "ahmed":
            processed = process_vtp_runs(
                "ahmed",
                input_dir=str(dataset_cfg["input_dir"]),
                num_workers=ahmed_workers,
                output_dir=output_dir,
                target_reduction=target_reduction,
            )
            summary[dataset_name] = len(processed)
        elif dataset_name == "blended_wing":
            processed = process_blended_wing(
                vtk_dir=str(dataset_cfg["vtk_dir"]),
                case_data_path=str(dataset_cfg["case_data_path"]),
                output_dir=output_dir,
                num_workers=blended_wing_workers,
            )
            summary[dataset_name] = len(processed)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    print("\n=== Preprocessing Summary ===")
    total = 0
    for dataset_name in datasets:
        count = summary.get(dataset_name, 0)
        total += count
        print(f"  {dataset_name}: {count} samples")
    print(f"  total: {total} samples")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CFD data directly into canonical GAOT3 training data.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *RUNNER_DATASET_ORDER],
        help="Datasets to process. Use 'all' to run all four.",
    )
    parser.add_argument(
        "--output-path",
        "--output-root",
        dest="output_path",
        default=None,
        help=("Optional base output folder. Each selected dataset writes to <output_path>/<dataset_name>."),
    )
    parser.add_argument("--euler-workers", type=int, default=int(RUNNER_DEFAULT_WORKERS["euler"]))
    parser.add_argument("--drivaer-workers", type=int, default=int(RUNNER_DEFAULT_WORKERS["drivaer"]))
    parser.add_argument("--ahmed-workers", type=int, default=int(RUNNER_DEFAULT_WORKERS["ahmed"]))
    parser.add_argument("--blended-wing-workers", type=int, default=int(RUNNER_DEFAULT_WORKERS["blended_wing"]))
    parser.add_argument(
        "--target-reduction",
        type=float,
        default=RUNNER_DEFAULT_TARGET_REDUCTION,
        help="Mesh decimation target (0.0-1.0). 0.6 = keep 40%% of vertices (drivaer/ahmed only).",
    )
    args = parser.parse_args()

    datasets = RUNNER_DATASET_ORDER if "all" in args.datasets else args.datasets
    run_datasets(
        datasets=datasets,
        output_root=args.output_path,
        euler_workers=args.euler_workers,
        drivaer_workers=args.drivaer_workers,
        ahmed_workers=args.ahmed_workers,
        blended_wing_workers=args.blended_wing_workers,
        target_reduction=args.target_reduction,
    )


if __name__ == "__main__":
    main()
