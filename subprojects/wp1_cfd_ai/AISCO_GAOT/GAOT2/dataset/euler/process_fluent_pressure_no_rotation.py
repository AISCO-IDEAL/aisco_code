#!/usr/bin/env python3
"""Process Fluent results into GAOT `.pt` files without x canonicalization."""

import argparse

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.euler.process_fluent_pressure import (
    AIR_DENSITY,
    FORCE_DIR,
    NUM_WORKERS,
    RESULTS_DIR,
    process_all_runs,
)

OUTPUT_DIR = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full_no_rotation"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Fluent .cas.h5/.dat.h5 into GAOT .pt without canonicalizing x.",
    )
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--force_dir", default=FORCE_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--run_ids", nargs="+", type=int, default=None, help="Specific run IDs to process (e.g., 1 2 3)")
    parser.add_argument("--max_runs", type=int, default=None, help="Process at most N runs")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help="Number of parallel workers (default: 4)")
    parser.add_argument(
        "--air-density",
        type=float,
        default=AIR_DENSITY,
        help=(
            f"Air density in kg/m³ used for Cp, wall-shear recovery, and Cd (default: {AIR_DENSITY}). "
            "Set to 1.2041 for runs where the Fluent reference density was changed."
        ),
    )
    args = parser.parse_args()

    process_all_runs(
        results_dir=args.results_dir,
        force_dir=args.force_dir,
        output_dir=args.output_dir,
        run_ids=args.run_ids,
        max_runs=args.max_runs,
        num_workers=args.workers,
        air_density=args.air_density,
        canonicalize_orientation=False,
    )
