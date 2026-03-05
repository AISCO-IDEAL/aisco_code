import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Processes custom Euler/Fluent run data into PyG-compatible .pt files.
# Expects:
#   - .ply geometry files in GEOMETRY/ folder (e.g. RUN_001.ply)
#   - force report .out files in RUN/EULER_OUTPUT/ folder (e.g. RUN_001_force_report.out)
#
# The force report has 3 header lines, then space-separated columns:
#   Iteration  drag_coefficient  drag_force
# We extract the last row's drag_force (3rd column).
#
# Saves .pt files containing pos (positions), c (normals), and y (drag force),
# matching the format of processed_drivaer so it can be used as training data.
# NOTE: For the multi-head format (force/cd with NaN masking), use process_my_euler_multi.py instead.

NUM_WORKERS = 8  # number of parallel workers for mesh processing
MIN_FORCE_COLS = 3  # minimum columns expected in a force report line
MM_TO_M_THRESHOLD = 50  # if any coordinate exceeds this (mm), convert to meters


def parse_force_report(filepath):
    """
    Parse a Fluent force report .out file.
    Returns the last value of the 3rd column (drag_force).
    """
    last_value = None
    with open(filepath) as f:
        for raw_line in f:
            line = raw_line.strip()
            # Skip header/comment lines (start with " or ()
            if not line or line.startswith(('"', "(")):
                continue
            parts = line.split()
            if len(parts) >= MIN_FORCE_COLS:
                try:
                    last_value = float(parts[2])  # 3rd column = drag_force
                except ValueError:
                    continue
    return last_value


def _process_single_euler(ply_file, force_dir, output_dir):
    """Process a single Euler run. Returns save_name on success, None on failure."""
    basename = os.path.basename(ply_file)
    run_id = basename.replace(".ply", "")

    match = re.search(r"RUN_(\d+)", run_id)
    run_number = match.group(1) if match else None
    force_file = os.path.join(force_dir, f"RUN_{run_number}_force_report.out") if run_number else None
    if not run_number or not os.path.exists(force_file):
        return None

    # 1. Read mesh and compute normals
    try:
        mesh = pv.read(ply_file)
        if mesh.n_points == 0:
            return None
        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
        pos = torch.tensor(mesh.points, dtype=torch.float32)
        c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
        # Auto-detect units: if any coordinate exceeds threshold, assume mm → convert to meters
        if pos.abs().max() > MM_TO_M_THRESHOLD:
            pos = pos * 1e-3
        # Euler PLY: nose is at +x, DrivAer nose is at -x.
        # Negate x to match convention: nose at negative x, tail at positive x.
        pos[:, 0] = -pos[:, 0]
        c[:, 0] = -c[:, 0]
    except Exception:
        return None

    # 2. Parse force report
    # Fluent force monitor uses drag direction = -x, so reported value is negative.
    # Negate so data.force = positive drag in +x (flow) direction.
    try:
        drag_force_raw = parse_force_report(force_file)
    except Exception:
        return None
    if drag_force_raw is None:
        return None
    drag_force = -drag_force_raw  # negate: negative Fluent report → positive drag

    # 3. Build and save PyG Data object (multi-head format for compatibility)
    data = Data(pos=pos, c=c)
    data.x = None
    data.force = torch.tensor([drag_force], dtype=torch.float32)  # [N], positive = drag in +x
    data.cd = torch.tensor([float("nan")], dtype=torch.float32)  # not available for Euler
    data.filename = run_id

    run_num_int = int(run_number)
    save_name = f"run_{run_num_int}"
    torch.save(data, os.path.join(output_dir, f"{save_name}.pt"))
    return save_name


def process_euler_runs(geometry_dir, force_dir, output_dir, num_workers=NUM_WORKERS):
    """
    Processes Euler/Fluent run data into PyG-compatible .pt files.
    Uses multiprocessing to speed up mesh reading and normal computation.
    """
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(geometry_dir, "RUN_*.ply")))
    print(f"Found {len(ply_files)} .ply files. Processing with {num_workers} workers...")

    order_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_euler, pf, force_dir, output_dir): pf for pf in ply_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Euler runs"):
            result = future.result()
            if result is not None:
                order_list.append(result)

    # Sort numerically (futures complete out of order)
    order_list.sort(key=lambda x: int(x.split("_")[1]))

    with open(os.path.join(output_dir, "order_.txt"), "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nSuccess! Processed {len(order_list)} runs.")
    print(f"Data and order_.txt saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    _default_workers = max(1, _slurm_cpus - 4) if _slurm_cpus > 0 else NUM_WORKERS

    parser = argparse.ArgumentParser(description="Preprocess Euler PLY runs (single-target / legacy format)")
    parser.add_argument("--geometry-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/GEOMETRY")
    parser.add_argument("--force-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/EULER_OUTPUT")
    parser.add_argument("--output-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler")
    parser.add_argument("--num-workers", type=int, default=_default_workers)
    args = parser.parse_args()

    print(f"Using {args.num_workers} workers (SLURM_CPUS_PER_TASK={_slurm_cpus})")
    process_euler_runs(args.geometry_dir, args.force_dir, args.output_dir, args.num_workers)
