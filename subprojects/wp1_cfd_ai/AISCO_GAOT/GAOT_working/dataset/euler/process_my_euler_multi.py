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
# Saves .pt files containing pos (positions), c (normals), force (drag force),
# and cd (NaN, since Euler force reports don't have a valid Cd).
# This format supports multi-head training with masked loss alongside DrivAer data.

NUM_WORKERS = 8  # number of parallel workers for mesh processing


def parse_force_report(filepath):
    """
    Parse a Fluent force report .out file.
    Returns the last value of the 3rd column (drag_force).
    """
    last_value = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Skip header/comment lines (start with " or ()
            if not line or line.startswith('"') or line.startswith("("):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    last_value = float(parts[2])  # 3rd column = drag_force
                except ValueError:
                    continue
    return last_value


def _process_single_euler(ply_file, force_dir, output_dir):
    """Process a single Euler run. Returns save_name on success, None on failure."""
    basename = os.path.basename(ply_file)  # e.g. RUN_001.ply
    run_id = basename.replace(".ply", "")  # e.g. RUN_001

    match = re.search(r"RUN_(\d+)", run_id)
    if not match:
        return None

    run_number = match.group(1)
    force_file = os.path.join(force_dir, f"RUN_{run_number}_force_report.out")
    if not os.path.exists(force_file):
        return None

    # 1. Read mesh and compute normals
    try:
        mesh = pv.read(ply_file)
        if mesh.n_points == 0:
            return None
        mesh = mesh.compute_normals(
            cell_normals=False, point_normals=True, auto_orient_normals=True
        )
        pos = torch.tensor(mesh.points, dtype=torch.float32)
        c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
    except Exception:
        return None

    # 2. Parse force report
    try:
        drag_force = parse_force_report(force_file)
        if drag_force is None:
            return None
    except Exception:
        return None

    # 3. Build and save PyG Data object
    data = Data(pos=pos, c=c)
    data.x = None
    data.force = torch.tensor([drag_force], dtype=torch.float32)
    data.cd = torch.tensor([float("nan")], dtype=torch.float32)
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
        futures = {
            executor.submit(_process_single_euler, pf, force_dir, output_dir): pf
            for pf in ply_files
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing Euler runs"
        ):
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
    geometry_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/GEOMETRY"
    force_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/EULER_OUTPUT"
    output_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_multi"

    process_euler_runs(geometry_dir, force_dir, output_dir)
