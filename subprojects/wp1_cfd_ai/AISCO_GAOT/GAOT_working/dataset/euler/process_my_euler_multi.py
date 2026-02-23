import glob
import os
import re

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


def process_euler_runs(geometry_dir, force_dir, output_dir):
    """
    Processes Euler/Fluent run data into PyG-compatible .pt files.
    Matches .ply geometry files with force report .out files by run number.
    """
    os.makedirs(output_dir, exist_ok=True)
    order_list = []

    # Find all .ply files and extract run numbers
    ply_files = sorted(glob.glob(os.path.join(geometry_dir, "RUN_*.ply")))

    for ply_file in tqdm(ply_files, desc="Processing Euler runs"):
        basename = os.path.basename(ply_file)  # e.g. RUN_001.ply
        run_id = basename.replace(".ply", "")  # e.g. RUN_001

        # Extract run number for matching
        match = re.search(r"RUN_(\d+)", run_id)
        if not match:
            print(f"Could not parse run number from {basename}. Skipping...")
            continue
        run_number = match.group(1)  # e.g. '001'

        # Build path to force report
        force_file = os.path.join(force_dir, f"RUN_{run_number}_force_report.out")

        if not os.path.exists(force_file):
            print(f"Missing force report for {run_id}. Skipping...")
            continue

        # 1. Extract geometry (pos) and normals (c) from .ply
        try:
            mesh = pv.read(ply_file)

            # Skip empty meshes
            if mesh.n_points == 0:
                print(f"Empty mesh for {run_id} (0 points). Skipping...")
                continue

            mesh = mesh.compute_normals(
                cell_normals=False, point_normals=True, auto_orient_normals=True
            )

            pos = torch.tensor(mesh.points, dtype=torch.float32)
            c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
        except Exception as e:
            print(f"Error reading mesh for {run_id}: {e}")
            continue

        # 2. Extract drag force from force report - last row, 3rd column
        try:
            drag_force = parse_force_report(force_file)
            if drag_force is None:
                print(f"No valid data in force report for {run_id}. Skipping...")
                continue
        except Exception as e:
            print(f"Error reading force report for {run_id}: {e}")
            continue

        # 3. Build and save PyG Data object with multi-head targets
        data = Data(pos=pos, c=c)
        data.x = None
        data.force = torch.tensor([drag_force], dtype=torch.float32)
        data.cd = torch.tensor([float('nan')], dtype=torch.float32)  # Euler has no valid Cd
        data.filename = run_id

        # Save as run_X.pt to match drivaer naming convention
        run_num_int = int(run_number)
        save_name = f"run_{run_num_int}"
        save_path = os.path.join(output_dir, f"{save_name}.pt")
        torch.save(data, save_path)
        order_list.append(save_name)

    # Generate order.txt
    with open(os.path.join(output_dir, "order.txt"), "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nSuccess! Processed {len(order_list)} runs.")
    print(f"Data and order.txt saved to: {output_dir}")


if __name__ == "__main__":
    geometry_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/GEOMETRY"
    force_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/EULER_OUTPUT"
    output_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler"

    process_euler_runs(geometry_dir, force_dir, output_dir)
