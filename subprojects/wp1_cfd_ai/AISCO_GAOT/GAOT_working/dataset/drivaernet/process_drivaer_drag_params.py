import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import pyvista as pv
import torch
from torch_geometric.data import Data
from tqdm import tqdm

NUM_WORKERS = 8  # number of parallel workers for mesh processing

def _process_single_drivaer(run_folder, output_dir):
    """Process a single DrivAer run folder. Returns folder_name on success, None on failure."""
    folder_name = os.path.basename(run_folder)
    run_id = folder_name.split("_")[1]

    # File paths
    ply_file = os.path.join(run_folder, f"drivaer_{run_id}.ply")
    # Pointing to the new file containing the calculated drag force
    csv_file = os.path.join(run_folder, f"drag_param_{run_id}.csv")

    if not os.path.exists(ply_file) or not os.path.exists(csv_file):
        return None

    # 1. Read mesh and compute normals
    try:
        mesh = pv.read(ply_file)
        mesh = mesh.compute_normals(
            cell_normals=False, point_normals=True, auto_orient_normals=True
        )
        pos = torch.tensor(mesh.points, dtype=torch.float32)
        c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
    except Exception:
        return None

    # 2. Read Cd and Force from the new CSV
    try:
        rd = pd.read_csv(csv_file)
        # Using uppercase 'Cd' as per DrivAer convention
        cd_value = rd['Cd'].iloc[0]
        force_value = rd['Drag_Force_N'].iloc[0]
    except Exception:
        return None

    # 3. Build and save PyG Data object
    data = Data(pos=pos, c=c)
    data.x = None
    data.cd = torch.tensor([cd_value], dtype=torch.float32)
    data.force = torch.tensor([force_value], dtype=torch.float32) # Populated with Newtons
    data.filename = folder_name

    torch.save(data, os.path.join(output_dir, f"{folder_name}.pt"))
    return folder_name


def process_custom_drivaer(base_path, output_dir, num_workers=NUM_WORKERS):
    """
    Processes custom DrivAer run folders into PyG-compatible .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    run_folders = glob.glob(os.path.join(base_path, "run_*"))
    run_folders.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    print(f"Found {len(run_folders)} run folders. Processing with {num_workers} workers...")

    order_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_process_single_drivaer, rf, output_dir): rf
            for rf in run_folders
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing DrivAer runs"
        ):
            result = future.result()
            if result is not None:
                order_list.append(result)

    # Sort numerically
    order_list.sort(key=lambda x: int(x.split("_")[1]))

    with open(os.path.join(output_dir, "order_.txt"), "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nSuccess! Processed {len(order_list)} runs.")

if __name__ == "__main__":
    # Your cluster paths
    base_path = "/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete"
    output_dir = "/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_drag_params"

    process_custom_drivaer(base_path, output_dir)
