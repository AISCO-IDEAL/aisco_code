import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyvista as pv
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm import tqdm

NUM_WORKERS = 32  # number of parallel workers for mesh processing


def _interpolate_cp_from_vtp(ply_pos, vtp_path, cp_key="static(p)_coeffMean"):
    """
    Read VTP boundary mesh, convert cell Cp to point data, then interpolate
    onto PLY mesh vertices using nearest-neighbor lookup.

    Parameters
    ----------
    ply_pos : np.ndarray [N, 3]
        PLY vertex coordinates (Ahmed is already z-up, no swap needed).
    vtp_path : str
        Path to boundary_{id}.vtp file.
    cp_key : str
        Cell data key for pressure coefficient.

    Returns
    -------
    cp : np.ndarray [N]  -- Cp values at each PLY vertex.
    """
    vtp = pv.read(vtp_path)
    vtp_pt = vtp.cell_data_to_point_data()
    vtp_cp = vtp_pt.point_data[cp_key]  # [M]
    vtp_pts = vtp_pt.points  # [M, 3]

    tree = cKDTree(vtp_pts)
    _, idx = tree.query(ply_pos, k=1)
    return vtp_cp[idx].astype(np.float32)


def _process_single_ahmed(run_folder, output_dir):
    """Process a single Ahmed run folder. Returns folder_name on success, None on failure."""
    folder_name = os.path.basename(run_folder)
    run_id = folder_name.split("_")[1]

    # File paths
    ply_file = os.path.join(run_folder, f"ahmed_{run_id}.ply")
    vtp_file = os.path.join(run_folder, f"boundary_{run_id}.vtp")
    # Pointing to the new file containing the calculated drag force
    csv_file = os.path.join(run_folder, f"drag_param_{run_id}.csv")

    if not os.path.exists(ply_file) or not os.path.exists(csv_file):
        return None

    # 1. Read mesh and compute normals
    try:
        mesh = pv.read(ply_file)
        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
        pos = torch.tensor(mesh.points, dtype=torch.float32)
        c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
    except Exception:
        return None

    # 2. Interpolate surface Cp from VTP onto PLY mesh
    if os.path.exists(vtp_file):
        try:
            cp_np = _interpolate_cp_from_vtp(pos.numpy(), vtp_file)
            cp = torch.tensor(cp_np, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
        except Exception:
            cp = torch.full((pos.shape[0], 1), float("nan"), dtype=torch.float32)
    else:
        cp = torch.full((pos.shape[0], 1), float("nan"), dtype=torch.float32)

    # 3. Read Cd and Force from the new CSV
    try:
        rd = pd.read_csv(csv_file)
        # Using lowercase 'cd' as per Ahmed Body convention
        cd_value = rd["cd"].iloc[0]
        force_value = rd["Drag_Force_N"].iloc[0]
    except Exception:
        return None

    # 4. Build and save PyG Data object
    data = Data(pos=pos, c=c)
    data.x = cp
    data.cd = torch.tensor([cd_value], dtype=torch.float32)
    data.force = torch.tensor([force_value], dtype=torch.float32)  # Populated with Newtons
    data.filename = folder_name

    torch.save(data, os.path.join(output_dir, f"{folder_name}.pt"))
    return folder_name


def process_custom_ahmed(base_path, output_dir, num_workers=NUM_WORKERS):
    """
    Processes custom Ahmed run folders into PyG-compatible .pt files.
    """
    os.makedirs(output_dir, exist_ok=True)

    run_folders = glob.glob(os.path.join(base_path, "run_*"))
    run_folders.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    print(f"Found {len(run_folders)} run folders. Processing with {num_workers} workers...")

    order_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_ahmed, rf, output_dir): rf for rf in run_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Ahmed runs"):
            result = future.result()
            if result is not None:
                order_list.append(result)

    # Sort numerically
    order_list.sort(key=lambda x: int(x.split("_")[1]))

    with open(os.path.join(output_dir, "order_.txt"), "w") as f:
        f.writelines(f"{name}\n" for name in order_list)

    print(f"\nSuccess! Processed {len(order_list)} runs.")


if __name__ == "__main__":
    import argparse

    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    _default_workers = max(1, _slurm_cpus - 4) if _slurm_cpus > 0 else NUM_WORKERS

    parser = argparse.ArgumentParser(description="Preprocess Ahmed body PLY runs (multi-head format)")
    parser.add_argument("--base-path", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/ahmed_complete")
    parser.add_argument("--output-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_ahmed_drag_params")
    parser.add_argument("--num-workers", type=int, default=_default_workers)
    args = parser.parse_args()

    print(f"Using {args.num_workers} workers (SLURM_CPUS_PER_TASK={_slurm_cpus})")
    process_custom_ahmed(args.base_path, args.output_dir, args.num_workers)
