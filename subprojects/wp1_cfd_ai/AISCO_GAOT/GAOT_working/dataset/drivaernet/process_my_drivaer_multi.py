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

# Expects per-run folder containing:
#   drivaer_{id}.ply   -- decimated surface geometry (~386K pts, y-up)
#   boundary_{id}.vtp  -- full CFD boundary mesh with cell data (~8.8M cells, z-up)
#   force_mom_{id}.csv -- aerodynamic coefficients (Cd, etc.)
#
# Saves .pt file containing:
#   pos   (N,3)  -- vertex coordinates (z-up after y<->z swap)
#   c     (N,3)  -- outward surface normals (z-up)
#   x     (N,1)  -- surface pressure coefficient Cp (interpolated from VTP)
#   cd    (1,)   -- drag coefficient from CSV
#   force (1,)   -- NaN (DrivAer CSVs only have Cd, not force in Newtons)
#   filename str

NUM_WORKERS = 8  # number of parallel workers for mesh processing
V_INF_DRIVAER = 38.89  # m/s
COND_EPS = 1e-6


def _compute_geom_conditioning(pos: torch.Tensor, v_inf: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build geometry-aware conditioning vectors from unnormalized coordinates."""
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    max_extent = torch.clamp(extent.max(), min=COND_EPS)
    aspect = extent / max_extent
    conditioning = torch.cat(
        [
            torch.tensor([v_inf], dtype=torch.float32),
            extent.to(dtype=torch.float32),
            aspect.to(dtype=torch.float32),
        ],
        dim=0,
    ).unsqueeze(0)
    return extent, aspect, conditioning


def _interpolate_cp_from_vtp(ply_pos_zup, vtp_path, cp_key="CpMeanTrim"):
    """
    Read VTP boundary mesh, convert cell Cp to point data, then interpolate
    onto PLY mesh vertices using nearest-neighbor lookup.

    Parameters
    ----------
    ply_pos_zup : np.ndarray [N, 3]
        PLY vertex coordinates AFTER y<->z swap (z-up convention).
    vtp_path : str
        Path to boundary_{id}.vtp file.
    cp_key : str
        Cell data key for pressure coefficient.

    Returns
    -------
    cp : np.ndarray [N]  -- Cp values at each PLY vertex.
    """
    vtp = pv.read(vtp_path)
    # VTP is already z-up, no axis swap needed
    vtp_pt = vtp.cell_data_to_point_data()
    vtp_cp = vtp_pt.point_data[cp_key]  # [M]
    vtp_pts = vtp_pt.points  # [M, 3]

    tree = cKDTree(vtp_pts)
    _, idx = tree.query(ply_pos_zup, k=1)
    return vtp_cp[idx].astype(np.float32)


def _process_single_drivaer(run_folder, output_dir, target_col):
    """Process a single DrivAer run folder. Returns folder_name on success, None on failure."""
    folder_name = os.path.basename(run_folder)
    run_number = folder_name.split("_")[1]

    ply_file = os.path.join(run_folder, f"drivaer_{run_number}.ply")
    vtp_file = os.path.join(run_folder, f"boundary_{run_number}.vtp")
    csv_file = os.path.join(run_folder, f"force_mom_{run_number}.csv")

    if not os.path.exists(ply_file) or not os.path.exists(csv_file):
        return None

    # 1. Read PLY mesh and compute normals
    try:
        mesh = pv.read(ply_file)
        mesh = mesh.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
        pos = torch.tensor(mesh.points, dtype=torch.float32)
        c = torch.tensor(mesh.point_data["Normals"], dtype=torch.float32)
        # DrivAer PLY is y-up (x=flow, y=vertical, z=lateral).
        # Swap y<->z to canonical z-up convention (x=flow, y=lateral, z=vertical).
        pos = pos[:, [0, 2, 1]]
        c = c[:, [0, 2, 1]]
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

    # 3. Read Cd from CSV
    try:
        rd = pd.read_csv(csv_file)
        drag_value = rd[target_col].iloc[-1]
    except Exception:
        return None

    # 4. Build and save PyG Data object
    data = Data(pos=pos, c=c)
    data.x = cp
    data.cd = torch.tensor([drag_value], dtype=torch.float32)
    data.force = torch.tensor([float("nan")], dtype=torch.float32)
    geom_extent, geom_aspect, conditioning = _compute_geom_conditioning(pos, V_INF_DRIVAER)
    data.v_inf = torch.tensor([V_INF_DRIVAER], dtype=torch.float32)
    data.geom_extent = geom_extent
    data.geom_aspect = geom_aspect
    data.conditioning = conditioning
    data.filename = folder_name

    torch.save(data, os.path.join(output_dir, f"{folder_name}.pt"))
    return folder_name


def process_custom_drivaer(base_path, output_dir, target_col="Cd", num_workers=NUM_WORKERS):
    """
    Processes custom DrivAer run folders into PyG-compatible .pt files.
    Uses multiprocessing to speed up mesh reading and normal computation.
    """
    os.makedirs(output_dir, exist_ok=True)

    run_folders = glob.glob(os.path.join(base_path, "run_*"))
    run_folders.sort(key=lambda x: int(os.path.basename(x).split("_")[1]))
    print(f"Found {len(run_folders)} run folders. Processing with {num_workers} workers...")

    order_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_drivaer, rf, output_dir, target_col): rf for rf in run_folders}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing DrivAer runs"):
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

    parser = argparse.ArgumentParser(description="Preprocess DrivAer PLY runs (multi-head format)")
    parser.add_argument("--base-path", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/drivaer_complete")
    parser.add_argument("--output-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_multi")
    parser.add_argument("--target-col", default="Cd", help="Column name for Cd in force_mom_X.csv")
    parser.add_argument("--num-workers", type=int, default=_default_workers)
    args = parser.parse_args()

    print(f"Using {args.num_workers} workers (SLURM_CPUS_PER_TASK={_slurm_cpus})")
    process_custom_drivaer(args.base_path, args.output_dir, target_col=args.target_col, num_workers=args.num_workers)
