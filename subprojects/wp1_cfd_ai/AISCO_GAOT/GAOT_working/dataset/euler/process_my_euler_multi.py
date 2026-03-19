import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pyvista as pv
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data
from tqdm import tqdm

# Processes custom Euler/Fluent run data into PyG-compatible .pt files.
# Expects:
#   - .ply geometry files in GEOMETRY/ folder (e.g. RUN_001.ply)
#   - force report .out files in RUN/EULER_OUTPUT/ folder (e.g. RUN_001_force_report.out)
#   - Fluent result files in RUN/RESULTS/ folder (e.g. RUN_001_results.cas.h5, .dat.h5)
#
# The force report has 3 header lines, then space-separated columns:
#   Iteration  drag_coefficient  drag_force
# We extract the last row's drag_force (3rd column).
#
# Saves .pt files containing:
#   pos   (N,3)  -- vertex coordinates (meters, nose at -x)
#   c     (N,3)  -- outward surface normals (x-negated)
#   x     (N,1)  -- surface pressure (Pa, interpolated from Fluent HDF5)
#   force (1,)   -- drag force in Newtons (positive)
#   cd    (1,)   -- NaN (not available from Euler force reports)
#   filename str

NUM_WORKERS = 8  # number of parallel workers for mesh processing
SURFACE_ZONE_NAME = "car_surface"
MIN_FORCE_COLS = 3  # minimum columns expected in a force report line
MM_TO_M_THRESHOLD = 50  # if any coordinate exceeds this (mm), convert to meters
V_INF_EULER = 25.0  # m/s
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


def _extract_surface_pressure(cas_path, dat_path, zone_name=SURFACE_ZONE_NAME):
    """
    Extract surface pressure from Fluent HDF5 results and return vertex
    positions and interpolated pressure values for the given zone.

    Returns
    -------
    pos_np : np.ndarray [N, 3]  -- surface vertex coordinates (meters)
    pressure_np : np.ndarray [N] -- static pressure at each vertex (Pa)
    """
    # 1. Parse zone topology
    with h5py.File(cas_path, "r") as cas:
        zt = cas["meshes/1/faces/zoneTopology"]
        names_raw = zt["name"][0]
        if isinstance(names_raw, bytes):
            names_raw = names_raw.decode()
        names = names_raw.split(";")

        if zone_name not in names:
            return None, None
        idx = names.index(zone_name)

        zone_start = int(zt["minId"][idx]) - 1  # 0-based
        zone_end = int(zt["maxId"][idx])
        zone_end - zone_start

        # 2. Face-to-node connectivity for the target zone
        nnodes = cas["meshes/1/faces/nodes/1/nnodes"][:]
        nodes_flat = cas["meshes/1/faces/nodes/1/nodes"][:]

    # Build cumulative offset into flat node array
    offsets = np.zeros(len(nnodes) + 1, dtype=np.int64)
    np.cumsum(nnodes, out=offsets[1:])

    face_nodes = []
    for fi in range(zone_start, zone_end):
        s = offsets[fi]
        e = offsets[fi + 1]
        face_nodes.append(nodes_flat[s:e] - 1)  # 1-based → 0-based

    # 3. Unique surface nodes → coordinates
    unique_global_ids = np.unique(np.concatenate(face_nodes))
    global_to_local = {gid: lid for lid, gid in enumerate(unique_global_ids)}
    n_vertices = len(unique_global_ids)

    with h5py.File(cas_path, "r") as cas:
        all_coords = cas["meshes/1/nodes/coords/1"][:]
    pos_np = all_coords[unique_global_ids].astype(np.float32)

    # 4. Extract face-center pressure and interpolate to vertices
    with h5py.File(dat_path, "r") as dat:
        face_pressure_all = dat["results/1/phase-1/faces/SV_P/1"][:]
    face_pressure = face_pressure_all[zone_start:zone_end]

    # Simple averaging: each face contributes equally to its vertices
    accum = np.zeros(n_vertices, dtype=np.float64)
    weight = np.zeros(n_vertices, dtype=np.float64)
    for fi, nodes in enumerate(face_nodes):
        val = face_pressure[fi]
        w = 1.0 / len(nodes)
        for n in nodes:
            lid = global_to_local[n]
            accum[lid] += val * w
            weight[lid] += w
    mask = weight > 0
    accum[mask] /= weight[mask]

    return pos_np, accum.astype(np.float32)


def _process_single_euler(ply_file, force_dir, results_dir, output_dir):
    """Process a single Euler run. Returns save_name on success, None on failure."""
    basename = os.path.basename(ply_file)  # e.g. RUN_001.ply
    run_id = basename.replace(".ply", "")  # e.g. RUN_001

    match = re.search(r"RUN_(\d+)", run_id)
    run_number = match.group(1) if match else None
    force_file = os.path.join(force_dir, f"RUN_{run_number}_force_report.out") if run_number else None
    if not run_number or force_file is None or not os.path.exists(force_file):
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

    # 2. Interpolate surface pressure from Fluent HDF5 onto PLY mesh
    cas_path = os.path.join(results_dir, f"RUN_{run_number}_results.cas.h5")
    dat_path = os.path.join(results_dir, f"RUN_{run_number}_results.dat.h5")
    cp_tensor = torch.full((pos.shape[0], 1), float("nan"), dtype=torch.float32)

    if os.path.exists(cas_path) and os.path.exists(dat_path):
        try:
            hdf5_pos, hdf5_pressure = _extract_surface_pressure(cas_path, dat_path)
            if hdf5_pos is not None and hdf5_pressure is not None:
                # HDF5 mesh is in meters with nose at -x (same as our convention)
                tree = cKDTree(hdf5_pos)
                # Query with our PLY mesh (already in meters, x-negated)
                _, idx = tree.query(pos.numpy(), k=1)
                cp_np = hdf5_pressure[idx]
                cp_tensor = torch.tensor(cp_np, dtype=torch.float32).unsqueeze(-1)
        except Exception:
            pass  # cp_tensor stays NaN

    # 3. Parse force report
    # Fluent force monitor uses drag direction = -x, so reported value is negative.
    # Negate so data.force = positive drag in +x (flow) direction.
    try:
        drag_force_raw = parse_force_report(force_file)
    except Exception:
        return None
    if drag_force_raw is None:
        return None
    drag_force = -drag_force_raw  # negate: negative Fluent report → positive drag

    # 4. Build and save PyG Data object
    data = Data(pos=pos, c=c)
    data.x = cp_tensor
    data.force = torch.tensor([drag_force], dtype=torch.float32)  # [N], positive = drag in +x
    data.cd = torch.tensor([float("nan")], dtype=torch.float32)  # not available for Euler
    geom_extent, geom_aspect, conditioning = _compute_geom_conditioning(pos, V_INF_EULER)
    data.v_inf = torch.tensor([V_INF_EULER], dtype=torch.float32)
    data.geom_extent = geom_extent
    data.geom_aspect = geom_aspect
    data.conditioning = conditioning
    data.filename = run_id

    run_num_int = int(run_number)
    save_name = f"run_{run_num_int}"
    torch.save(data, os.path.join(output_dir, f"{save_name}.pt"))
    return save_name


def process_euler_runs(geometry_dir, force_dir, results_dir, output_dir, num_workers=NUM_WORKERS):
    """
    Processes Euler/Fluent run data into PyG-compatible .pt files.
    Uses multiprocessing to speed up mesh reading and normal computation.
    """
    os.makedirs(output_dir, exist_ok=True)

    ply_files = sorted(glob.glob(os.path.join(geometry_dir, "RUN_*.ply")))
    print(f"Found {len(ply_files)} .ply files. Processing with {num_workers} workers...")

    order_list = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_single_euler, pf, force_dir, results_dir, output_dir): pf for pf in ply_files}
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

    # Auto-detect workers from SLURM; fall back to NUM_WORKERS constant
    _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    _default_workers = max(1, _slurm_cpus - 4) if _slurm_cpus > 0 else NUM_WORKERS

    parser = argparse.ArgumentParser(description="Preprocess Euler PLY runs (multi-head format)")
    parser.add_argument("--geometry-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/GEOMETRY")
    parser.add_argument("--force-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/EULER_OUTPUT")
    parser.add_argument("--results-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/FLUENT/RUN/RESULTS")
    parser.add_argument("--output-dir", default="/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_multi")
    parser.add_argument("--num-workers", type=int, default=_default_workers)
    args = parser.parse_args()

    print(f"Using {args.num_workers} workers (SLURM_CPUS_PER_TASK={_slurm_cpus})")
    process_euler_runs(args.geometry_dir, args.force_dir, args.results_dir, args.output_dir, args.num_workers)
