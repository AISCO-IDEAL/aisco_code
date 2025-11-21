import numpy as np
import pandas as pd
import torch
import trimesh
from pvlib import solarposition
from datetime import datetime


# ============================================================
# Utility
# ============================================================

def to_tensor(x, device="cpu"):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# ============================================================
# Differentiable Geometry
# ============================================================

def compute_face_normals(vertices, faces):
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-12)
    return n


# ============================================================
# Sun direction (non-diff but does not block graph)
# ============================================================

def get_sun_direction(latitude, longitude, altitude, time_utc):
    solpos = solarposition.get_solarposition(
        time_utc,
        latitude,
        longitude,
        altitude=altitude
    )

    azimuth = np.deg2rad(solpos['azimuth'].iloc[0])
    elevation = np.deg2rad(solpos['elevation'].iloc[0])
    zenith = np.pi/2 - elevation

    x = np.sin(zenith) * np.sin(azimuth)
    y = np.sin(zenith) * np.cos(azimuth)
    z = np.cos(zenith)

    # RETURN TORCH
    return to_tensor([x, y, z])


# ============================================================
# Differentiable solar components
# ============================================================

def facenormal_angle(vertices, faces, sun_dir):
    normals = compute_face_normals(vertices, faces)
    s = sun_dir / (torch.norm(sun_dir) + 1e-12)
    cosang = torch.sum(normals * s[None,:], dim=-1)
    cosang = torch.clamp(cosang, 0.0, 1.0)
    ang = torch.acos(cosang) * 180.0 / np.pi
    return ang  # (F,)


def powerratio_iam(theta_deg, b0=0.05):
    theta_rad = theta_deg * np.pi / 180.0
    cos_t = torch.cos(theta_rad)
    cos_t = torch.clamp(cos_t, min=1e-6)
    iam = 1.0 - b0 * (1.0 / cos_t - 1.0)

    # Soft cutoff at 90 degrees
    soft = torch.sigmoid(-(theta_deg - 90.0) * 40.0)
    return iam * soft


def sky_face(vertices, faces, threshold=0.01):
    normals = compute_face_normals(vertices, faces)
    soft = torch.sigmoid((normals[:,2] - threshold) * 50.0)
    return soft  # (F,)


def shadowing_factor(vertices, faces, sun_dir):
    normals = compute_face_normals(vertices, faces)
    s = sun_dir / (torch.norm(sun_dir) + 1e-12)
    vis = torch.clamp(torch.sum(normals * (-s)[None,:], dim=-1), min=0.0)
    shadow = torch.sigmoid(10.0 * vis)
    return shadow  # (F,)


# ============================================================
# Differentiable soft top-area selection
# ============================================================

def soft_top_area(areas, weights, max_area_mm2):
    """
    Differentiable approximation of selecting the top faces by power
    until the selected area reaches max_area_mm2.

    This version uses a sharp sigmoid gate to mimic a hard threshold,
    making the selection region visible and the performance index realistic.
    """

    # --------------------------------------------------------
    # 1. Compute adaptive threshold τ as the median of weights
    # --------------------------------------------------------
    tau = torch.median(weights)

    # --------------------------------------------------------
    # 2. Sharp sigmoid -> behaves like a differentiable step-function
    #    Faces with weight > τ -> mask ~ 1
    #    Faces with weight < τ -> mask ~ 0
    # --------------------------------------------------------
    sharpness = 25.0   # Increase if needed
    base_mask = torch.sigmoid((weights - tau) * sharpness)

    # --------------------------------------------------------
    # 3. Scale so total selected area ≈ target 6 m²
    # --------------------------------------------------------
    total_area = torch.sum(base_mask * areas)
    scale = max_area_mm2 / (total_area + 1e-12)

    # Mask scaled but kept in [0,1]
    mask = torch.clamp(base_mask * scale, max=1.0)

    return mask



# ============================================================
# Differentiable performance index
# ============================================================

def pv_perf_index(areas, per_face_power, max_area=6.0):
    max_area_mm2 = max_area * 1e6

    # per_face_power: (F, T)
    avg_w = per_face_power.mean(dim=1)  # (F,)

    soft_mask = soft_top_area(areas, avg_w, max_area_mm2)

    numerator = torch.sum(areas * avg_w * soft_mask)
    performance = numerator / max_area_mm2

    return performance, soft_mask


# ============================================================
# EVALUATOR (DIFFERENTIABLE)
# ============================================================

def evaluate_solar_mesh_diff(vertices, faces, lat, lon, alt, times_utc):
    device = vertices.device

    F = faces.shape[0]
    per_face_power_list = []

    sky_mask = sky_face(vertices, faces)

    for t in times_utc:
        sun_dir = get_sun_direction(lat, lon, alt, t).to(device)

        ang = facenormal_angle(vertices, faces, sun_dir)
        shadow = shadowing_factor(vertices, faces, -sun_dir)
        iam = powerratio_iam(ang)

        power = iam * (1.0 - shadow) * sky_mask
        per_face_power_list.append(power)

    per_face_power = torch.stack(per_face_power_list, dim=1)  # (F,T)

    # Areas: trimesh is nondiff → recompute with torch
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=-1)

    perf, softmask = pv_perf_index(areas, per_face_power)
    return perf, softmask, per_face_power


# ============================================================
# NON-DIFF DRIVER THAT LOADS STL + CSV AND CALLS THE ABOVE
# ============================================================

def evaluate_from_points_csv(stl_path, points_csv_path):
    # Load CSV (this is safe because the diff graph is inside evaluate_solar_mesh_diff)
    df = pd.read_csv(points_csv_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df_time_acst = df['time'].dt.tz_localize('Australia/Darwin', ambiguous='infer')
    df['time_utc'] = df_time_acst.dt.tz_convert('UTC')

    # Load mesh
    base_mesh = trimesh.load_mesh(stl_path)
    vertices_np = base_mesh.vertices.astype(np.float32)
    faces_np = base_mesh.faces.astype(np.int64)

    # Move to torch
    device = "cpu"
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device, requires_grad=True)
    faces = torch.tensor(faces_np, dtype=torch.long, device=device)

    # Build time list
    times = df['time_utc'].tolist()
    lat = df['Lat'].iloc[0]
    lon = df['Long'].iloc[0]
    alt = df['altitude'].iloc[0]

    perf, softmask, power = evaluate_solar_mesh_diff(vertices, faces, lat, lon, alt, times)

    # For visualization: pick faces with softmask > 0.5
    selected = (softmask.detach().cpu().numpy() > 0.5)
    selected_faces = np.where(selected)[0]

    vis_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np[selected_faces]
    )

    return perf.item(), vis_mesh


# ============================================================
# MAIN (UNCHANGED AS REQUESTED)
# ============================================================
def evaluate_single_timestamp(stl_path, latitude, longitude, altitude, timestamp_utc):
    """
    Compute the PV performance index of the solar mesh for ONE specific
    sun position (one location + one timestamp).
    """

    # -------------------------
    # Load mesh (non-diff)
    # -------------------------
    mesh = trimesh.load_mesh(stl_path)
    vertices_np = mesh.vertices.astype(np.float32)
    faces_np = mesh.faces.astype(np.int64)

    # Convert to torch (diff starts here)
    device = "cpu"
    vertices = torch.tensor(vertices_np, dtype=torch.float32, device=device, requires_grad=True)
    faces = torch.tensor(faces_np, dtype=torch.long, device=device)

    # -------------------------
    # Compute sun direction
    # -------------------------
    sun_dir = get_sun_direction(latitude, longitude, altitude, timestamp_utc).to(device)

    # -------------------------
    # Compute sky mask, angles, shadowing, IAM
    # -------------------------
    sky_mask = sky_face(vertices, faces)
    angles = facenormal_angle(vertices, faces, sun_dir)
    shadow = shadowing_factor(vertices, faces, -sun_dir)
    iam = powerratio_iam(angles)

    # Power per face (shape F,1 to match pv_perf_index)
    per_face_power = (iam * (1 - shadow) * sky_mask).unsqueeze(1)

    # -------------------------
    # Face areas (computed in torch for gradients)
    # -------------------------
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=-1)

    # -------------------------
    # Performance index
    # -------------------------
    perf, softmask = pv_perf_index(areas, per_face_power)

    # -------------------------
    # Visualization mesh
    # -------------------------
    selected = (softmask.detach().cpu().numpy() > 0.5)
    selected_faces = np.where(selected)[0]

    vis_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np[selected_faces]
    )

    return perf.item(), softmask, vis_mesh

"""if __name__ == "__main__":

    perf, best_mesh = evaluate_from_points_csv(
        stl_path = r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\stl_models\SilvrettaAeroskin.stl",
        points_csv_path= r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\race_routes\points.csv"
    )

    print("Performance index:", perf)

    # Full mesh for sky visualization
    full_mesh = trimesh.load_mesh(
        r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\stl_models\SilvrettaAeroskin.stl"
    )

    # Sky faces (green) — safe to detach
    vertices_np = full_mesh.vertices.astype(np.float32)
    faces_np = full_mesh.faces.astype(np.int64)

    V = torch.tensor(vertices_np)
    F = torch.tensor(faces_np)
    sky_mask = sky_face(V, F)
    sky_faces = (sky_mask > 0.5).nonzero()[:,0].numpy()

    sky_mesh = trimesh.Trimesh(
        vertices=vertices_np,
        faces=faces_np[sky_faces]
    )
    sky_mesh.visual.face_colors = [0,255,0,120]

    # Best mesh (red)
    best_mesh.visual.face_colors = [255,0,0,150]

    scene = trimesh.Scene([sky_mesh, best_mesh])
    scene.show(notebook = True)
"""

if __name__ == "__main__":
    import pandas as pd

    # ONE specific test time (UTC)
    test_time = pd.to_datetime("2023-08-20 04:00:00", utc=True)

    stl_path = r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\stl_models\SilvrettaAeroskin.stl"

    # Choose any location you want to test
    latitude  = -23.700
    longitude = 133.870
    altitude  = 600

    perf, softmask, vis_mesh = evaluate_single_timestamp(
        stl_path,
        latitude,
        longitude,
        altitude,
        test_time
    )

    print("===== Single Timestamp PV Index Test =====")
    print("Timestamp:", test_time)
    print("Performance index:", perf)
    print("Max mask value:", float(softmask.max()))
    print("Faces selected:", int((softmask > 0.5).sum()))

    # Optional visualization
    # vis_mesh.show()
