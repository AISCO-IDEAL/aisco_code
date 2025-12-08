from datetime import datetime

import numpy as np
import pandas as pd
import torch
import trimesh
from pvlib import solarposition


# ============================================================
#  SUN.PY  (unchanged)
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

    zenith_angle = np.pi/2 - elevation
    
    x = np.sin(zenith_angle) * np.sin(azimuth)
    y = np.sin(zenith_angle) * np.cos(azimuth)
    z = np.cos(zenith_angle)

    return np.array([x, y, z])


# ============================================================
#  READ_MESH.PY (DIFFERENTIABLE VERSION)
# ============================================================

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32)

def _compute_face_normals(vertices, faces):
    v0 = vertices[faces[:,0]]
    v1 = vertices[faces[:,1]]
    v2 = vertices[faces[:,2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    n = n / (torch.norm(n, dim=-1, keepdim=True) + 1e-9)
    return n

def facenormal_angle(mesh: trimesh.Trimesh, sun_direction=np.array([0,0,1])):
    V = _to_tensor(mesh.vertices)
    F = torch.tensor(mesh.faces, dtype=torch.long)

    s = _to_tensor(sun_direction)
    s = s / (torch.norm(s) + 1e-9)

    normals = _compute_face_normals(V, F)
    cosang = torch.sum(normals * s[None, :], dim=-1)
    cosang = torch.clamp(cosang, min=0.0, max=1.0)

    ang = torch.acos(cosang) * 180.0 / np.pi
    return ang

def powerratio_iam(theta_grad, b0=0.05):
    theta = _to_tensor(theta_grad)
    theta_rad = theta * np.pi / 180.0
    cos_t = torch.cos(theta_rad)
    cos_t = torch.clamp(cos_t, min=1e-6)

    iam = 1.0 - b0 * (1.0 / cos_t - 1.0)

    sharpness = 40.0
    soft_cut = torch.sigmoid(-(theta - 90.0) * sharpness)

    return iam * soft_cut

def sky_face(mesh, threshold=0.01):
    V = _to_tensor(mesh.vertices)
    F = torch.tensor(mesh.faces, dtype=torch.long)
    normals = _compute_face_normals(V, F)

    sharp = 50.0
    soft_mask = torch.sigmoid((normals[:,2] - threshold) * sharp)

    return mesh, soft_mask

def shadowing_factor(mesh: trimesh.Trimesh, sun_direction=np.array([0,0,-1])):
    V = _to_tensor(mesh.vertices)
    F = torch.tensor(mesh.faces, dtype=torch.long)
    s = _to_tensor(sun_direction)
    s = s / (torch.norm(s) + 1e-9)

    normals = _compute_face_normals(V, F)
    vis = torch.clamp(torch.sum(normals * (-s)[None,:], dim=-1), min=0.0)
    shadow = torch.sigmoid(10.0 * vis)

    return shadow


# ============================================================
#  PV_INDEX_FOR_RUN.PY (DIFFERENTIABLE VERSION)
# ============================================================

def _soft_top_area(areas, weights, max_area_mm2):
    temp = 0.05
    soft_sel = torch.softmax(weights / temp, dim=0)
    total_area = torch.sum(soft_sel * areas)
    scale = max_area_mm2 / (total_area + 1e-9)
    soft_sel = soft_sel * scale
    soft_sel = torch.clamp(soft_sel, max=1.0)

    return soft_sel

def pv_perf_index(top_mesh, periodic_power_weights, max_area=6.0):
    max_area_mm2 = max_area * 1e6

    pw = torch.tensor(periodic_power_weights.to_numpy(), dtype=torch.float32)
    avg_w = torch.mean(pw, dim=1)

    areas = torch.tensor(top_mesh.area_faces, dtype=torch.float32)

    soft_mask = _soft_top_area(areas, avg_w, max_area_mm2)

    total = torch.sum(areas * avg_w * soft_mask)
    performance_index = total / max_area_mm2

    selected = (soft_mask > 0.5).cpu().numpy()
    selected_faces = np.where(selected)[0]

    vis_mesh = trimesh.Trimesh(
        vertices=top_mesh.vertices,
        faces=top_mesh.faces[selected_faces]
    )

    return float(performance_index), vis_mesh


# ============================================================
#  MAIN (SELF-CONTAINED EXAMPLE)
# ============================================================

def evaluate_solar_mesh(stl_path, lat, lon, altitude, times_utc):
    mesh = trimesh.load_mesh(stl_path)

    top_mesh, soft_sky = sky_face(mesh)
    F = len(top_mesh.faces)

    res = pd.DataFrame(index=np.arange(F))

    for i, t in enumerate(times_utc):
        sun_dir = get_sun_direction(lat, lon, altitude, t)

        ang = facenormal_angle(top_mesh, sun_dir)
        shadows = shadowing_factor(top_mesh, -1 * sun_dir)
        power = powerratio_iam(ang)

        power = power * (1.0 - shadows)
        power = power * soft_sky

        res[f"time_{i}"] = power.detach().cpu().numpy()

    perf_idx, best_mesh = pv_perf_index(top_mesh, res)
    return perf_idx, best_mesh

  

def evaluate_from_points_csv(stl_path, points_csv_path):

    # -------------------------------------------------------
    # 1. Load race route points.csv (your exact code)
    # -------------------------------------------------------
    df = pd.read_csv(points_csv_path)

    df['time'] = pd.to_datetime(
        df['time'],
        format='%Y-%m-%d %H:%M:%S'
    )
    df_time_acst = df['time'].dt.tz_localize('Australia/Darwin', ambiguous='infer')
    df['time_utc'] = df_time_acst.dt.tz_convert('UTC')

    # -------------------------------------------------------
    # 2. Load mesh
    # -------------------------------------------------------
    mesh = trimesh.load_mesh(stl_path)

    # Extract sky-facing mesh + soft sky weights
    top_mesh, soft_sky = sky_face(mesh)
    F = len(top_mesh.faces)

    # DataFrame for per-face power for each timestamp
    res = pd.DataFrame(index=np.arange(F))

    # -------------------------------------------------------
    # 3. Loop through each race route point
    # -------------------------------------------------------
    for i, row in df.iterrows():

        sun_dir = get_sun_direction(
            row['Lat'],
            row['Long'],
            row['altitude'],
            row['time_utc']
        )

        # Compute angles, shadowing, IAM-based power
        angles = facenormal_angle(top_mesh, sun_dir)
        shadows = shadowing_factor(top_mesh, -1 * sun_dir)
        power = powerratio_iam(angles)

        # Apply soft sky-facing mask and shadow mask
        power = power * (1.0 - shadows) * soft_sky

        # Convert torch tensor to numpy for storage
        res[f"time_{i}"] = power.detach().cpu().numpy()

    # -------------------------------------------------------
    # 4. Compute final performance index and best mesh region
    # -------------------------------------------------------
    perf_idx, best_mesh = pv_perf_index(top_mesh, res)

    return perf_idx, best_mesh

if __name__ == "__main__":
    perf, best_mesh = evaluate_from_points_csv(
        stl_path = r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\stl_models\SilvrettaAeroskin.stl",
        points_csv_path= r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\race_routes\points.csv"

    )

    print("Performance index:", perf)

    full_mesh = trimesh.load_mesh(r"C:\Users\mail\OneDrive - ETH Zurich\ETH\Aisco\AISCO-code\AISCO\subprojects\WP3_Solarestimator\data\stl_models\SilvrettaAeroskin.stl",
)

    # Get the sky-facing mesh (for green visualization)
    _, soft_sky = sky_face(full_mesh)
    sky_mask = (soft_sky > 0.5).cpu().numpy()
    sky_faces = np.where(sky_mask)[0]
    sky_mesh = trimesh.Trimesh(
        vertices=full_mesh.vertices,
        faces=full_mesh.faces[sky_faces]
    )
    sky_mesh.visual.face_colors = [0, 255, 0, 120]   # green transparent

    # Best mesh is already constructed by pv_perf_index
    best_mesh.visual.face_colors = [255, 0, 0, 150]  # red transparent

    # Create scene and show
    scene = trimesh.Scene([sky_mesh, best_mesh])
    scene.show(notebook=True)



