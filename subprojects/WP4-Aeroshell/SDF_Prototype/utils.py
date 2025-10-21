# utils.py
import torch
import numpy as np
from skimage import measure
import trimesh
import os
import datetime
from sdf import SDFNetwork

# In sdf_utils.py
def visualize_sdf(model, grid_size=32, threshold=0.0, filename="sdf_mesh.ply"):
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    z = np.linspace(-2, 2, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():  # Disable gradient computation
        sdf_values = model(grid_points_tensor)[0].cpu().numpy().reshape((grid_size, grid_size, grid_size))

    verts, faces, normals, _ = measure.marching_cubes(sdf_values, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(filename)
    print(f"Mesh saved to {filename}.")

def save_model(model: SDFNetwork, save_dir: str = "/cluster/home/eschaerer/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/main_models"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sdf_model_{datetime.date.today()}.pth')
    torch.save(model, save_path)
    print(f"Model saved to: {save_path}")
    return save_path

def load_model(model_path: str) -> SDFNetwork:
    model = torch.load(model_path)
    return model