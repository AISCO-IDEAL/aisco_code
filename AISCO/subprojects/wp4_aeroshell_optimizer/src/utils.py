# utils.py

import numpy as np
import torch
import trimesh
from sdf import SDFNetwork
from skimage import measure


# In utils.py
def visualize_sdf(model, grid_size=128, threshold=0.0, filename="sdf_mesh.ply"):
    x = np.linspace(-8, 8, grid_size)
    y = np.linspace(-8, 8, grid_size)
    z = np.linspace(-8, 8, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():  # Disable gradient computation
        sdf_values = model(grid_points_tensor)[0].cpu().numpy().reshape((grid_size, grid_size, grid_size))

    verts, faces, normals, _ = measure.marching_cubes(sdf_values, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export(filename)
    print(f"Mesh saved to {filename}.")

def load_model(model_path: str) -> SDFNetwork:
    model = torch.load(model_path)
    return model
