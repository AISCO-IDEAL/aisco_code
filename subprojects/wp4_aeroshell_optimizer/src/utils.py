# utils.py

import numpy as np
import torch
import trimesh
from sdf import SDFNetwork
from skimage import measure
import os
import kaolin_test
# In utils.py
def visualize_sdf(model, username = "aalthaus"
    , grid_size: int =128,
    threshold: float = 0.0, filename: str="sdf_mesh.ply",device: str='cuda'):
    """
    x = torch.linspace(-8, 8, grid_size, device= device)
    y = torch.linspace(-8, 8, grid_size, device= device)
    z = torch.linspace(-8, 8, grid_size, device= device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
    grid_points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    #is this line necessary?
    #grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    with torch.no_grad():  # Disable gradient computation
        sdf_values = model(grid_points)[0] #.cpu().numpy().reshape((grid_size, grid_size, grid_size))

    verts, faces, normals, _ = measure.marching_cubes(sdf_values, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    """
    verts,faces = kaolin_test.get_diff_mesh(model)
    mesh=kaolin_test.flexicubes_to_PLY(verts,faces,filename=filename)
    export_path= "AISCO-code/subprojects/wp4_aeroshell_optimizer/src/MainModels/"
    mesh.export(os.path.join(export_path,filename))
    print(f"Mesh saved to {export_path}/{filename}.")

def load_model(model_path: str) -> SDFNetwork:
    model = torch.load(model_path)
    return model
