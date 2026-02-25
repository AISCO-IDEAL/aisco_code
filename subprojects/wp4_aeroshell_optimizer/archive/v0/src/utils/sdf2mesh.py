import os

import numpy as np
import trimesh
from skimage import measure

from ..models.sdf import SDFNetwork
from .kaolin_utils import flexicubes_to_PLY, get_diff_mesh
from typing import Optional


def save_sdf2mesh(
    model,
    grid_size: int = 128,
    threshold: float = 0.0,
    filename: str = "sdf_mesh.ply",
    mesh_dir: Optional[str] = None,
    device: str = "cuda",
):
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

    verts, faces = get_diff_mesh(model, grid_res=grid_size,device=device)

    if mesh_dir:
        export_path = mesh_dir
        export_path = os.path.join(export_path, filename)
        mesh = flexicubes_to_PLY(verts, faces, filename=filename, mesh_dir=mesh_dir)
        print(f"Mesh saved to {export_path}.")
        return mesh
    return
