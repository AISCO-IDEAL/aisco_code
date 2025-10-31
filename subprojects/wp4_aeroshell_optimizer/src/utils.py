# utils.py

import json
import os
import time
from pathlib import Path

import kaolin_test
import numpy as np
import torch
import trimesh
from sdf import SDFNetwork
from skimage import measure


# In utils.py
def save_sdf2mesh(model, 
    grid_size: int =128,
    threshold: float = 0.0,
    mesh_dir: str = "AISCO-code/subprojects/wp4_aeroshell_optimizer/data",
    filename: str="sdf_mesh.ply",
    device: str='cuda'):

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



    verts,faces = kaolin_test.get_diff_mesh(model,grid_res=grid_size)
    
    if mesh_dir:
        export_path= mesh_dir
        export_path=os.path.join(export_path,filename)
        mesh=kaolin_test.flexicubes_to_PLY(verts,faces,filename=filename)
        print(f"Mesh saved to {export_path}.")
        return mesh
    return

def load_model(model_path: str) -> SDFNetwork:
    model = torch.load(model_path)
    return model


def filenamer(hyperparams: dict, general_info: dict, metadata: dict) -> str:
    # Create a filename based on current timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    filename = f"{metadata['Username']}_{general_info['Model']}_{timestamp}.json"

    # Define the full path to save the JSON file
    save_path = Path(f"/cluster/scratch/{metadata['Username']}/models")
    full_path = save_path / filename

    # Create the JSON structure
    json_data = {
        "Metadata": {
            "Username": metadata['Username'],
            "Timestamp": timestamp,
            "Filename": filename
        },
        "General Information": general_info,
        "Hyperparameters": hyperparams
    }

    # Save the JSON file
    full_path.write_text(json.dumps(json_data, indent=4))

    return filename
