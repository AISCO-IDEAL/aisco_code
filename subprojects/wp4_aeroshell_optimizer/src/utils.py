# utils.py

import json
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
from sdf import SDFNetwork
from skimage import measure


# In utils.py
def visualize_sdf(model, username = "aalthaus", grid_size=128, threshold=0.0, filename="sdf_mesh.ply"):
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
    export_path = Path(f"/cluster/home/{username}/AISCO-code/subprojects/wp4_aeroshell_optimizer/src/MainModels")
    mesh.export(str(export_path / filename))
    print(f"Mesh saved to {export_path / filename}.")

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
