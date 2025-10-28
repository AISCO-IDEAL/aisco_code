import datetime
import os
import typing as T

import numpy as np
import torch
import trimesh
from sdf import SDFNetwork
from skimage import measure
from torch import nn


def sphere_gt(points: torch.Tensor, coordinates: torch.Tensor = torch.tensor([0,0,0]), radius: float = 1.0) -> torch.Tensor:
    """
    Generate signed distance ground truth values for a sphere centered at the origin.
    Args:
        points: Input coordinates of shape (batch_size, 3)
        radius: Radius of the sphere
    """
    return torch.norm(points, dim=-1) - radius

def train_sphere(radius: float = 1.0) -> SDFNetwork:
    """
    Train an SDFNetwork to represent a sphere of given radius.
    Args:
        radius: Radius of the sphere
    Returns:
        Trained SDFNetwork model
    """
    model = SDFNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 100
    batch_size = 1024
    for epoch in range(num_epochs):
        points = torch.rand(batch_size, 3) * 4.0 - 2.0  # Sample in range [-2, 2]
        sdf_values = sphere_gt(points, radius=radius)
        optimizer.zero_grad()
        preds = model(points)[0].squeeze()
        loss = nn.MSELoss()(preds, sdf_values)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    return model

def get_sdf_sphere_network(use_original: bool = False, radius: float = 1.0, save_network: bool = True,username: str="aalthaus") -> SDFNetwork | T.Callable:
    """
    Return an SDFNetwork representing a sphere of given radius.
    Args:
        use_original: If True, train a new model; if False, load a pre-trained model.
        radius: Radius of the sphere.
    """
    if use_original:
        model = train_sphere(radius=radius)
        if save_network:
            save_dir = f"/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/dummy_models"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'sdf_sphere_model_{datetime.date.today()}.pth')
            torch.save(model, save_path)
            print(f"Model saved to: {save_path}")
        return model
    else:
        save_dir = f"/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/dummy_models"
        model_path = os.path.join(save_dir, 'sdf_new_sphere_model.pth')
        model = torch.load(model_path)
        return model

def visualise_model(model: SDFNetwork, grid_size: int = 32, threshold: float = 0.0):
    x = np.linspace(-2, 2, grid_size)
    y = np.linspace(-2, 2, grid_size)
    z = np.linspace(-2, 2, grid_size)
    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
    with torch.no_grad():
        sdf_values = model(grid_points_tensor)[0].cpu().numpy().reshape((grid_size, grid_size, grid_size))
    verts, faces, normals, values = measure.marching_cubes(sdf_values, level=threshold)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    mesh.export('sdf_sphere_mesh.ply')  # Save to file
    print("Mesh saved to sdf_sphere_mesh.ply. Transfer to local machine for visualization. Use 3D Point Cloud Visualizer (PLY, ...) Extension in VS Code by Florian Nick.")


# Train and save the model
model = get_sdf_sphere_network(use_original=True, radius=1.0, save_network=True)

# Visualize the model
mesh = visualise_model(model, grid_size=32, threshold=0.0)