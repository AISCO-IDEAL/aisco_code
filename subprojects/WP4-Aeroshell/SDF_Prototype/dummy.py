#this file provides an SDF model that generates a dummy object like a sphere or cube for testing purposes
import torch
import torch.nn as nn
import typing as T
import datetime
from sdf import SDFNetwork
import os

def sphere_gt(points: torch.Tensor, coordinates: torch.Tensor = torch.tensor([0,0,0]), radius: float = 1.0) -> torch.Tensor:
    """
    Generate signed distance ground truth values for a sphere centered at the origin.
    
    Args:
        points: Input coordinates of shape (batch_size, 3)
        radius: Radius of the sphere
    """

    return torch.norm(points, dim=-1) - radius


def train_sphere(radius: float = 1.0) -> SDFNetwork:

    #Unfixed: Adjust so spheres outside the 2x2 box work as well and so coordinates can be adjusted

    """
    Train an SDFNetwork to represent a sphere of given radius.
    
    Args:
        0 < radius <= 2.0 : Radius of the sphere 
    Returns:
        Trained SDFNetwork model
    """
    model = SDFNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 600
    batch_size = 1024

    for epoch in range(num_epochs):
        # Sample random points in 3D space
        points = torch.rand(batch_size, 3) * 4.0 - 2.0  # Sample in range [-2, 2]
        sdf_values = sphere_gt(points, radius=radius)

        optimizer.zero_grad()
        preds = model(points).squeeze()
        loss = nn.MSELoss()(preds, sdf_values)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model

def get_sdf_sphere_network(use_original: bool = False, radius: float = 1.0, save_network = True) -> T.Union[SDFNetwork, T.Callable]:

    """
    
    Return an SDFNetwork representing a sphere of given radius.
    Args:
        use_original: If True, train a new model based on our main structure; if False, load a pre-trained simpler dummy model.
        radius: Radius of the sphere to be represented by the SDFNetwork, used if use_original = True.
    
    """

    if use_original:
        model= train_sphere(radius=radius)
        if save_network:
            save_dir = "subprojects\WP4-Aeroshell\SDF_Prototype\dummy_models"  # Define a proper save directory
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'sdf_sphere_model_{datetime.date.today()}.pth')
            torch.save(model, save_path)
        return model
    else: 
        model=torch.load('subprojects\WP4-Aeroshell\SDF_Prototype\dummy_models\sdf_sphere_model.pth')
        return model


#Used this line to generate basic sphere model
#get_sdf_sphere_network(use_original=True, radius=1.0, save_network=True)