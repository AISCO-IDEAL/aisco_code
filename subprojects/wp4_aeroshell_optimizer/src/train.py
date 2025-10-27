from sdf import SDFNetwork
import torch
import typing as T
from losses import SDFLosses
import datetime
import os
import numpy as np
from utils import visualize_sdf

def train_sdf_model(
    model: SDFNetwork,
    config: T.Dict,
    point_cloud: torch.Tensor = None,
    load_pretrained_model: bool = False,
    model_path: str = None,
    save_dir: str = None,
    stl_file_path: str = None,
) -> SDFNetwork:
    """
    Train the given SDFNetwork model based on the provided configuration.
    Pre-conditions:
        - model: An instance of SDFNetwork, already initialized and ready for training.
        - config: A dictionary containing at least the following keys:
            - learning_rate: Positive float (default: 1e-3).
            - num_epochs: Positive integer (default: 1000).
            - batch_size: Positive integer (default: 1024).
            - (Optional) Any additional parameters required by SDFLosses or the training process.
    Post-conditions:
        - The model's parameters are updated via gradient descent for num_epochs iterations.
        - The loss is printed every 10 epochs.
        - The trained model is saved and returned.
    """
    # Load pre-trained model if specified
    if load_pretrained_model and model_path:
        model = torch.load(model_path)
        print(f"Model loaded from: {model_path}")

    # Generate point cloud if STL file is provided
    if stl_file_path:
        point_cloud = stl_to_point_cloud(stl_file_path)
        point_cloud = torch.from_numpy(point_cloud).float().requires_grad_(True)
        np.save('shell0_point_cloud.npy', point_cloud.detach().numpy())
        print(f"Point cloud generated and saved as 'shell0_point_cloud.npy' with {len(point_cloud)} points.")

    # Initialize optimizer and loss function
    learning_rate = config.get("learning_rate", 1e-3)
    num_epochs = config.get("num_epochs", 1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SDFLosses(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = loss_fn.get_total_loss(model, point_cloud)
        total_loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}")

    # Save the trained model
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        datum = datetime.datetime.now()
        save_path = os.path.join(save_dir, f'sdf_model_{datum.strftime("%m_%d_%H_%M")}.pth')
        torch.save(model, save_path)
        print(f"Model saved to: {save_path}")

    # Visualize the SDF
    visualize_sdf(model, grid_size=32, threshold=0.0, filename="sdf_aeroshell_mesh_BiggerSphereStart.ply")
    print("Training complete. A mesh has been generated and saved as 'sdf_aeroshell_mesh.ply'.")

    return model
