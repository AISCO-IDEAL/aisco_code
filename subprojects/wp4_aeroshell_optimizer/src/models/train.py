import datetime
import os
import typing as T

import numpy as np
import torch
import wandb

from ..models.sdf import SDFNetwork
from ..utils.losses import SDFLosses
from ..utils.sdf2mesh import save_sdf2mesh


def train_sdf_model(
    model: SDFNetwork,
    train_config: dict,
    model_path: str = None,
    mesh_path: str = None,
    model_name: str = "sdf_model",
    mesh_name: str = "sdf_mesh",
    track_run: bool = False
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

    # Initialize optimizer and loss function
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    num_epochs = train_config["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SDFLosses(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    for epoch in range(num_epochs):
        points = torch.randn(1000, 3, requires_grad=True, device="cuda")
        optimizer.zero_grad()
        total_loss = loss_fn.get_total_loss(model=model, points=points, epoch_name=epoch)
        total_loss.backward()
        optimizer.step()
        if track_run:
            wandb.log({"epoch": epoch, "total_loss": total_loss.item()})
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}")

    # Save the trained model
    if model_path:
        os.makedirs(model_path, exist_ok=True)
        datum = datetime.datetime.now()
        model_path = os.path.join(
            model_path, f"{model_name}_{datum.strftime('%m_%d_%H_%M')}.pth"
        )
        torch.save(model, model_path)
        print(f"Model saved to: {model_path}")

    # Visualize the SDF
    if mesh_path:
        os.makedirs(mesh_path, exist_ok=True)
        datum = datetime.datetime.now()
        save_sdf2mesh(
            model,
            grid_size=128,
            threshold=0.0,
            mesh_dir=mesh_path,
            filename=f"{mesh_name}_{datum.strftime('%m_%d_%H_%M')}.ply",
            device="cuda",
        )
        print(f"A mesh has been generated and saved as '{mesh_name}.ply'.")

    return model
