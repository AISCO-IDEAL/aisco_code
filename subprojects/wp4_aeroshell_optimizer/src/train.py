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
    model_path: str = None,
    save_dir: str = None,
    
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
    learning_rate = float(config.get("learning_rate"))
    num_epochs = config["num_epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SDFLosses(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    for epoch in range(num_epochs):
        points = torch.randn(1000, 3, requires_grad=True,device='cuda') 
        optimizer.zero_grad()
        total_loss = loss_fn.get_total_loss(model, points)
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
    visualize_sdf(model, grid_size=64, threshold=0.0, filename="NewPipelineTest.ply")
    print("Training complete. A mesh has been generated and saved as 'sdf_aeroshell_mesh.ply'.")

    return model
