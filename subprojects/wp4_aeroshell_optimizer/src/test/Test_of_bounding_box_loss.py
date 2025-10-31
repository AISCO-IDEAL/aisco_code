
import torch
from losses import SDFLosses
from sdf import SDFNetwork
from /cluster/home/clhuber/AISCO-code/subprojects/wp4_aeroshell_optimizer/src/utils/SDF_visualisation.py import SDFVisualizer

""" Test script for Bounding Box Loss function
Trains the SDFNEtwork on just the bounding box loss and visualizes the result using SDF_visualizer
"""

if __name__ == "__main__":
    # Initialize SDF Network
    model = SDFNetwork()

    # Initialize Losses with only bounding box loss weight set
    losses = SDFLosses(weights=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Define optimizerpip
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = losses.bounding_boxes_loss(model)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Bounding Box Loss: {loss.item()}")

    # Visualize the result using SDFVisualizer
    visualizer = SDFVisualizer(model)
    visualizer.visualize_slice(direction='xy', slice_index=50)
