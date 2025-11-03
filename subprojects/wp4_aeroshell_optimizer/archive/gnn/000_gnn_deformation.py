import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# --- 1. Generate Synthetic Data ---
def generate_synthetic_data(num_samples=100, num_nodes=3, num_points=100):
    """
    Generate synthetic CAD assembly data for training.

    Pre-conditions:
        - num_samples: int > 0 (number of samples to generate)
        - num_nodes: int > 0 (number of parts per assembly)
        - num_points: int > 0 (number of points per part's point cloud)

    Post-conditions:
        - Returns a tuple (x, edge_index, point_clouds) where:
            - x: torch.Tensor of shape [num_samples, num_nodes, 7] (node features: position, orientation, volume)
            - edge_index: torch.Tensor of shape [2, num_edges] (graph connectivity)
            - point_clouds: torch.Tensor of shape [num_samples, num_nodes, num_points, 3] (3D point clouds for each part)
    """
    # Node features: [position (3), orientation (3), volume (1)]
    x = torch.rand(num_samples, num_nodes, 7)
    x[:, :, 0:3] = torch.rand(num_samples, num_nodes, 3) * 2 - 1  # Position in [-1, 1]
    x[:, :, 3:6] = (
        torch.rand(num_samples, num_nodes, 3) * np.pi - np.pi / 2
    )  # Orientation in [-π/2, π/2]
    x[:, :, 6] = torch.rand(num_samples, num_nodes) * 0.05  # Volume in [0, 0.05]

    # Edges: Fully connected graph (simplified)
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long
    )

    # Point clouds: Random 3D points for each part
    point_clouds = torch.randn(num_samples, num_nodes, num_points, 3) * 0.1

    return x, edge_index, point_clouds


# --- 2. Define GNN Model ---
class SimpleGNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, output_dim=3):
        """
        Initialize a simple GNN for part deformation prediction.

        Pre-conditions:
            - input_dim: int > 0 (dimension of input node features)
            - hidden_dim: int > 0 (dimension of hidden layers)
            - output_dim: int > 0 (dimension of output deformation vectors, typically 3 for [dx, dy, dz])

        Post-conditions:
            - Initializes a GNN with two GCN layers and a linear output layer.
        """
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GNN.

        Pre-conditions:
            - x: torch.Tensor of shape [num_nodes, input_dim] (node features)
            - edge_index: torch.Tensor of shape [2, num_edges] (graph connectivity)

        Post-conditions:
            - Returns a torch.Tensor of shape [num_nodes, output_dim] (deformation vectors for each part)
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)


# --- 3. Loss Functions ---
def cd_loss(deformed_clouds):
    """
    Compute drag loss (frontal area proxy).

    Pre-conditions:
        - deformed_clouds: torch.Tensor of shape [num_samples, num_nodes, num_points, 3]

    Post-conditions:
        - Returns a scalar torch.Tensor representing the mean absolute frontal area (x-coordinate).
    """
    return torch.abs(deformed_clouds[..., 0]).mean()  # Minimize frontal area


def solar_loss(deformed_clouds):
    """
    Compute solar performance loss (maximize exposed area).

    Pre-conditions:
        - deformed_clouds: torch.Tensor of shape [num_samples, num_nodes, num_points, 3]

    Post-conditions:
        - Returns a scalar torch.Tensor in [0, 1], where higher values indicate better solar exposure.
    """
    return 1.0 / (1.0 + torch.abs(deformed_clouds[..., 1]).mean())  # Maximize side area


# --- 4. Training Loop ---
def train(num_epochs=50):
    """
    Train the GNN to optimize part deformations.

    Pre-conditions:
        - num_epochs: int > 0 (number of training epochs)

    Post-conditions:
        - Returns a tuple (model, deformed_clouds) where:
            - model: Trained SimpleGNN instance
            - deformed_clouds: torch.Tensor of shape [num_samples, num_nodes, num_points, 3] (deformed point clouds)
    """
    # Generate data
    x, edge_index, point_clouds = generate_synthetic_data()
    model = SimpleGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        batch_deformations = []

        # Process each sample
        for i in range(x.shape[0]):
            data = Data(x=x[i], edge_index=edge_index)
            deform = model(data.x, data.edge_index)
            batch_deformations.append(deform)

        batch_deformations = torch.stack(batch_deformations)
        deformed_clouds = point_clouds + batch_deformations.unsqueeze(
            2
        )  # Apply deformations

        # Compute losses
        loss_cd = cd_loss(deformed_clouds)
        loss_solar = solar_loss(deformed_clouds)
        loss = 0.5 * loss_cd + 0.5 * loss_solar  # Weighted sum

        # Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}, Loss: {loss.item():.4f}, CD: {loss_cd.item():.4f}, Solar: {loss_solar.item():.4f}"
            )

    return model, deformed_clouds


# --- 5. Visualization ---
def plot_point_clouds(original, deformed, node_idx=0):
    """
    Plot original and deformed point clouds for comparison.

    Pre-conditions:
        - original: torch.Tensor of shape [num_samples, num_nodes, num_points, 3]
        - deformed: torch.Tensor of shape [num_samples, num_nodes, num_points, 3]
        - node_idx: int (index of the part to visualize)

    Post-conditions:
        - Displays a 3D plot comparing original (blue) and deformed (red) point clouds.
    """
    fig = plt.figure(figsize=(12, 6))

    # Original point cloud
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(
        original[0, node_idx, :, 0].detach().numpy(),
        original[0, node_idx, :, 1].detach().numpy(),
        original[0, node_idx, :, 2].detach().numpy(),
        c="b",
    )
    ax1.set_title("Original")

    # Deformed point cloud
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.scatter(
        deformed[0, node_idx, :, 0].detach().numpy(),
        deformed[0, node_idx, :, 1].detach().numpy(),
        deformed[0, node_idx, :, 2].detach().numpy(),
        c="r",
    )
    ax2.set_title("Deformed")

    plt.show()


# --- 6. Execute ---
if __name__ == "__main__":
    model, deformed_clouds = train(num_epochs=50)
    x, _, point_clouds = generate_synthetic_data(num_samples=1)
    plot_point_clouds(point_clouds, deformed_clouds)
