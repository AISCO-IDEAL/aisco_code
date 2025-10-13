import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import für torch_geometric
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# --- 1. Synthetische Daten generieren ---
def generate_synthetic_data(num_samples=100, num_nodes=3, num_points=100):
    # Knotenfeatures: [Position (3), Orientierung (3), Volumen (1)]
    x = torch.rand(num_samples, num_nodes, 7)  # Zufällige Features
    x[:, :, 0:3] = torch.rand(num_samples, num_nodes, 3) * 2 - 1  # Position: [-1, 1]
    x[:, :, 3:6] = torch.rand(num_samples, num_nodes, 3) * np.pi - np.pi/2  # Orientierung: [-π/2, π/2]
    x[:, :, 6] = torch.rand(num_samples, num_nodes) * 0.05  # Volumen: [0, 0.05]

    # Kanten: Jedes Bauteil ist mit jedem anderen verbunden (vereinfacht)
    edge_index = torch.tensor([[0, 0, 1, 1, 2, 2],
                               [1, 2, 0, 2, 0, 1]], dtype=torch.long)

    # Punktewolken: [num_samples, num_nodes, num_points, 3]
    point_clouds = torch.randn(num_samples, num_nodes, num_points, 3) * 0.1

    return x, edge_index, point_clouds

# --- 2. GNN-Modell definieren ---
class SimpleGNN(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=32, output_dim=3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

# --- 3. Loss-Funktionen ---
def cd_loss(deformed_clouds):
    return torch.abs(deformed_clouds[..., 0]).mean()  # Frontalfläche minimieren

def solar_loss(deformed_clouds):
    return -torch.abs(deformed_clouds[..., 1]).mean()  # Seitliche Fläche maximieren

# --- 4. Training ---
def train():
    # Daten generieren
    x, edge_index, point_clouds = generate_synthetic_data()
    model = SimpleGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Trainingsschleife
    for epoch in range(50):
        optimizer.zero_grad()
        batch_deformations = []
        for i in range(x.shape[0]):
            data = Data(x=x[i], edge_index=edge_index)
            deform = model(data.x, data.edge_index)
            batch_deformations.append(deform)
        batch_deformations = torch.stack(batch_deformations)

        # Punktewolken deformieren
        deformed_clouds = point_clouds + batch_deformations.unsqueeze(2)

        # Loss berechnen
        loss_cd = cd_loss(deformed_clouds)
        loss_solar = solar_loss(deformed_clouds)
        loss = 0.5 * loss_cd + 0.5 * loss_solar  # Gewichtete Summe

        # Backpropagation
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, CD: {loss_cd.item():.4f}, Solar: {-loss_solar.item():.4f}")

    return model, deformed_clouds

# --- 5. Visualisierung ---
def plot_point_clouds(original, deformed, node_idx=0):
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[0, node_idx, :, 0], original[0, node_idx, :, 1], original[0, node_idx, :, 2], c='b')
    ax1.set_title("Original")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(deformed[0, node_idx, :, 0], deformed[0, node_idx, :, 1], deformed[0, node_idx, :, 2], c='r')
    ax2.set_title("Deformed")
    plt.show()

# --- 6. Ausführen ---
if __name__ == "__main__":
    model, deformed_clouds = train()
    x, _, point_clouds = generate_synthetic_data(num_samples=1)
    plot_point_clouds(point_clouds, deformed_clouds)
