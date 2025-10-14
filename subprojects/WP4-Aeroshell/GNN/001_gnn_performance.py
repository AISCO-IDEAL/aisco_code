import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

def generate_synthetic_data(num_samples=100, num_nodes=3):
    x = torch.rand(num_samples, num_nodes, 9)
    x[:, :, 0:3] = torch.rand(num_samples, num_nodes, 3) * 2 - 1  # Position
    x[:, :, 3:6] = torch.rand(num_samples, num_nodes, 3) * np.pi - np.pi/2  # Orientation
    x[:, :, 6] = torch.rand(num_samples, num_nodes) * 0.05  # Volume
    x[:, :, 7] = torch.rand(num_samples, num_nodes) * 1500 + 250  # Solar output (Watts)
    x[:, :, 8] = torch.rand(num_samples, num_nodes) * 0.45 + 0.05  # Local Cd contribution
    edge_index = torch.combinations(torch.arange(num_nodes), 2).t()
    edge_index = edge_index.repeat(num_samples, 1)
    return x, edge_index

def drag_loss(x):
    return torch.sum(x[:, :, 8])

def solar_loss(x):
    return -torch.sum(x[:, :, 7])

class SolarCarGNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=32, output_dim=2):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(x.device))
        return self.lin(x)

    
def train(num_epochs=50, num_samples=100, num_nodes=3):
    x, _ = generate_synthetic_data(num_samples, num_nodes)
    model = SolarCarGNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    drag_losses, solar_losses = [], []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0
        epoch_drag, epoch_solar = 0, 0

        for i in range(num_samples):
            xi = x[i]
            edge_index_i = torch.combinations(torch.arange(num_nodes), 2).t()
            pred = model(xi.reshape(-1, 9), edge_index_i)
            true_cd = torch.sum(xi[:, 8])
            true_solar = torch.sum(xi[:, 7])
            loss_drag = F.mse_loss(pred[0, 0], true_cd)
            loss_solar = F.mse_loss(pred[0, 1], true_solar)
            loss = loss_drag + loss_solar
            loss.backward()
            total_loss += loss.item()
            epoch_drag += loss_drag.item()
            epoch_solar += loss_solar.item()

        optimizer.step()
        drag_losses.append(epoch_drag/num_samples)
        solar_losses.append(epoch_solar/num_samples)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Avg Drag Loss: {epoch_drag/num_samples:.4f}, Avg Solar Loss: {epoch_solar/num_samples:.4f}")

    return model, drag_losses, solar_losses

def plot_training_progress(drag_losses, solar_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(drag_losses, label="Drag Loss (MSE)")
    plt.plot(solar_losses, label="Solar Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Progress: Drag and Solar Loss")
    plt.legend()
    plt.grid()
    plt.show()

def plot_predictions_vs_true(model, x, num_samples=5):
    model.eval()
    true_cds, pred_cds = [], []
    true_solars, pred_solars = [], []

    for i in range(num_samples):
        xi = x[i]
        edge_index_i = torch.combinations(torch.arange(x.size(1)), 2).t()
        with torch.no_grad():
            pred = model(xi.reshape(-1, 9), edge_index_i)
        true_cds.append(torch.sum(xi[:, 8]).item())
        pred_cds.append(pred[0, 0].item())
        true_solars.append(torch.sum(xi[:, 7]).item())
        pred_solars.append(pred[0, 1].item())

    plt.figure(figsize=(12, 5))

    # Drag Coefficient
    plt.subplot(1, 2, 1)
    plt.scatter(true_cds, pred_cds, c='blue', label="Predictions")
    plt.plot([min(true_cds), max(true_cds)], [min(true_cds), max(true_cds)], 'r--', label="Perfect Prediction")
    plt.xlabel("True Drag Coefficient (Cd)")
    plt.ylabel("Predicted Drag Coefficient (Cd)")
    plt.title("Drag Coefficient: True vs. Predicted")
    plt.legend()
    plt.grid()

    # Solar Power
    plt.subplot(1, 2, 2)
    plt.scatter(true_solars, pred_solars, c='green', label="Predictions")
    plt.plot([min(true_solars), max(true_solars)], [min(true_solars), max(true_solars)], 'r--', label="Perfect Prediction")
    plt.xlabel("True Solar Power (kW)")
    plt.ylabel("Predicted Solar Power (kW)")
    plt.title("Solar Power: True vs. Predicted")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    model, drag_losses, solar_losses = train(num_epochs=50)
    plot_training_progress(drag_losses, solar_losses)

    x, _ = generate_synthetic_data(num_samples=10)
    plot_predictions_vs_true(model, x)

    with torch.no_grad():
        edge_index = torch.combinations(torch.arange(x.size(1)), 2).t()
        pred = model(x.reshape(-1, 9), edge_index)
    print("\nExample Prediction:")
    print(f"Predicted Drag Coefficient (c_d): {pred[0, 0].item():.4f}")
    print(f"Predicted Solar Power (kW): {pred[0, 1].item():.4f}")
    print(f"True Drag Coefficient (c_d): {torch.sum(x[0, :, 8]).item():.4f}")
    print(f"True Solar Power (kW): {torch.sum(x[0, :, 7]).item():.4f}")
