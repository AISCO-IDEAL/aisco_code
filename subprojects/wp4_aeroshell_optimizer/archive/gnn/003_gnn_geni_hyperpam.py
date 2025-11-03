import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

# ===== Hyperparameters =====
HYPERPARAMS = {
    # Shape/Area
    "pointcloud_size": 1000,
    # Autoencoder
    "autoencoder_latent_sizes": [3, 5, 7],  # Latent sizes to try
    "autoencoder_epochs": 1,  # Epochs per latent size
    "autoencoder_lr": 0.001,  # Learning rate
    "autoencoder_allowed_loss": 0.5,  # Stopping loss
    # Geni
    "geni_epochs": 1,  # Training epochs
    "geni_lr": 0.001,  # Learning rate
    "geni_alpha": 10,  # Softmax/softmin alpha
    # Data
    "num_areas_x": 2,  # Number of areas in x-direction
    "num_areas_y": 2,  # Number of areas in y-direction
    # Plotting
    "plot_title": "Original (blue/red) vs Optimized (orange)",
}


# ===== Shape Class =====
class Shape:
    def __init__(self, points, pointcloud=False):
        self.size = HYPERPARAMS["pointcloud_size"]
        self.orgcloud = pointcloud
        if pointcloud:
            if self.size != len(points):
                raise ValueError("Point cloud size must match.")
            self.pointcloud = points
        else:
            self.points = points
            self.pointcloud = self.create_pointcloud()
        self.graph = self.create_graph()

    def create_pointcloud(self):
        self.points.append(self.points[0])
        self.length = []
        for i in range(len(self.points) - 1):
            self.length.append(
                np.linalg.norm(np.array(self.points[i + 1]) - np.array(self.points[i]))
            )
        self.total_length = sum(self.length)
        pointcloud = []
        for i in range(len(self.points) - 1):
            num_points = int(self.size * self.length[i] / self.total_length)
            for j in range(num_points):
                point = np.array(self.points[i]) + (j / num_points) * (
                    np.array(self.points[i + 1]) - np.array(self.points[i])
                )
                pointcloud.append(point)
        while len(pointcloud) < self.size:
            pointcloud.append(pointcloud[-1])
        return np.array(pointcloud)

    def create_graph(self):
        x = torch.tensor(self.pointcloud, dtype=torch.float32)
        edge_index = torch.combinations(torch.arange(self.size), 2).t()
        return Data(x=x, edge_index=edge_index)

    def plot(self, color="blue"):
        plt.scatter(self.pointcloud[:, 0], self.pointcloud[:, 1], s=1, color=color)
        if not self.orgcloud:
            plt.scatter(
                np.array(self.points)[:, 0],
                np.array(self.points)[:, 1],
                color="red",
                marker="x",
            )
        plt.axis("equal")

    def p(self, t):
        if t < 0 or t >= self.size:
            raise ValueError("t must be between 0 and size")
        return self.pointcloud[t]


# ===== Area Class =====
class Area:
    def __init__(self, inner_shape, outer_shape):
        self.inner_shape = inner_shape
        self.outer_shape = outer_shape
        self.size = inner_shape.size
        self.graph = self.create_graph()

    def create_graph(self):
        inner = self.inner_shape.graph.x
        outer = self.outer_shape.graph.x
        x = torch.cat([inner, outer], dim=0)
        edge_index = torch.cat(
            [
                self.inner_shape.graph.edge_index,
                self.outer_shape.graph.edge_index + self.size,
                torch.stack(
                    [torch.arange(self.size), torch.arange(self.size, 2 * self.size)],
                    dim=0,
                ),
            ],
            dim=1,
        )
        return Data(x=x, edge_index=edge_index)

    def create_shape(self, mu):
        if len(mu) != self.size:
            raise ValueError("mu must have the same length as the point cloud")

        inner = self.inner_shape.pointcloud
        outer = self.outer_shape.pointcloud
        mu2 = np.array(mu).reshape(-1, 1)

        # Linear interpolation between inner and outer shape points
        points = (1 - mu2) * inner + mu2 * outer
        return Shape(points, pointcloud=True)

    def pointcloud_torch(self, mu: torch.Tensor) -> torch.Tensor:
        inner = self.inner_shape.graph.x
        outer = self.outer_shape.graph.x
        mu2 = mu.view(-1, 1)
        return (1 - mu2) * inner + mu2 * outer

    def plot(self, color_inner="blue", color_outer="orange"):
        self.inner_shape.plot(color=color_inner)
        self.outer_shape.plot(color=color_outer)


# ===== AutoencoderModel =====
class AutoencoderModel(nn.Module):
    def __init__(self, latent_size):
        super(AutoencoderModel, self).__init__()
        self.conv1 = GCNConv(2, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2000 * 2),
        )

    def forward(self, x, edge_index):
        # Encoder
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        latent = self.conv3(x, edge_index)
        latent = global_mean_pool(latent, torch.zeros(latent.size(0), dtype=torch.long))
        # Decoder
        output = self.decoder(latent)
        return output.view(-1, 2000, 2)


# ===== Autoencoder =====
class Autoencoder:
    def __init__(self, areas):
        self.areas = areas
        self.size = areas[0].size
        self.model, self.latent_size = self.eval_model()

    def eval_model(self):
        best_model = None
        for latent_size in HYPERPARAMS["autoencoder_latent_sizes"]:
            model = AutoencoderModel(latent_size)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=HYPERPARAMS["autoencoder_lr"]
            )
            for epoch in range(HYPERPARAMS["autoencoder_epochs"]):
                total_loss = 0
                for area in self.areas:
                    data = area.graph
                    output = model(data.x, data.edge_index)
                    loss = F.mse_loss(output, data.x.view(-1, 2000, 2))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(
                    f"Autoencoder Latent Size: {latent_size}, Epoch {epoch}, Loss: {total_loss / len(self.areas):.6f}"
                )
            if total_loss / len(self.areas) < HYPERPARAMS["autoencoder_allowed_loss"]:
                best_model = model
                print(f"Best latent size found: {latent_size}")
                break
        else:
            best_model = model
            print(f"Using latent size: {latent_size} (loss threshold not met)")
        return best_model, latent_size

    def encode(self, area):
        data = area.graph
        with torch.no_grad():
            latent = F.relu(self.model.conv1(data.x, data.edge_index))
            latent = F.relu(self.model.conv2(latent, data.edge_index))
            latent = self.model.conv3(latent, data.edge_index)
            latent = global_mean_pool(
                latent, torch.zeros(latent.size(0), dtype=torch.long)
            )
        return latent.numpy().tolist()


# ===== GeniModel =====
class GeniModel(nn.Module):
    def __init__(self, latent_size, output_size):
        super(GeniModel, self).__init__()
        self.fc1 = nn.Linear(latent_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


# ===== Geni =====
class Geni:
    def __init__(self, areas):
        self.areas = areas
        self.autoencoder = Autoencoder(self.areas)
        self.latent_size = self.autoencoder.latent_size
        self.output_size = areas[0].size
        self.model = GeniModel(self.latent_size, self.output_size)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=HYPERPARAMS["geni_lr"]
        )

    def encode_torch(self, area):
        data = area.graph
        with torch.no_grad():
            latent = F.relu(self.autoencoder.model.conv1(data.x, data.edge_index))
            latent = F.relu(self.autoencoder.model.conv2(latent, data.edge_index))
            latent = self.autoencoder.model.conv3(latent, data.edge_index)
            latent = global_mean_pool(
                latent, torch.zeros(latent.size(0), dtype=torch.long)
            )
        return latent

    def train(self):
        for epoch in range(HYPERPARAMS["geni_epochs"]):
            total_loss = 0.0
            self.optimizer.zero_grad()
            batch_losses = []
            for area in self.areas:
                z = self.encode_torch(area)
                mu = self.model(z)  # Kein edge_index nötig!
                pts = area.pointcloud_torch(mu)
                loss = self.loss_function(pts)
                batch_losses.append(loss)
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            self.optimizer.step()
            print(f"Geni Epoch {epoch}, Total Loss: {total_loss.item():.6f}")

    def loss_function(self, points):
        alpha = HYPERPARAMS["geni_alpha"]

        def soft_max(x):
            return torch.logsumexp(alpha * x, dim=0) / alpha

        def soft_min(x):
            return -torch.logsumexp(-alpha * x, dim=0) / alpha

        x_max, x_min = soft_max(points[:, 0]), soft_min(points[:, 0])
        y_max, y_min = soft_max(points[:, 1]), soft_min(points[:, 1])
        width, height = x_max - x_min, y_max - y_min
        diffs = points[1:] - points[:-1]
        circumference = torch.norm(diffs, dim=1).sum() + torch.norm(
            points[0] - points[-1]
        )
        loss = circumference / 10 + (height - 0.5) + 1 / width
        return loss

    def get_mu(self, area):
        latent = self.encode_torch(area)
        with torch.no_grad():
            output = self.model(latent)
        return output.squeeze(0).cpu().numpy()  # now shape = (1000,)


# ===== Main Execution =====
def main():
    # Create shapes and areas
    quader1 = Shape([[0, 0], [1, 0], [1.5, 0.5], [1, 1], [0, 1], [-0.5, 0.5]])
    quader2 = Shape([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
    area = Area(quader2, quader1)
    areas = []
    for x in range(HYPERPARAMS["num_areas_x"]):
        for y in range(HYPERPARAMS["num_areas_y"]):
            points1 = [
                [0, 0],
                [1, 0 - 0.2 * y],
                [1.5 + 0.1 * x, 0.5 + 0.05 * y],
                [1 + 0.1 * x, 1 + 0.1 * y],
                [0, 1],
                [-0.5 - 0.2 * x, 0.5 + 0.02 * y],
            ]
            points2 = [
                [0.25 - 0.1 * x, 0.25],
                [0.75 + 0.1 * x, 0.25 - 0.1 * y],
                [0.75 - 0.1 * x, 0.75 + 0.1 * y],
                [0.25 + 0.1 * x, 0.75 - 0.1 * y],
            ]
            quader1 = Shape(points1)
            quader2 = Shape(points2)
            area1 = Area(quader2, quader1)
            areas.append(area1)

    # Train and visualize
    geni = Geni(areas)
    geni.train()
    new_shape = area.create_shape(geni.get_mu(area))
    area.plot(color_inner="blue", color_outer="orange")
    new_shape.plot(color="green")
    plt.title(HYPERPARAMS["plot_title"])
    plt.show()


if __name__ == "__main__":
    main()
