import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

##################################
# 1. Definition of Classes
##################################


# 2.1 Shape: Represents a 2D shape as a graph
class Shape:
    def __init__(self, points, pointcloud=False):
        self.size = 1000
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
        # Create edges based on spatial proximity (e.g., k-nearest neighbors)
        x = torch.tensor(self.pointcloud, dtype=torch.float32)
        edge_index = torch.combinations(torch.arange(self.size), 2).t()
        return Data(x=x, edge_index=edge_index)

    def plot(self):
        plt.scatter(self.pointcloud[:, 0], self.pointcloud[:, 1], s=1)
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


# 2.2 Area: Represents an area as a graph
class Area:
    def __init__(self, inner_shape, outer_shape):
        self.inner_shape = inner_shape
        self.outer_shape = outer_shape
        self.size = inner_shape.size
        self.graph = self.create_graph()

    def create_graph(self):
        inner = self.inner_shape.graph.x  # shape: (1000, 2)
        outer = self.outer_shape.graph.x  # shape: (1000, 2)
        x = torch.cat([inner, outer], dim=0)  # shape: (2000, 2)
        edge_index = torch.cat(
            [
                self.inner_shape.graph.edge_index,  # edges within inner shape
                self.outer_shape.graph.edge_index
                + self.size,  # edges within outer shape (offset by 1000)
                torch.stack(
                    [
                        torch.arange(self.size),  # connect inner[i] to outer[i]
                        torch.arange(self.size, 2 * self.size),
                    ],
                    dim=0,
                ),
            ],
            dim=1,
        )
        return Data(x=x, edge_index=edge_index)


# 2.3 Autoencoder: GNN-based autoencoder
class Autoencoder:
    def __init__(self, areas):
        self.areas = areas
        self.size = areas[0].size
        self.model, self.latent_size = self.eval_model()

    def eval_model(self):
        latent_size = 3
        allowed_loss = 0.06
        best_model = None
        while True:
            model = AutoencoderModel(self.size, latent_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(30):
                total_loss = 0
                for area in self.areas:
                    data = area.graph
                    output = model(data.x, data.edge_index)
                    loss = F.mse_loss(output, data.x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0:
                    print(
                        f"Autoencoder Latent Size: {latent_size}, Epoch {epoch}, Loss: {total_loss / len(self.areas)}"
                    )
            if total_loss / len(self.areas) < allowed_loss:
                best_model = model
                print("Best latent size found:", latent_size)
                break
            latent_size += 1
        return best_model, latent_size

    def encode(self, area):
        data = area.graph
        with torch.no_grad():
            latent = model.encoder(data.x, data.edge_index)
        return latent.numpy().tolist()


# 2.4 AutoencoderModel: GNN-based autoencoder (fixed)
class Autoencoder:
    def __init__(self, areas):
        self.areas = areas
        self.size = areas[0].size
        self.model, self.latent_size = self.eval_model()

    def eval_model(self):
        latent_size = 3
        allowed_loss = 0.06
        best_model = None
        while True:
            model = AutoencoderModel(2, latent_size)  # in_channels=2
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(30):
                total_loss = 0
                for area in self.areas:
                    data = area.graph
                    output = model(data.x, data.edge_index)
                    loss = F.mse_loss(output, data.x.view(-1, 2000, 2))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0:
                    print(
                        f"Autoencoder Latent Size: {latent_size}, Epoch {epoch}, Loss: {total_loss / len(self.areas)}"
                    )
            if total_loss / len(self.areas) < allowed_loss:
                best_model = model
                print("Best latent size found:", latent_size)
                break
            latent_size += 1
        return best_model, latent_size

    # ... (rest of the class)


# 2.6 GeniModel: GNN-based model (fixed)
class GeniModel(nn.Module):
    def __init__(self, latent_size, output_size):
        super(GeniModel, self).__init__()
        self.conv1 = GCNConv(latent_size, 64)
        self.conv2 = GCNConv(64, 32)
        self.lin = nn.Linear(32, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        return torch.sigmoid(self.lin(x))


# 2.4 AutoencoderModel: GNN-based autoencoder
class AutoencoderModel(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AutoencoderModel, self).__init__()
        # Input: 2000 nodes, each with 2 features (x, y)
        self.conv1 = GCNConv(2, 128)  # in_channels=2, out_channels=128
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2000 * 2),  # Output: flattened (2000, 2)
        )

    def forward(self, x, edge_index):
        # Encoder
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        latent = self.conv3(x, edge_index)
        latent = global_mean_pool(latent, torch.zeros(latent.size(0), dtype=torch.long))
        # Decoder
        output = self.decoder(latent)
        return output.view(-1, 2000, 2)  # Reshape to (batch, 2000, 2)


# 2.5 Geni: GNN-based reinforcement learning
class Geni:
    def __init__(self, areas):
        self.areas = areas
        self.autoencoder = Autoencoder(self.areas)
        self.latent_size = self.autoencoder.latent_size
        self.output_size = areas[0].size  # mu is a vector of size `area.size`
        self.model = GeniModel(self.latent_size, self.output_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def encode_torch(self, area):
        data = area.graph
        with torch.no_grad():
            z = self.autoencoder.model.encoder(data.x, data.edge_index)
            z = global_mean_pool(z, torch.zeros(z.size(0), dtype=torch.long))
        return z

    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            self.optimizer.zero_grad()
            batch_losses = []
            for area in self.areas:
                z = self.encode_torch(area)
                mu = self.model(
                    z.unsqueeze(0), area.graph.edge_index
                )  # Add batch dimension
                pts = area.pointcloud_torch(mu.squeeze(0))  # Remove batch dimension
                loss = self.loss_function(pts)
                batch_losses.append(loss)
            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}")

    def loss_function(self, points, alpha=10):
        def soft_max(x):
            return torch.logsumexp(alpha * x, dim=0) / alpha

        def soft_min(x):
            return -torch.logsumexp(-alpha * x, dim=0) / alpha

        x_max = soft_max(points[:, 0])
        x_min = soft_min(points[:, 0])
        y_max = soft_max(points[:, 1])
        y_min = soft_min(points[:, 1])
        width = x_max - x_min
        height = y_max - y_min
        diffs = points[1:] - points[:-1]
        circumference = torch.norm(diffs, dim=1).sum() + torch.norm(
            points[0] - points[-1]
        )
        loss = circumference / 10 + (height - 0.5) + 1 / width
        return loss

    def get_mu(self, area):
        latent = self.autoencoder.encode(area)
        latent = torch.tensor(latent, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(latent.unsqueeze(0), area.graph.edge_index)
        return output.squeeze(0).numpy()


# 2.6 GeniModel: GNN-based model
class GeniModel(nn.Module):
    def __init__(self, latent_size, output_size):
        super(GeniModel, self).__init__()
        self.conv1 = GCNConv(latent_size, 64)
        self.conv2 = GCNConv(64, 32)
        self.lin = nn.Linear(32, output_size)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long))
        return torch.sigmoid(self.lin(x))


##################################
# 3. Main Code Execution
quader1 = Shape([[0, 0], [1, 0], [1.5, 0.5], [1, 1], [0, 1], [-0.5, 0.5]])
quader2 = Shape([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]])
area = Area(quader2, quader1)
areas = []
for x in range(5):
    for y in range(5):
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

geni = Geni(areas)
for i in range(10):
    geni.train(100)
    new_shape = area.create_shape(geni.get_mu(area))
    area.plot()
    new_shape.plot()
    plt.show()
