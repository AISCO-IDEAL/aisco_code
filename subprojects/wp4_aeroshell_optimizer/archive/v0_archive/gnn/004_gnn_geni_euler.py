import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.nn import GCNConv, global_mean_pool


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


# ===== Hyperparameters =====
HYPERPARAMS = {
    "pointcloud_size": 1000,
    "autoencoder_latent_sizes": [3, 5, 7],
    "autoencoder_epochs": 1,
    "autoencoder_lr": 0.001,
    "autoencoder_allowed_loss": 0.5,
    "geni_epochs": 1,
    "geni_lr": 0.001,
    "geni_alpha": 10,
    "num_areas_x": 2,
    "num_areas_y": 2,
    "plot_title": "Original (blue/red) vs Optimized (orange)",
}

# ===== Shape and Area Classes =====
# (Keep your Shape and Area classes as-is)


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
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        latent = self.conv3(x, edge_index)
        latent = global_mean_pool(
            latent, torch.zeros(latent.size(0), dtype=torch.long, device=x.device)
        )
        output = self.decoder(latent)
        return output.view(-1, 2000, 2)


# ===== Autoencoder =====
class Autoencoder:
    def __init__(self, areas, rank):
        self.areas = areas
        self.size = areas[0].size
        self.rank = rank
        self.model, self.latent_size = self.eval_model()

    def eval_model(self):
        best_model = None
        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )
        for latent_size in HYPERPARAMS["autoencoder_latent_sizes"]:
            model = AutoencoderModel(latent_size).to(device)
            model = DDP(model, device_ids=[self.rank])
            optimizer = torch.optim.Adam(
                model.parameters(), lr=HYPERPARAMS["autoencoder_lr"]
            )
            for epoch in range(HYPERPARAMS["autoencoder_epochs"]):
                total_loss = 0
                for area in self.areas:
                    data = area.graph.to(device)
                    output = model(data.x, data.edge_index)
                    loss = F.mse_loss(output, data.x.view(-1, 2000, 2))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(self.areas)
                if self.rank == 0:
                    print(
                        f"Autoencoder Latent Size: {latent_size}, Epoch {epoch}, Loss: {avg_loss:.6f}"
                    )
                if avg_loss < HYPERPARAMS["autoencoder_allowed_loss"]:
                    best_model = model
                    if self.rank == 0:
                        print(f"Best latent size found: {latent_size}")
                    break
        best_model = model
        if self.rank == 0:
            print(f"Using latent size: {latent_size} (loss threshold not met)")
        return best_model, latent_size

    def encode(self, area):
        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )
        data = area.graph.to(device)
        with torch.no_grad():
            latent = F.relu(self.model.conv1(data.x, data.edge_index))
            latent = F.relu(self.model.conv2(latent, data.edge_index))
            latent = self.model.conv3(latent, data.edge_index)
            latent = global_mean_pool(
                latent, torch.zeros(latent.size(0), dtype=torch.long, device=device)
            )
        return latent.cpu().numpy().tolist()


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
    def __init__(self, areas, rank, world_size):
        self.areas = areas
        self.rank = rank
        self.world_size = world_size
        self.autoencoder = Autoencoder(self.areas, self.rank)
        self.latent_size = self.autoencoder.latent_size
        self.output_size = areas[0].size
        device = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model = GeniModel(self.latent_size, self.output_size).to(device)
        self.model = DDP(self.model, device_ids=[self.rank])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=HYPERPARAMS["geni_lr"]
        )
        self.device = device

    def encode_torch(self, area):
        data = area.graph.to(self.device)
        with torch.no_grad():
            latent = F.relu(self.autoencoder.model.conv1(data.x, data.edge_index))
            latent = F.relu(self.autoencoder.model.conv2(latent, data.edge_index))
            latent = self.autoencoder.model.conv3(latent, data.edge_index)
            latent = global_mean_pool(
                latent,
                torch.zeros(latent.size(0), dtype=torch.long, device=self.device),
            )
        return latent

    def train(self):
        for epoch in range(HYPERPARAMS["geni_epochs"]):
            total_loss = 0.0
            self.optimizer.zero_grad()
            batch_losses = []
            for area in self.areas:
                z = self.encode_torch(area)
                mu = self.model(z)
                pts = area.pointcloud_torch(mu).to(self.device)
                loss = self.loss_function(pts)
                batch_losses.append(loss)
            total_loss = torch.stack(batch_losses).mean()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss / self.world_size
            total_loss.backward()
            self.optimizer.step()
            if self.rank == 0:
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
        return output.squeeze(0).cpu().numpy()


# ===== Main Execution =====
def main(rank, world_size):
    setup(rank, world_size)
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
    geni = Geni(areas, rank, world_size)
    geni.train()
    if rank == 0:
        area = areas[0]
        new_shape = area.create_shape(geni.get_mu(area))
        area.plot(color_inner="blue", color_outer="orange")
        new_shape.plot(color="green")
        plt.title(HYPERPARAMS["plot_title"])
        plt.savefig("result.png")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    main(rank, world_size)
