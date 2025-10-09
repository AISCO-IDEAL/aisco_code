"""
This folder is until now documented by comments in the code files.
It was started on  Mittwoch 8.10.2025 by Clemens.

Structure Overview:
0. Imports
1. Definition of Functions
2. Definition of Classes
3. Main Code Execution
"""
#################################
# 0. Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib . pyplot as plt


##################################
# 1. Definition of Functions
def soft_max(x, alpha=10):
    return torch.logsumexp(alpha * x, dim=0) / alpha

def soft_min(x, alpha=10):
    return -torch.logsumexp(-alpha * x, dim=0) / alpha



##################################
# 2. Definition of Classes

class Shape:
    def __init__(self, points, pointcloud=False):
        self.size = 1000
        self.orgcloud=pointcloud
        if pointcloud:
            if self.size != len(points):
                raise ValueError("If pointcloud is True, points must be of size 'size'")
            self.pointcloud = points
        else:
            self.points = points
            self.pointcloud = self.create_pointcloud()
    def create_pointcloud(self): # creates a pointcloud of size "size" of points equally spread along the shape
        self.points.append(self.points[0])
        self.length = [] # contains the distance of the coresponding point to the next point
        for i in range(len(self.points)-1):
            self.length.append(np.linalg.norm(np.array(self.points[i+1])-np.array(self.points[i])))
        self.total_length = sum(self.length)
        pointcloud = []
        for i in range(len(self.points)-1):
            num_points = int(self.size * self.length[i] / self.total_length)
            for j in range(num_points):
                point = np.array(self.points[i]) + (j / num_points) * (np.array(self.points[i+1]) - np.array(self.points[i]))
                point = [point[0], point[1]]
                pointcloud.append(point)
        while len(pointcloud) < self.size:
            pointcloud.append(pointcloud[-1])
        return np.array(pointcloud)
    def plot(self):
        plt.scatter(np.array(self.pointcloud)[:,0], np.array(self.pointcloud)[:,1], s=1)
        if not self.orgcloud:
            plt.scatter(np.array(self.points)[:,0], np.array(self.points)[:,1], color='red', marker='x')
        plt.axis('equal')
    def p(self, t):
        if t<0 or t> self.size:
            raise ValueError("t must be between 0 and size")
        return self.pointcloud[t]


class Area:
    def __init__(self, inner_shape, outer_shape):
        self.inner_shape = inner_shape
        self.outer_shape = outer_shape
        self.size = inner_shape.size
    def p(self, t, mu):
        if t<0 or t> self.size:
            raise ValueError("t must be between 0 and size")
        if mu<0 or mu>1:
            raise ValueError("mu must be between 0 and 1")
        inner_point = self.inner_shape.p(t)
        outer_point = self.outer_shape.p(t)
        po = (1-mu)*inner_point + mu*outer_point
        return [po[0], po[1]]
    def create_shape(self, mu):
        if len(mu)!=self.size:
            raise ValueError("mu must be of size 'size'")
        points = []
        for t in range(self.size):
            points.append(self.p(t, mu[t]))
        return Shape(points, pointcloud=True)
    def plot(self):
        self.inner_shape.plot()
        self.outer_shape.plot()
    def pointcloud_torch(self, mu: torch.Tensor) -> torch.Tensor:
        # mu: (size,) tensor in [0,1]
        inner = torch.tensor(self.inner_shape.pointcloud, dtype=torch.float32, device=mu.device)  # (size,2)
        outer = torch.tensor(self.outer_shape.pointcloud, dtype=torch.float32, device=mu.device)
        mu2 = mu.view(-1, 1)  # (size,1)
        return (1 - mu2) * inner + mu2 * outer  # (size,2)

class Autoencoder:
    def __init__(self, areas):
        self.areas = areas
        self.size = areas[0].size
        self.model, self.latent_size = self.eval_model()
    
    def eval_model(self): #trains the autoencoder for diffrent sized latent spaces until the smalles possible latent space with a reconstruction loss below 0.5%
        latent_size = 3
        best_model = None
        while True:
            model = AutoencoderModel(self.size, latent_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for epoch in range(30):
                total_loss = 0
                for area in self.areas:
                    input_data = torch.tensor(np.concatenate((area.inner_shape.pointcloud.flatten(), area.outer_shape.pointcloud.flatten())), dtype=torch.float32)
                    output_data = model(input_data)
                    loss = F.mse_loss(output_data, input_data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 10 == 0:
                    print(f'Autoencoder Latent Size: {latent_size}, Epoch {epoch}, Loss: {total_loss/len(self.areas)}')
            if total_loss/len(self.areas) < 0.06:
                best_model = model
                print('Best latent size found:', latent_size)
                break
            latent_size += 1
        return best_model, latent_size

    def encode(self, area):
        input_data = torch.tensor(np.concatenate((area.inner_shape.pointcloud.flatten(), area.outer_shape.pointcloud.flatten())), dtype=torch.float32)
        with torch.no_grad():
            latent = self.model.encoder(input_data)
        #format latent to list
        latent = latent.numpy().tolist()
        return latent

class AutoencoderModel(nn.Module):
    def __init__(self, input_size, latent_size):
        super(AutoencoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size*4, 512*2),
            nn.ReLU(),
            nn.Linear(512*2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512*2),
            nn.ReLU(),
            nn.Linear(512*2, input_size*4),
            nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.encoder(x)
        output = self.decoder(latent)
        return output
# Geni is a neuronal network using reenforcement leraning to find the best mu for the area. As input it getsthe to a latent space reduced pointclouds of the inner and outer shapes of the area. they are reduced by an autoencoder.
class Geni:
    def __init__(self, areas):
        self.areas = areas
        self.autoencoder = Autoencoder(self.areas)
        self.latent_size = self.autoencoder.latent_size
        self.model = nn.Sequential(
            nn.Linear(self.latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, area.size),
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)    
    def encode_torch(self, area):
        x = torch.tensor(
            np.concatenate((area.inner_shape.pointcloud.flatten(),
                            area.outer_shape.pointcloud.flatten())),
            dtype=torch.float32
        )
        with torch.no_grad():
            z = self.autoencoder.model.encoder(x)
        return z  # torch tensor (no grad needed)
    def train(self, epochs):
        for epoch in range(epochs):
            total_loss = 0.0
            self.optimizer.zero_grad()

            batch_losses = []
            for area in self.areas:
                z = self.encode_torch(area)               # (latent_size,)
                mu = self.model(z)                        # (size,) differentiable
                pts = area.pointcloud_torch(mu)           # (size,2) differentiable wrt mu
                loss = self.loss_function(pts)            # scalar tensor
                batch_losses.append(loss)

            total_loss = torch.stack(batch_losses).mean()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Total Loss: {total_loss.item():.6f}")

    def loss_function(self, points, alpha=10):
        # points = nn.Parameter(torch.tensor(shape.pointcloud, dtype=torch.float32))

        def soft_max(x): return torch.logsumexp(alpha * x, dim=0) / alpha
        def soft_min(x): return -torch.logsumexp(-alpha * x, dim=0) / alpha

        x_max = soft_max(points[:, 0])
        x_min = soft_min(points[:, 0])
        y_max = soft_max(points[:, 1])
        y_min = soft_min(points[:, 1])

        width = x_max - x_min
        height = y_max - y_min

        diffs = points[1:] - points[:-1]
        circumference = torch.norm(diffs, dim=1).sum() + torch.norm(points[0] - points[-1])

        loss = circumference/10  +(height-0.5) + 1/width 
        return loss

    def get_mu(self, area):
        latent = self.autoencoder.encode(area)
        latent = torch.tensor(latent, dtype=torch.float32)
        with torch.no_grad():
            output = self.model(latent)
        mu = output.numpy()
        return mu


##################################
# 3. Main Code execution
quader1 = Shape([[0,0],[1,0],[1.5, 0.5], [1,1],[0,1], [-0.5, 0.5]])
quader2 = Shape([[0.25,0.25],[0.75,0.25],[0.75,0.75],[0.25,0.75]])
area = Area(quader2, quader1)

areas = []
for x in range(5):
    for y in range(5):
        points1 = [[0,0],[1,0-0.2*y],[1.5+0.1*x, 0.5+0.05*y], [1+0.1*x,1+0.1*y],[0,1], [-0.5-0.2*x, 0.5+0.02*y]]
        points2 = [[0.25-0.1*x,0.25],[0.75+0.1*x,0.25-0.1*y],[0.75-0.1*x,0.75+0.1*y],[0.25+0.1*x,0.75-0.1*y]]
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