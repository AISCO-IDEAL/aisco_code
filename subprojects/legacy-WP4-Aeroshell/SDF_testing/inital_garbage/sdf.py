import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#inital copilot generation, to be checked

class SDFNetwork(nn.Module):
    def __init__(self, d_in=2, d_out=1, d_hidden=50, n_layers=3):
        super(SDFNetwork, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.n_layers = n_layers

        layers = []
        for l in range(n_layers-1):
            if l == 0:
                layer = nn.Linear(d_in, d_hidden)
            else:
                layer = nn.Linear(d_hidden, d_hidden)


            layers.append(layer)
            #layers.append(nn.ReLU(d_hidden))
        layers.append(nn.Linear(d_hidden, d_out,bias=False))
        self.layers = nn.ModuleList(layers)
        self.activation = F.leaky_relu


    def forward(self, x):
        h = x
        for l in range(self.n_layers):
            layer = self.layers[l]
            #if l in self.skip_in:
            #    h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = layer(h)
            if l < self.n_layers - 2:
                h = self.activation(h)

        return h


    def validate(self, radius,size=100):

        phis=np.linspace(0,2*np.pi,size)
        points=np.zeros((size,2))
        for i in range(size):
            points[i][0]=radius*np.cos(phis[i])
            points[i][1]=radius*np.sin(phis[i])
        points=torch.tensor(points,dtype=torch.float32)
        with torch.no_grad():
            out_t = self(points)
            out_t = out_t.detach().cpu()
            # Some PyTorch builds raise RuntimeError when calling .numpy()
            # if the torch installation was built without numpy support. Use
            # a safe fallback: try .numpy(), otherwise convert via .tolist()
            try:
                sdf_values = out_t.numpy()
            except Exception:
                sdf_values = np.array(out_t.tolist())
        loss=np.mean(np.abs(sdf_values))
        print("Validation loss in avg units away: ",loss)



    #Custom Network Member Function to visualize SDF on 2x2 grid:

    def visualize_sdf(self, grid_size=100):
        """Evaluate the network on a 2x2 grid and plot the scalar output.

        Returns the computed grid (shape (grid_size, grid_size)).
        """
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)

        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N,2)

        points_tensor = torch.tensor(points, dtype=torch.float32)
        with torch.no_grad():
            out_t = self(points_tensor)
            out_t = out_t.detach().cpu()
            # Some PyTorch builds raise RuntimeError when calling .numpy()
            # if the torch installation was built without numpy support. Use
            # a safe fallback: try .numpy(), otherwise convert via .tolist()
            try:
                sdf_values = out_t.numpy()
            except Exception:
                sdf_values = np.array(out_t.tolist())

        sdf_grid = sdf_values.reshape((grid_size, grid_size))

        # Create the plot
        plt.figure(figsize=(8, 8))
        plt.contourf(xx, yy, sdf_grid, levels=50)
        plt.colorbar(label='SDF Value')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('SDF Visualization')
        plt.show()

        return sdf_grid
    def visualize_gradients(self, grid_size=100):
        """Evaluate the gradients on a 2x2 grid and plot their magnitude."""
        x = np.linspace(-2, 2, grid_size)
        y = np.linspace(-2, 2, grid_size)
        xx, yy = np.meshgrid(x, y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=-1)

        points_tensor = torch.tensor(points, dtype=torch.float32, requires_grad=True)
        out_t = self(points_tensor)

        grad_outputs = torch.ones_like(out_t)
        gradients = torch.autograd.grad(outputs=out_t,
                                      inputs=points_tensor,
                                      grad_outputs=grad_outputs,
                                      create_graph=True)[0]

        gradient_magnitudes = torch.norm(gradients, dim=1).detach().cpu()
        try:
            gradient_magnitudes = gradient_magnitudes.numpy()
        except Exception:
            gradient_magnitudes = np.array(gradient_magnitudes.tolist())
        gradient_grid = gradient_magnitudes.reshape((grid_size, grid_size))

        plt.figure(figsize=(8, 8))
        plt.contourf(xx, yy, gradient_grid, levels=50, vmin=0, vmax=2)
        plt.colorbar(label='Gradient Magnitude')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('SDF Gradient Magnitudes')
        plt.show()

        return gradient_grid
