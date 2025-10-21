import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T
import numpy as np
import trimesh
import os
import datetime
from skimage import measure

class SDFNetwork(nn.Module):
    def __init__(self, in_dim=3, hidden=128, layers=6, use_skip_connection=False):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.use_skip_connection = use_skip_connection
        self.input_layer = nn.Linear(in_dim, hidden)
        self.middle_layers = nn.ModuleList()
        for i in range(layers - 1):
            self.middle_layers.append(nn.Linear(hidden, hidden))
        if use_skip_connection and layers > 4:
            self.skip_layer = nn.Linear(hidden + in_dim, hidden)
        self.output_layer = nn.Linear(hidden, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_laplacian=False):
        original_x = x
        x = F.leaky_relu(self.input_layer(x), negative_slope=0.1)
        for i, layer in enumerate(self.middle_layers):
            if i % 2 == 0:
                x = torch.tanh(layer(x))
            else:
                x = F.leaky_relu(layer(x), negative_slope=0.1)
            if (self.use_skip_connection and
                i == len(self.middle_layers) // 2 and
                hasattr(self, 'skip_layer')):
                x = torch.cat([x, original_x], dim=1)
                x = F.leaky_relu(self.skip_layer(x), negative_slope=0.1)
        sdf = self.output_layer(x).squeeze(-1)

        # Only compute gradients if input requires grad (i.e., during training)
        if x.requires_grad:
            grad_sdf = torch.zeros((x.shape[0], self.in_dim), device=x.device)
            for dim in range(self.in_dim):
                grad = torch.autograd.grad(
                    outputs=sdf,
                    inputs=original_x,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                grad_sdf[:, dim] = grad[:, dim]
            print("grad_sdf shape:", grad_sdf.shape)  # Debug print
            if not return_laplacian:
                return sdf, grad_sdf
            # Compute Laplacian (sum of second derivatives)
            laplacian_sdf = 0.0
            for dim in range(self.in_dim):
                second_deriv = torch.autograd.grad(
                    outputs=grad_sdf[:, dim],
                    inputs=original_x,
                    grad_outputs=torch.ones_like(grad_sdf[:, dim]),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                laplacian_sdf += second_deriv[:, dim]
            return sdf, grad_sdf, laplacian_sdf
        else:
            # During inference, return only SDF and None for grad/laplacian
            return sdf, None