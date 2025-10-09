import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#inital copilot generation, to be checked
class SDFNetwork(nn.Module):
    def __init__(self, d_in=3, d_out=1, d_hidden=256, n_layers=8, skip_in=(4,), geometric_init=True, bias=0.5):
        super(SDFNetwork, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.skip_in = skip_in

        layers = []
        for l in range(n_layers):
            if l == 0:
                layer = nn.Linear(d_in, d_hidden)
            elif l in skip_in:
                layer = nn.Linear(d_hidden + d_in, d_hidden)
            else:
                layer = nn.Linear(d_hidden, d_hidden)

            if geometric_init:
                if l == n_layers - 1:
                    torch.nn.init.normal_(layer.weight, mean=np.sqrt(np.pi) / np.sqrt(d_hidden), std=0.0001)
                    torch.nn.init.constant_(layer.bias, -bias)
                elif l == 0:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.out_features))
                else:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(layer.out_features))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.activation = F.softplus

    def forward(self, x):
        h = x
        for l in range(self.n_layers):
            layer = self.layers[l]
            if l in self.skip_in:
                h = torch.cat([h, x], dim=-1) / np.sqrt(2)
            h = layer(h)
            if l < self.n_layers - 1:
                h = self.activation(h)
        return h