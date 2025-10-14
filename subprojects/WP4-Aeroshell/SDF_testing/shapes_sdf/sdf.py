import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFNet(nn.Module):
    def __init__(self, in_dim=2, hidden=64, layers=4, activation=nn.ReLU):
        super().__init__()
        mods = []
        last_dim = in_dim
        for _ in range(layers):
            mods += [nn.Linear(last_dim, hidden), activation()]
            last_dim = hidden
        mods.append(nn.Linear(last_dim, 1))   # final linear output (no activation)
        self.network = nn.Sequential(*mods)

        # initialize
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.network(x).squeeze(-1)
