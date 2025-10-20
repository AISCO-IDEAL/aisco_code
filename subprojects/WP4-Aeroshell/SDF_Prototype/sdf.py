import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as T

class SDFNetwork(nn.Module):
    """
    3D Signed Distance Field Neural Network

    This network takes 3D coordinates (x, y, z) as input and outputs a signed distance value.
    The architecture uses a mix of activations for better expressivity.
    """

    def __init__(self, in_dim=3, hidden=128, layers=6, use_skip_connection=False):
        """
        Pre-conditions:
            - in_dim: Must be a positive integer (default: 3, for (x, y, z)).
            - hidden: Must be a positive integer (default: 128).
            - layers: Must be an integer ≥ 2 (default: 6).
            - use_skip_connection: Boolean (default: False).

        Post-conditions:
            - self.in_dim, self.hidden, self.layers, self.use_skip_connection are set.
            - self.input_layer is a nn.Linear layer mapping in_dim to hidden.
            - self.middle_layers is a nn.ModuleList of (layers-1) nn.Linear layers, each mapping hidden to hidden.
            - If use_skip_connection is True and layers > 4, self.skip_layer is a nn.Linear layer mapping (hidden + in_dim) to hidden.
            - self.output_layer is a nn.Linear layer mapping hidden to 1.
            - All weights are initialized using Xavier/Kaiming initialization.
        """
        super().__init__()

        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.use_skip_connection = use_skip_connection

        # Build the network architecture with varied activations
        self.input_layer = nn.Linear(in_dim, hidden)

        # Middle layers with different activation functions for better expressivity
        self.middle_layers = nn.ModuleList()
        for i in range(layers - 1):
            self.middle_layers.append(nn.Linear(hidden, hidden))

        # Skip connection layer (optional)
        if use_skip_connection and layers > 4:
            self.skip_layer = nn.Linear(hidden + in_dim, hidden)

        # Output layer
        self.output_layer = nn.Linear(hidden, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Pre-conditions:
            - The network's layers (self.input_layer, self.middle_layers, self.skip_layer, self.output_layer) must exist.

        Post-conditions:
            - All weights in nn.Linear layers are initialized with xavier_uniform_.
            - All biases in nn.Linear layers are initialized to zero.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier uniform for tanh, Kaiming for ReLU-like activations
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Pre-conditions:
            - x: A tensor of shape (batch_size, 3) representing 3D coordinates.

        Post-conditions:
            - The input x is passed through the input layer with LeakyReLU activation.
            - For each middle layer:
                - If the layer index is even, tanh activation is applied.
                - If the layer index is odd, LeakyReLU activation is applied.
                - If use_skip_connection is True and at the middle layer, the input is concatenated with the original input and passed through the skip layer with LeakyReLU.
            - The output is passed through the output layer and squeezed to shape (batch_size,).
            - Returns a tensor of shape (batch_size,) with signed distance values.
        """
        original_x = x

        # Input layer with LeakyReLU
        x = torch.nn.functional.leaky_relu(self.input_layer(x), negative_slope=0.1)

        # Middle layers with alternating activations for better expressivity
        for i, layer in enumerate(self.middle_layers):
            if i % 2 == 0:
                # Use Tanh for smoother gradients
                x = torch.tanh(layer(x))
            else:
                # Use LeakyReLU for better gradient flow
                x = torch.nn.functional.leaky_relu(layer(x), negative_slope=0.1)

            # Skip connection at middle of network
            if (self.use_skip_connection and
                i == len(self.middle_layers) // 2 and
                hasattr(self, 'skip_layer')):
                x = torch.cat([x, original_x], dim=1)
                x = torch.nn.functional.leaky_relu(self.skip_layer(x), negative_slope=0.1)

        # Output layer (no activation - raw SDF values)
        return self.output_layer(x).squeeze(-1)
