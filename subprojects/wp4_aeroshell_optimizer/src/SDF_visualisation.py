
import matplotlib.pyplot as plt
import numpy as np
import torch
from sdf import SDFNetwork


class SDFVisualizer:
    def __init__(self, model: SDFNetwork):
        self.model = model
        self.inner_solid = None
        self.outer_solid = None

    def visualize_slice(self, direction: str, slice_index: int, grid_size: int = 100, grid_range: float = 2.0):
        """
        Args:
            direction: 'xy', 'xz', or 'yz' plane to slice
            slice_index: index of the slice along the orthogonal axis
            grid_size: number of points along each axis
            grid_range: range for x and y axes (-grid_range to grid_range)
        """
        x = np.linspace(-grid_range, grid_range, grid_size)
        y = np.linspace(-grid_range, grid_range, grid_size)
        X, Y = np.meshgrid(x, y)  # noqa: N806
        Z = np.zeros_like(X)  # noqa: N806

        if direction == 'xy':
            points = np.stack([X.flatten(), Y.flatten(), np.full(X.size, slice_index * (2 * grid_range) / grid_size - grid_range)], axis=-1)
        elif direction == 'xz':
            points = np.stack([X.flatten(), np.full(X.size, slice_index * (2 * grid_range) / grid_size - grid_range), Y.flatten()], axis=-1)
        elif direction == 'yz':
            points = np.stack([np.full(X.size, slice_index * (2 * grid_range) / grid_size - grid_range), X.flatten(), Y.flatten()], axis=-1)
        else:
            raise ValueError("Direction_must_be_'xy',_'xz',_or_'yz'.")

        points_tensor = torch.tensor(points).float()
        sdf_values, _ = self.model(points_tensor)
        Z = sdf_values.detach().numpy().reshape(X.shape)  # noqa: N806

        plt.contourf(X, Y, Z, levels=50, cmap='RdBu_r')
        plt.colorbar(label='SDF Value')
        plt.title(f'SDF Slice in {direction.upper()} plane at index {slice_index}')
        plt.xlabel('X-axis' if direction != 'yz' else 'Y-axis')
        plt.ylabel('Y-axis' if direction != 'xz' else 'Z-axis')
        plt.axis('equal')
        plt.show()
