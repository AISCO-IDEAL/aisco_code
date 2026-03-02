"""
Analytical SDF models for benchmarking differentiable isosurface extraction.

These models provide ground truth SDFs with known analytical properties
for validating gradient accuracy of different mesh extraction methods.
"""

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1 import BasicMLP


def load_testing_model(filename: str = "test_model.pth") -> BasicMLP:
    """
    Loads the fixed (!) model for the unit tests.
    Based on files in /cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data
    """
    model_config = {
        "in_dim": 3,
        "hidden": 64,
        "layers": 6,
    }
    model = BasicMLP(**model_config)  # type: ignore[arg-type]
    model.load_state_dict(torch.load(f"/cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data/{filename}"))
    return model


class SphereSDF(nn.Module):
    """
    Analytical SDF for a sphere: SDF(x, y, z) = sqrt(x² + y² + z²) - radius

    This provides a simple test case with known analytical gradients:
    - Surface area: A = 4πr²
    - ∂A/∂r = 8πr
    """

    def __init__(self, radius: float = 1000.0, center: tuple[float, float, float] = (0.0, 0.0, 0.0), device: str = "cuda"):
        """
        Args:
            radius: Sphere radius (default: 1000.0 for consistency with existing models)
            center: Sphere center coordinates (x, y, z)
        """
        super().__init__()
        # Make radius a learnable parameter for gradient computation
        self.radius = nn.Parameter(torch.tensor(radius, dtype=torch.float32))
        self.center = torch.tensor(center, dtype=torch.float32)  # Center is fixed (not optimized in typical benchmarks)
        # Center is fixed (not optimized in typical benchmarks)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.to(device)  # Move model to specified device for consistency with extractors

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to sphere surface.

        Args:
            points: (N, 3) tensor of xyz coordinates

        Returns:
            (N,) tensor of signed distances (negative inside, positive outside)
        """
        # Translate points relative to center
        centered = points - self.center
        # SDF = ||p - c|| - r
        distances = torch.norm(centered, dim=-1) - self.radius
        return distances

    def analytical_surface_area(self) -> torch.Tensor:
        """
        Compute analytical surface area: A = 4πr²

        Returns:
            Scalar tensor with surface area
        """
        return 4.0 * torch.pi * self.radius**2

    def analytical_gradient_wrt_radius(self) -> torch.Tensor:
        """
        Compute analytical gradient of surface area w.r.t. radius: ∂A/∂r = 8πr

        Returns:
            Scalar tensor with gradient
        """
        return 8.0 * torch.pi * self.radius


class CubeSDF(nn.Module):
    """
    Analytical SDF for an axis-aligned box (cube/rectangular prism).

    SDF uses the standard box distance function:
    - Inside: max of negative distances to faces
    - Outside: distance to nearest point on box

    For a cube with side length s:
    - Surface area: A = 6s²
    - ∂A/∂s = 12s
    """

    def __init__(self, side_length: float = 2000.0, center: tuple[float, float, float] = (0.0, 0.0, 0.0), device: str = "cuda"):
        """
        Args:
            side_length: Length of cube sides (creates a cube from -s/2 to +s/2 in each axis)
            center: Box center coordinates (x, y, z)
        """
        super().__init__()
        # Make side length a learnable parameter
        self.side_length = nn.Parameter(torch.tensor(side_length, dtype=torch.float32))
        self.center = torch.tensor(center, dtype=torch.float32)  # Center is fixed (not optimized in typical benchmarks)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.to(device)  # Move model to specified device for consistency with extractors

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to box surface.

        Args:
            points: (N, 3) tensor of xyz coordinates

        Returns:
            (N,) tensor of signed distances
        """
        # Translate to box-centered coordinates
        centered = points - self.center

        # Half extents of the box
        half_size = self.side_length / 2.0

        # Distance to box (standard box SDF formula)
        # q = |p| - half_size (component-wise)
        q = torch.abs(centered) - half_size

        # Outside distance: length of positive components
        outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=-1)

        # Inside distance: maximum negative component
        inside_dist = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)

        # Combine: outside points have outside_dist, inside points have inside_dist
        sdf = outside_dist + inside_dist

        return sdf

    def analytical_surface_area(self) -> torch.Tensor:
        """
        Compute analytical surface area of cube: A = 6s²

        Returns:
            Scalar tensor with surface area
        """
        return 6.0 * self.side_length**2

    def analytical_gradient_wrt_side_length(self) -> torch.Tensor:
        """
        Compute analytical gradient of surface area w.r.t. side length: ∂A/∂s = 12s

        Returns:
            Scalar tensor with gradient
        """
        return 12.0 * self.side_length


class RectangularBoxSDF(nn.Module):
    """
    Analytical SDF for a rectangular box with different dimensions.

    For a box with dimensions (a, b, c):
    - Surface area: A = 2(ab + ac + bc)
    - ∂A/∂a = 2(b + c), ∂A/∂b = 2(a + c), ∂A/∂c = 2(a + b)
    """

    def __init__(
        self,
        dimensions: tuple[float, float, float] = (2000.0, 1500.0, 1000.0),
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        device: str = "cuda",
    ):
        """
        Args:
            dimensions: Box dimensions (width, height, depth) along x, y, z axes
            center: Box center coordinates (x, y, z)
        """
        super().__init__()
        # Make dimensions learnable parameters
        self.dimensions = nn.Parameter(torch.tensor(dimensions, dtype=torch.float32))
        self.center = torch.tensor(center, dtype=torch.float32)  # Center is fixed (not optimized in typical benchmarks)
        self.register_buffer("center", torch.tensor(center, dtype=torch.float32))
        self.to(device)  # Move model to specified device for consistency with extractors

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute signed distance to rectangular box surface.

        Args:
            points: (N, 3) tensor of xyz coordinates

        Returns:
            (N,) tensor of signed distances
        """
        # Translate to box-centered coordinates
        centered = points - self.center

        # Half extents of the box
        half_extents = self.dimensions / 2.0

        # Distance to box (standard box SDF formula)
        q = torch.abs(centered) - half_extents

        # Outside distance: length of positive components
        outside_dist = torch.norm(torch.clamp(q, min=0.0), dim=-1)

        # Inside distance: maximum negative component
        inside_dist = torch.clamp(torch.max(q, dim=-1)[0], max=0.0)

        # Combine
        sdf = outside_dist + inside_dist

        return sdf

    def analytical_surface_area(self) -> torch.Tensor:
        """
        Compute analytical surface area: A = 2(ab + ac + bc)

        Returns:
            Scalar tensor with surface area
        """
        a, b, c = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        return 2.0 * (a * b + a * c + b * c)

    def analytical_gradients_wrt_dimensions(self) -> torch.Tensor:
        """
        Compute analytical gradients of surface area w.r.t. each dimension.

        Returns:
            (3,) tensor with [∂A/∂a, ∂A/∂b, ∂A/∂c]
        """
        a, b, c = self.dimensions[0], self.dimensions[1], self.dimensions[2]
        dA_da = 2.0 * (b + c)  # noqa: N806
        dA_db = 2.0 * (a + c)  # noqa: N806
        dA_dc = 2.0 * (a + b)  # noqa: N806
        return torch.stack([dA_da, dA_db, dA_dc])
