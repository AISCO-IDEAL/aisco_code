

import torch
def sphere_gt(
    points: torch.Tensor,
    coordinates: torch.Tensor = torch.tensor([0, 0, 0]),
    radius: float = 1.0,
) -> torch.Tensor:
    """
    Generate signed distance ground truth values for a sphere centered at the coordinates.
    Args:
        points: Input coordinates of shape (batch_size, 3)
        radius: Radius of the sphere
    """


    return torch.norm(points-coordinates, dim=-1) - radius


def cube_gt(
    points: torch.Tensor,
    coordinates: torch.Tensor = torch.tensor([0, 0, 0]),
    side_length: float = 1.0,
) -> torch.Tensor:
    """
    Generate signed distance ground truth values for a cube centered at the coordinates.
    Args:
        points: Input coordinates of shape (batch_size, 3)
        side_length: Length of the cube's sides
    """
    half_side = side_length / 2.0
    d = torch.abs(points - coordinates) - half_side
    d_clamped = torch.clamp(d, min=0)
    outside_distance = torch.norm(d_clamped, dim=-1)
    inside_distance = torch.max(d, dim=-1).values
    sdf_values = outside_distance + torch.min(inside_distance, torch.zeros_like(inside_distance))
    return sdf_values