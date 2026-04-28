import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.models.blendednet.gpu_sdf import mesh_sdf


def ply_gt(
    points: torch.Tensor,
    ply_vertices: torch.Tensor,
    ply_faces: torch.Tensor,
    chunk_size: int = 2048,
) -> torch.Tensor:
    """
    Signed distance ground truth from a pre-loaded PLY mesh.

    Args:
        points:       [N, 3] query coordinates (same device as ply_vertices)
        ply_vertices: [V, 3] mesh vertices (float32)
        ply_faces:    [F, 3] face indices (long)
        chunk_size:   query chunk size passed to mesh_sdf

    Returns:
        [N] signed distances — negative inside, positive outside
    """
    return mesh_sdf(points.float(), ply_vertices, ply_faces, chunk_size=chunk_size)


def sphere_gt(
    points: torch.Tensor,
    coordinates: torch.Tensor = torch.tensor([0, 0, 0]),
    radius: float = 1000.0,
) -> torch.Tensor:
    """
    Generate signed distance ground truth values for a sphere centered at the coordinates.
    Args:
        points: Input coordinates of shape (batch_size, 3)
        coordinates: Center of the sphere [x, y, z]
        radius: Radius of the sphere
    """

    coordinates = coordinates.to(points.device)
    return torch.norm(points - coordinates, dim=-1) - radius


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
    coordinates = coordinates.to(points.device)
    half_side = side_length / 2.0
    d = torch.abs(points - coordinates) - half_side
    d_clamped = torch.clamp(d, min=0)
    outside_distance = torch.norm(d_clamped, dim=-1)
    inside_distance = torch.max(d, dim=-1).values
    sdf_values = outside_distance + torch.min(inside_distance, torch.zeros_like(inside_distance))
    return sdf_values
