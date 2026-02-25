from collections.abc import (
    Callable,
)

import torch


def create_bbox_grid(bbox: list[list[int]] | None = None, resolution: int = 64, device: torch.device = "cuda") -> torch.Tensor:
    """
    Creates a 3D grid of torch points within the specified bounding box and resolution. on requested device
    """
    if bbox is None:
        bbox = [[-5000, 5000], [-5000, 5000], [-5000, 5000]]

    grid = torch.stack(
        torch.meshgrid(
            torch.linspace(bbox[0][0], bbox[0][1], resolution, device=device),
            torch.linspace(bbox[1][0], bbox[1][1], resolution, device=device),
            torch.linspace(bbox[2][0], bbox[2][1], resolution, device=device),
        ),
        dim=-1,
    )
    # Format into (N, 3) points
    grid = grid.reshape(-1, 3)
    return grid


def sample_sdf(sdf_model: Callable[[torch.Tensor], torch.Tensor], grid: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    """
    Samples the SDF function on the provided grid (made by create_bbox_grid/meshgrid) in batches to avoid OOM errors
    Args:
        sdf_model: Callable SDF function that takes in (N, 3) points and returns (N, 1) SDF values
        grid: (M, 3) grid of points to sample the SDF on
        batch_size: Number of points to sample in each batch

    Returns:
        sdf_values: (M, 1) SDF values corresponding to the input grid points
        Needs to be reshaped and renormalized by renormalise_vertices and sdf_samples_to_grid
    """
    sdf_values = []
    num_points = grid.shape[0]
    for i in range(0, num_points, batch_size):
        batch_points = grid[i : i + batch_size]
        batch_sdf = sdf_model(batch_points)
        sdf_values.append(batch_sdf)
    sdf_values = torch.cat(sdf_values, dim=0)
    return sdf_values


def sdf_samples_to_grid(sdf_values: torch.Tensor, resolution: int = 64) -> torch.Tensor:
    """
    Reshapes the sampled SDF values back into a 3D grid format for diso extractors
    Args:
        sdf_values: (M, 1) SDF values sampled on a grid
        resolution: Resolution of the original grid used for sampling
    Returns:
        sdf_grid: (resolution, resolution, resolution) reshaped SDF grid suitable for diso extractors
    """
    sdf_grid = sdf_values.reshape(resolution, resolution, resolution)
    return sdf_grid


def renormalise_vertices(points: torch.Tensor, bbox: list[list[int]] | None = None) -> torch.Tensor:
    """
    Normalises vertices that are the output of diso extractors back to the dimensions of original bounding box
    """
    if bbox is None:
        bbox = [[-5000, 5000], [-5000, 5000], [-5000, 5000]]

    scale = [(bbox[0][1] - bbox[0][0]), (bbox[1][1] - bbox[1][0]), (bbox[2][1] - bbox[2][0])]
    offset = [bbox[0][0], bbox[1][0], bbox[2][0]]
    points = points * torch.tensor(scale, device=points.device) + torch.tensor(offset, device=points.device)
    return points
