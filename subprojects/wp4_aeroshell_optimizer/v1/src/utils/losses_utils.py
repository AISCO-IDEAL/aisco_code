import torch


def get_points(num_points: int = 1000, bbox_size: float = 10.0) -> torch.Tensor:
    """
    Sample points in the bounding box of the model.
    Args:
        num_points: Number of points to sample
        bbox_size: Size of the bounding box (assumed cubic, centered at origin)
    Returns:
        points: Tensor of shape (num_points, 3) with sampled points
    """
    points = (
        torch.rand(num_points, 3, requires_grad=True, device="cuda") - 0.5
    ) * bbox_size
    return points
