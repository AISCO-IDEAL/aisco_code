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
    points = (torch.rand(num_points, 3, device="cuda") - 0.5) * bbox_size
    return points.detach().requires_grad_(True)


def deep_update(base, updates):
    """
    Recursively updates base dict with updates dict.
    Supports dot notation keys: e.g., 'training_params.learning_rate'.
    """
    for key, value in updates.items():
        if "." in key:
            # Split and insert recursively
            parts = key.split(".")
            d = base
            for part in parts[:-1]:
                if part not in d or not isinstance(d[part], dict):
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        elif isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base
