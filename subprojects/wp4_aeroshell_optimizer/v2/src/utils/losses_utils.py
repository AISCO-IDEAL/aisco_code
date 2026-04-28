import torch


def get_points(num_points: int, bbox_lo: list[float], bbox_hi: list[float], device: str = "cuda") -> torch.Tensor:
    """Sample points uniformly inside a per-axis bounding box.

    Args:
        num_points: Number of points to sample.
        bbox_lo: Lower bound for each axis [x_lo, y_lo, z_lo] in mm.
        bbox_hi: Upper bound for each axis [x_hi, y_hi, z_hi] in mm.
        device: Device to create the tensor on (default: ``"cuda"``).

    Returns:
        Tensor of shape ``(num_points, 3)`` with ``requires_grad=True``.
    """
    lo = torch.tensor(bbox_lo, dtype=torch.float32, device=device)
    hi = torch.tensor(bbox_hi, dtype=torch.float32, device=device)
    points = lo + torch.rand(num_points, 3, device=device) * (hi - lo)
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
