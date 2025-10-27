 ### GNN from Scratch
## Imports
import numpy as np

## Hyperparameters

hyperparams = {
    "l_range": (1.0, 2.0),
    "w_range": (1.0, 2.0),
    "h_range": (1.0, 2.0),
    "noise": 0.1,
}
## Data


# Point cloud generation
def generate_point_cloud(
    num_points: int,
    l_range: tuple[float, float] = (1.0, 2.0),
    w_range: tuple[float, float] = (1.0, 2.0),
    h_range: tuple[float, float] = (1.0, 2.0),
    noise: float = 0.0,
) -> np.ndarray:
    """
    Generates a random point cloud within a cuboid of dimensions (l, w, h).
    l, w, h are sampled i.i.d. from the given ranges.
    PRE: l_range, w_range, h_range are tuples of (min, max) and noise >= 0.
    POST: Returns a numpy array of shape (num_points, 3) representing the point cloud.
    """
    # Sample cuboid dimensions
    l = np.random.uniform(*l_range)
    w = np.random.uniform(*w_range)
    h = np.random.uniform(*h_range)

    # Generate points in [0, 1]^3
    points = np.random.rand(num_points, 3)

    # Scale to cuboid dimensions
    points[:, 0] *= l  # x: length
    points[:, 1] *= w  # y: width
    points[:, 2] *= h  # z: height

    # Add noise
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)

    # Clip to cuboid bounds
    points[:, 0] = np.clip(points[:, 0], 0, l)
    points[:, 1] = np.clip(points[:, 1], 0, w)
    points[:, 2] = np.clip(points[:, 2], 0, h)

    return points


## Performance Metrics


# Solar performance
def solar_performance(w: float, l: float, max_wl: float) -> float:
    """Evaluate solar performance (maximize top area: w*l)."""
    return (w * l) / max_wl  # Normalized to [0, 1]


# Aerodynamic performance
def aerodynamic_performance(w: float, h: float, max_wh: float) -> float:
    """Evaluate aerodynamic performance (minimize frontal area: w*h)."""
    return 1 - ((w * h) / max_wh)  # Invert to maximize performance


## Loss Function
def combined_loss(
    w: float, l: float, h: float, max_wl: float, max_wh: float, alpha: float = 0.5
) -> float:
    """
    Combine solar and aerodynamic performance into a single loss.
    alpha: weight for solar performance (0 <= alpha <= 1)
    """
    solar = solar_performance(w, l, max_wl)
    aero = aerodynamic_performance(w, h, max_wh)
    return (1 - alpha) * solar + alpha * aero  # Weighted sum


## GNN Model

## Training
