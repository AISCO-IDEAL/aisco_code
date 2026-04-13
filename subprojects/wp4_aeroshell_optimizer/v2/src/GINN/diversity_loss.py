"""Field-based diversity loss adapted from GINN (arXiv:2402.14009, Eq. 26)."""

import torch
from torch import Tensor


def pairwise_dist_fields(y: Tensor, norm_order: int, weights: Tensor | None = None) -> Tensor:
    """Pairwise L^p distance between SDF fields sampled at the same points.

    Uses torch.linalg.vector_norm for numerical stability (avoids NaNs from
    .square().mean().sqrt() pattern).

    Args:
        y: [num_fields, num_points] — SDF values per shape per sampled point.
        norm_order: p in L^p norm.
        weights: optional [num_points] importance weights (must sum to 1).
    Returns:
        [num_fields, num_fields] symmetric matrix of pairwise distances.
    """
    _n_fields, n_points = y.shape
    diff = y.unsqueeze(0) - y.unsqueeze(1)  # [N, N, P]
    weighted_diff = diff if weights is None else diff * weights ** (1 / norm_order)
    pairwise_dist = torch.linalg.vector_norm(weighted_diff, ord=norm_order, dim=-1)  # [N, N]
    if weights is None:
        pairwise_dist = pairwise_dist / (n_points ** (1 / norm_order))
    return pairwise_dist


def field_diversity_loss(
    y: Tensor,
    norm_order: int = 2,
    neighbor_agg_fn: str = "min",
) -> Tensor:
    """Diversity loss over SDF fields — negated so minimizing maximizes diversity.

    For each shape, aggregates distances to other shapes (default: nearest neighbor).
    Returns -(sqrt(dist).mean())^2, which gradient descent pushes toward more separated shapes.

    Args:
        y: [num_fields, num_points] — SDF values, with autograd enabled for gradient flow.
        norm_order: p for L^p field distance (default 2).
        neighbor_agg_fn: how to aggregate inter-shape distances per shape:
            'min'  — maximize distance to nearest neighbor (most principled, matches GINN paper)
            'mean' — maximize mean distance to all others
            'sum'  — maximize sum of distances to all others
    Returns:
        Scalar loss (negative → minimizing pushes shapes apart).
    """
    distances = pairwise_dist_fields(y, norm_order=norm_order)  # [N, N]

    if neighbor_agg_fn == "min":
        # Mask diagonal so each shape doesn't compare to itself
        mask = torch.eye(distances.size(0), dtype=torch.bool, device=y.device)
        distances = distances.masked_fill(mask, float("inf"))
        dists, _ = distances.min(dim=1)  # [N] — distance to nearest neighbour
    elif neighbor_agg_fn == "mean":
        dists = distances.mean(dim=1)
    elif neighbor_agg_fn == "sum":
        dists = distances.sum(dim=1)
    else:
        raise NotImplementedError(f"neighbor_agg_fn '{neighbor_agg_fn}' not implemented")

    # sqrt prevents one large distance from dominating; .mean().square() for smooth gradient
    return (-1.0) * dists.sqrt().mean().square()
