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
    max_div: float | None = None,
    leinster_temp: float | None = None,
    leinster_q: float | None = None,
) -> Tensor:
    """Diversity loss over SDF fields — negated so minimizing maximizes diversity.

    The `min`/`mean`/`sum` modes are *local*: each shape is scored against its
    neighbours individually. The `leinster` mode is *global*: it scores the whole
    population through an all-pairs similarity kernel (GINN `train/losses.py`),
    which pressures the population to spread evenly across the shape manifold —
    this is what produces a structured/organized latent space.

    Args:
        y: [num_fields, num_points] — SDF values, with autograd enabled for gradient flow.
        norm_order: p for L^p field distance (default 2).
        neighbor_agg_fn: how to aggregate inter-shape distances:
            'min'  — maximize distance to nearest neighbor (local repulsion)
            'mean' — maximize mean distance to all others
            'sum'  — maximize sum of distances to all others
            'leinster' — global all-pairs Leinster diversity (needs leinster_temp/q)
        max_div: optional GINN-style clamp. If set (negative value, e.g. -0.1), the
            loss becomes `clamp(loss - max_div, min=0)` — the diversity gradient turns
            off once shapes are "different enough", preventing the population from
            being pulled further apart at the expense of other objectives.
        leinster_temp: similarity-kernel temperature as a multiple of the mean inter-shape
            field distance (scale-free; ~1.0 is a sensible default). Only used for 'leinster'.
        leinster_q: diversity order q; only used for the 'leinster' mode (q=2 is the
            inverse-Simpson "effective number of distinct shapes").
    Returns:
        Scalar loss (negative → minimizing pushes shapes apart).
    """
    distances = pairwise_dist_fields(y, norm_order=norm_order)  # [N, N]

    if neighbor_agg_fn == "leinster":
        if leinster_temp is None or leinster_q is None:
            raise ValueError("neighbor_agg_fn='leinster' requires leinster_temp and leinster_q")
        # Global all-pairs diversity. The diagonal (self-similarity 1) is kept on
        # purpose — Leinster ordinariness is the mean over the full row.
        #
        # The temperature is taken *relative* to the mean inter-shape field distance, so
        # leinster_temp is scale-free: it is a multiple of "the typical distance between
        # two shapes". With a fixed temperature the mm-scale SDF distances drive every
        # exp() term to 0, pegging diversity at N. The normaliser is detached so it sets
        # the kernel width without itself carrying gradient.
        n_fields = distances.size(0)
        mean_dist = distances.detach().sum() / (n_fields * (n_fields - 1))
        temp = leinster_temp * mean_dist.clamp(min=1e-6)
        similarities = torch.exp(-distances / temp)
        ordinariness = similarities.mean(dim=1)  # [N]
        if leinster_q == 1:
            diversity = torch.exp(-torch.log(ordinariness).mean())
        else:
            diversity = ordinariness.pow(leinster_q - 1).mean().pow(1.0 / (1.0 - leinster_q))
        loss = (-1.0) * diversity
    else:
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
        loss = (-1.0) * dists.sqrt().mean().square()

    if max_div is not None:
        loss = torch.clamp(loss - max_div, min=0.0)
    return loss
