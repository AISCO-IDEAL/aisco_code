"""Latent-space utilities for population-based pretraining (Plan 005 / v0.3.0)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def build_unit_circle_Z(  # noqa: N802  # Z is conventional notation for latent code tensors in DeepSDF/GINN literature
    num_shapes: int,
    z_dim: int,
    radius: float,
    device: str | torch.device,
) -> Tensor:
    """Build a fixed population of latent vectors arranged on a unit circle.

    z₀ is always placed at the origin (consistent with force_z0_to_origin=True).
    Remaining z₁ … z_{N-1} are placed equidistantly on a circle of the given
    radius in the (z[0], z[1]) plane; all other dimensions are set to zero.

    Args:
        num_shapes: Total number of latent slots N (including z₀ at origin).
        z_dim: Dimensionality of each latent vector. Must be ≥ 2 for circle placement.
        radius: Radius of the circle. Typically 1.0.
        device: PyTorch device string or device object.

    Returns:
        Tensor of shape (num_shapes, z_dim) with no gradient.
    """
    if z_dim < 2 and num_shapes > 1:
        raise ValueError(
            f"z_dim must be ≥ 2 to place shapes on a circle, got z_dim={z_dim}. Reduce num_shapes to 1 or increase z_dim."
        )

    Z = torch.zeros(num_shapes, z_dim, device=device, dtype=torch.float32)

    if num_shapes == 1:
        return Z

    # z₁ … z_{N-1} placed equidistantly at 360/(N-1) degree intervals.
    n_circle = num_shapes - 1
    for k in range(n_circle):
        angle = 2.0 * math.pi * k / n_circle
        Z[k + 1, 0] = radius * math.cos(angle)
        Z[k + 1, 1] = radius * math.sin(angle)

    return Z
