"""Universal null losses for all loss categories.

This module provides null loss implementations that always return zero with minimal overhead.
All nulllosses share the same optimized implementation via the _BaseNullLoss class.

Usage:
    In losses_v0_config.yaml, reference this file for any null loss:

    solar:
      null_loss_solar_v1: [0, 0.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/universal_null_losses.py"]

    aero:
      null_loss_aero_v1: [0, 0.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/universal_null_losses.py"]

Design:
    - Caches a zero tensor to avoid repeated allocations
    - Explicitly sets requires_grad=False to prevent gradient tracking
    - Returns immediately without model evaluation
    - All category-specific classes inherit from _BaseNullLoss for consistency

Compliance:
    - Ruff: Passes all checks including ARG002 (unused arguments)
    - Mypy: Fully type-hinted and compliant
    - ALM: Returns zero without affecting ALM constraint handling
"""

from __future__ import annotations

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class _BaseNullLoss(ParentLoss):
    """Shared implementation for all null losses.

    This base class provides the optimized zero-return implementation
    that all category-specific null losses inherit from.
    """

    def _setup(self) -> None:
        """Initialize cached zero tensor to avoid repeated allocations."""
        # Cache a zero tensor without gradient tracking for maximum efficiency
        self._zero = torch.tensor(0.0, device=self.device, requires_grad=False)

    def _loss(self, model) -> torch.Tensor:  # noqa: ARG002
        """Return cached zero tensor.

        Args:
            model: The model (unused in null loss)

        Returns:
            Zero tensor without gradients
        """
        return self._zero


# ============================================================================
# Category-Specific Null Losses
# ============================================================================
# Each loss category has its own class to maintain type distinctness
# for debugging and logging purposes. All share the same implementation.
# ============================================================================


class NullLossSolarV1(_BaseNullLoss):
    """Null loss for solar energy objectives."""


class NullLossAeroV1(_BaseNullLoss):
    """Null loss for aerodynamic objectives."""


class NullLossBbV1(_BaseNullLoss):
    """Null loss for bounding box constraints."""


class NullLossDevelopabilityV0(_BaseNullLoss):
    """Null loss for developability constraints.

    Note: Sets post_opt_enabled=False to skip post-optimization in WIRE_v2.
    Null losses should not trigger expensive mesh optimization.
    """

    def _setup(self) -> None:
        """Initialize null loss with cached zero tensor and disabled post-optimization."""
        super()._setup()
        # WIRE_v2 checks this attribute to decide whether to run C++ mesh optimization
        # Set to False for null losses to skip post-processing entirely
        self.post_opt_enabled = False


class NullLossEikonalV0(_BaseNullLoss):
    """Null loss for eikonal equation constraints."""


class NullLossSmoothnessV0(_BaseNullLoss):
    """Null loss for smoothness constraints."""


class NullLossDiversityV0(_BaseNullLoss):
    """Null loss for diversity objectives."""


class NullLossSurfaceAreaV0(_BaseNullLoss):
    """Null loss for surface area auxiliary losses."""
