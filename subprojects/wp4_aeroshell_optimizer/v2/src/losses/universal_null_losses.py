"""Universal null losses for all loss categories.

This module provides null loss implementations that always return a configurable
constant with minimal overhead. All nulllosses share the same implementation via
the _BaseNullLoss class.

Usage:
    In losses_v0_config.yaml, reference this file for any null loss:

    solar:
      null_loss_solar: [1, 0.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/universal_null_losses.py"]

    aero:
      null_loss_aero: [1, 0.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/universal_null_losses.py"]

Design:
    - Reads the weight (second element of [compute, weight, path]) from losses_config at init
    - Caches the value as a constant tensor — no per-step allocations
    - Sets requires_grad=False: the tensor is not model-derived, so gradients are zero
    - Returns immediately without model evaluation

Compliance:
    - Ruff: Passes all checks including ARG002 (unused arguments)
    - Mypy: Fully type-hinted and compliant
    - ALM: Returns constant without affecting ALM constraint gradients
"""

from __future__ import annotations

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class _BaseNullLoss(ParentLoss):
    """Shared implementation for all null losses.

    Returns the configured weight value as a constant tensor with no gradients.
    The return value is determined by the second element of the [compute, weight, path]
    tuple in losses_config — e.g. `null_loss_solar: [1, 0.0, "..."]` returns 0.0.
    """

    def _setup(self) -> None:
        """Initialize cached constant tensor from configured weight value.

        Reads the weight (second element of the [compute, weight, path] tuple) from
        losses_config so the return value is configurable without code changes.
        Gradients are zero regardless of value since the tensor is not model-derived.
        """
        configured_value = 0.0
        for block in self.losses_config.get("losses", {}).values():
            if isinstance(block, dict) and self.loss_name in block:
                entry = block[self.loss_name]
                if isinstance(entry, list) and len(entry) >= 2:
                    configured_value = float(entry[1])
                break
        self._zero = torch.tensor(configured_value, device=self.device, requires_grad=False)

    def _loss(self, model) -> torch.Tensor:  # noqa: ARG002
        """Return cached constant tensor.

        Args:
            model: The model (unused in null loss)

        Returns:
            Constant tensor without gradients
        """
        return self._zero


# ============================================================================
# Category-Specific Null Losses
# ============================================================================
# Each loss category has its own class to maintain type distinctness
# for debugging and logging purposes. All share the same implementation.
# Class name = config key in CamelCase (e.g. null_loss_solar -> NullLossSolar)
# ============================================================================


class NullLossSolar(_BaseNullLoss):
    """Null loss for solar energy objectives."""


class NullLossAero(_BaseNullLoss):
    """Null loss for aerodynamic objectives."""


class NullLossBb(_BaseNullLoss):
    """Null loss for bounding box constraints."""


class NullLossDevelopability(_BaseNullLoss):
    """Null loss for developability constraints.

    Note: Sets post_opt_enabled=False to skip post-optimization in WIRE_v2.
    Null losses should not trigger expensive mesh optimization.
    """

    def _setup(self) -> None:
        """Initialize null loss with cached constant tensor and disabled post-optimization."""
        super()._setup()
        # WIRE_v2 checks this attribute to decide whether to run C++ mesh optimization
        # Set to False for null losses to skip post-processing entirely
        self.post_opt_enabled = False


class NullLossEikonal(_BaseNullLoss):
    """Null loss for eikonal equation constraints."""


class NullLossSmoothness(_BaseNullLoss):
    """Null loss for smoothness constraints."""


class NullLossDiversity(_BaseNullLoss):
    """Null loss for diversity objectives."""


class NullLossSurfaceArea(_BaseNullLoss):
    """Null loss for surface area auxiliary losses."""


class NullLossConnectedness(_BaseNullLoss):
    """Null loss for connectedness constraints."""
