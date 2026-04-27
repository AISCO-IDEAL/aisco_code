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
    """Null loss for solar energy objectives, returns a default so the CSE can work properly"""

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

        if self.losses_config["general_info"]["CSE"].lower() == "true":
            if configured_value != 1.0:
                print(
                    f"Warning: Since CSE is enabled, should the solar loss be turned off via the _disabled_epochs key, it stores the CSE default value specified in the cse_defaults key for cruising speed estimation. {self.loss_name} weight is {configured_value}, not 1.0., which indicates that you attempt to override the default value used. Make sure this is intentional."
                )
            # override the configured value with the default
            configured_value = self.losses_config["general_info"]["cse_defaults"]["solar"]
        self._zero = torch.tensor(configured_value, device=self.device, requires_grad=False)

    def _loss(self, model) -> torch.Tensor:  # noqa: ARG002
        """Return cached constant tensor.

        Args:
            model: The model (unused in null loss)

        Returns:
            Constant tensor without gradients
        """
        return self._zero


class NullLossAero(_BaseNullLoss):
    """Null loss for solar energy objectives, returns a default so the CSE can work properly"""

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

        if self.losses_config["general_info"]["CSE"].lower() == "true":
            if configured_value != 1.0:
                print(
                    f"Warning: Since CSE is enabled, should the drag loss be turned off via the _disabled_epochs key, it stores the CSE default value specified in the cse_defaults key for cruising speed estimation. {self.loss_name} weight is {configured_value}, not 1.0., which indicates that you attempt to override the default value used. Make sure this is intentional."
                )
            # override the configured value with the default
            configured_value = self.losses_config["general_info"]["cse_defaults"]["drag"]
        self._zero = torch.tensor(configured_value, device=self.device, requires_grad=False)

    def _loss(self, model) -> torch.Tensor:  # noqa: ARG002
        """Return cached constant tensor.

        Args:
            model: The model (unused in null loss)

        Returns:
            Constant tensor without gradients
        """
        return self._zero


class NullLossBb(_BaseNullLoss):
    """Null loss for bounding box constraints."""


class NullLossDevelopability(_BaseNullLoss):
    """Null loss for developability constraints.

    Returns a constant (zero gradient) during training. Optionally delegates
    post-optimization to a real DevelopabilityV0 instance when
    `developability_config.post_opt_enabled: True` is set in the losses config.
    """

    def _setup(self) -> None:
        """Initialize null loss with post-opt delegation if configured."""
        super()._setup()
        dev_cfg = self.losses_config.get("developability_config", {})
        self.post_opt_enabled: bool = bool(dev_cfg.get("post_opt_enabled", False))

        if self.post_opt_enabled:
            from subprojects.wp4_aeroshell_optimizer.v2.src.losses.developability_loss.developability_v0 import (  # noqa: PLC0415
                DevelopabilityV0,
            )

            # Read the real weight from config; force compute=1 to bypass parent_loss early-return
            dev_entry = self.losses_config.get("losses", {}).get("developability", {}).get("developability_v0", [0, 1.0])
            delegate_weight = [1, float(dev_entry[1]) if len(dev_entry) >= 2 else 1.0]
            self._delegate = DevelopabilityV0(
                device=self.device,
                loss_name="developability_v0",
                losses_config=self.losses_config,
                weight=delegate_weight,
            )
            self.mesh_extractor = self._delegate.mesh_extractor
            self.mesh_isovalue = self._delegate.mesh_isovalue
            self.mesh_batch_size = self._delegate.mesh_batch_size

    def post_optimize(self, vertices, faces, run_name: str, storage_dir: str) -> dict:
        """Delegate post-optimization to DevelopabilityV0."""
        return self._delegate.post_optimize(vertices, faces, run_name, storage_dir)


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
