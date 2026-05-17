"""TDD tests for three visualization fixes:

A) Per-slot grad payloads populated in _backward_l2_single_pass and _backward_l2_two_pass.
B) solar vis enabled: compute() skip_vis parameter propagated correctly.
C) Interval enforcement: vis_type_config respects epoch % interval.

Run these tests BEFORE implementing the fixes to verify they fail for the right reason,
then again after to confirm they pass.
"""

from __future__ import annotations

import inspect

import pytest

from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import (
    DiversityTrainer,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss_v2 import (
    TotalLossV2,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.vis_config_helpers import (
    vis_is_enabled,
    vis_type_config,
)

# ---------------------------------------------------------------------------
# Change C: vis_type_config interval enforcement
# ---------------------------------------------------------------------------


class TestVisTypeConfigIntervalEnforcement:
    """vis_type_config must respect epoch % interval when epoch is supplied."""

    def _make_visual_types(self, interval: int = 10) -> dict:
        return {
            "mesh_shape": {
                "enabled": True,
                "interval": interval,
                "colormap": "viridis",
            }
        }

    def test_epoch_none_no_interval_check(self) -> None:
        """epoch=None (default) → no interval check → returns config."""
        vt = self._make_visual_types(interval=10)
        cfg = vis_type_config(vt, "mesh_shape")
        assert cfg is not None

    def test_epoch_on_boundary_returns_config(self) -> None:
        """epoch=10, interval=10 → 10%10==0 → returns config."""
        vt = self._make_visual_types(interval=10)
        cfg = vis_type_config(vt, "mesh_shape", epoch=10)
        assert cfg is not None

    def test_epoch_off_boundary_returns_none(self) -> None:
        """epoch=5, interval=10 → 5%10!=0 → returns None (not due)."""
        vt = self._make_visual_types(interval=10)
        cfg = vis_type_config(vt, "mesh_shape", epoch=5)
        assert cfg is None

    def test_epoch_zero_is_always_due(self) -> None:
        """epoch=0 → 0%interval==0 → always due."""
        vt = self._make_visual_types(interval=10)
        cfg = vis_type_config(vt, "mesh_shape", epoch=0)
        assert cfg is not None

    def test_interval_one_always_due(self) -> None:
        """interval=1 → every epoch is due."""
        vt = self._make_visual_types(interval=1)
        for epoch in (0, 1, 7, 100):
            cfg = vis_type_config(vt, "mesh_shape", epoch=epoch)
            assert cfg is not None, f"epoch={epoch} should be due with interval=1"

    def test_disabled_type_still_returns_none_regardless_of_epoch(self) -> None:
        """Disabled type must return None even when epoch is on a boundary."""
        vt = {"mesh_shape": {"enabled": False, "interval": 10}}
        cfg = vis_type_config(vt, "mesh_shape", epoch=10)
        assert cfg is None

    def test_vis_is_enabled_respects_epoch(self) -> None:
        """vis_is_enabled(epoch=5) with interval=10 must return False."""
        vt = self._make_visual_types(interval=10)
        assert vis_is_enabled(vt, "mesh_shape", epoch=5) is False
        assert vis_is_enabled(vt, "mesh_shape", epoch=10) is True


# ---------------------------------------------------------------------------
# Change B: TotalLossV2.compute() skip_vis parameter
# ---------------------------------------------------------------------------


class TestTotalLossComputeSkipVis:
    """compute() must forward skip_vis → skip_telemetry to loss instances."""

    def test_compute_has_skip_vis_parameter(self) -> None:
        """TotalLossV2.compute must accept skip_vis keyword argument."""
        sig = inspect.signature(TotalLossV2.compute)
        assert "skip_vis" in sig.parameters, "TotalLossV2.compute() must have a skip_vis parameter. Add: skip_vis: bool = False"

    def test_skip_vis_default_is_false(self) -> None:
        """skip_vis must default to False (vis enabled by default)."""
        sig = inspect.signature(TotalLossV2.compute)
        param = sig.parameters.get("skip_vis")
        assert param is not None
        assert param.default is False, f"skip_vis default must be False, got {param.default!r}"


# ---------------------------------------------------------------------------
# Change A: per-slot grad payloads populated in l2 backward paths
# ---------------------------------------------------------------------------


class TestL2BackwardPerSlotPayloads:
    """_backward_l2_single_pass and _backward_l2_two_pass must populate per_slot_grad_payloads."""

    @pytest.mark.parametrize("method_name", ["_backward_l2_single_pass", "_backward_l2_two_pass"])
    def test_per_slot_payloads_populated(self, method_name: str) -> None:
        """l2 backward methods must populate per_slot_grad_payloads when vis_all_losses=True."""
        # Check that the method fills per_slot_grad_payloads by inspecting source.
        source = inspect.getsource(getattr(DiversityTrainer, method_name))

        assert "per_slot_grad_payloads.append" in source, (
            f"{method_name} must contain 'per_slot_grad_payloads.append(...)'. "
            f"Collection block is missing — add it after compute() in the slot loop."
        )

    def test_l2_two_pass_pass1_has_skip_vis(self) -> None:
        """_backward_l2_two_pass Pass 1 (no_grad) must call compute with skip_vis=True."""
        source = inspect.getsource(DiversityTrainer._backward_l2_two_pass)

        # The Pass 1 no_grad block must suppress vis
        assert "skip_vis=True" in source, (
            "_backward_l2_two_pass Pass 1 must call compute(..., skip_vis=True) to suppress "
            "vis writes during the no_grad pass (no slot is set in Pass 1)."
        )
