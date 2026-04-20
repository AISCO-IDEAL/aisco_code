"""Comprehensive tests for universal null losses.

This module tests that:
1. All nulllosses can be dynamically loaded
2. All nulllosses return zero without gradients
3. Nulllosses have minimal overhead (cached tensor)
4. Nulllosses work correctly with ALM (Augmented Lagrangian Method)
5. Multiple nulllosses can be active simultaneously
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.universal_null_losses import (  # noqa: E402
    NullLossAeroV1,
    NullLossBbV1,
    NullLossDevelopabilityV0,
    NullLossEikonalV0,
    NullLossSmoothnessV0,
    NullLossSolarV1,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config  # noqa: E402
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import (  # noqa: E402
    load_class_from_path,
)


# Fixtures
@pytest.fixture
def device():
    """Provide device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def losses_config():
    """Load losses configuration."""
    return load_config(v="v2", cfg_name="losses_v0_config.yaml")


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""

    class MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.randn(10, 10))

        def forward(self, x):
            return self.param @ x

    return MockModel()


# Test 1: Direct Instantiation
class TestDirectInstantiation:
    """Test that all nulllosses can be directly instantiated."""

    @pytest.mark.parametrize(
        "loss_class,loss_name",
        [
            (NullLossAeroV1, "null_loss_aero_v1"),
            (NullLossSolarV1, "null_loss_solar_v1"),
            (NullLossBbV1, "null_loss_bb_v1"),
            (NullLossDevelopabilityV0, "null_loss_developability_v0"),
            (NullLossEikonalV0, "null_loss_eikonal_v0"),
            (NullLossSmoothnessV0, "null_loss_smoothness_v0"),
        ],
    )
    def test_instantiation(self, loss_class, loss_name, device, losses_config):
        """Test that nulllosses can be instantiated."""
        loss = loss_class(
            device=device,
            loss_name=loss_name,
            losses_config=losses_config,
            weight=[1.0, 1.0],  # Force initialization even if config has compute=0
        )
        assert loss is not None
        assert hasattr(loss, "_zero"), f"{loss_class.__name__} missing _zero tensor (setup not called)"


# Test 2: Zero Return Value
class TestZeroReturn:
    """Test that all nulllosses return zero."""

    @pytest.mark.parametrize(
        "loss_class,loss_name",
        [
            (NullLossAeroV1, "null_loss_aero_v1"),
            (NullLossSolarV1, "null_loss_solar_v1"),
            (NullLossBbV1, "null_loss_bb_v1"),
            (NullLossDevelopabilityV0, "null_loss_developability_v0"),
            (NullLossEikonalV0, "null_loss_eikonal_v0"),
            (NullLossSmoothnessV0, "null_loss_smoothness_v0"),
        ],
    )
    def test_returns_zero(self, loss_class, loss_name, device, losses_config, mock_model):
        """Test that nulllosses return exactly zero."""
        loss = loss_class(
            device=device,
            loss_name=loss_name,
            losses_config=losses_config,
            weight=[1.0, 1.0],  # Enable the loss
        )

        result = loss.get_loss(mock_model, epoch_name=0)

        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0
        assert result.device.type == device


# Test 3: Gradient Tracking
class TestGradientTracking:
    """Test that nulllosses don't track gradients."""

    @pytest.mark.parametrize(
        "loss_class,loss_name",
        [
            (NullLossAeroV1, "null_loss_aero_v1"),
            (NullLossSolarV1, "null_loss_solar_v1"),
            (NullLossBbV1, "null_loss_bb_v1"),
            (NullLossDevelopabilityV0, "null_loss_developability_v0"),
        ],
    )
    def test_no_gradient_tracking(self, loss_class, loss_name, device, losses_config, mock_model):
        """Test that nulllosses don't create gradient graphs."""
        loss = loss_class(
            device=device,
            loss_name=loss_name,
            losses_config=losses_config,
            weight=[1.0, 1.0],
        )

        result = loss.get_loss(mock_model, epoch_name=0)

        # The result should not require gradients
        assert not result.requires_grad


# Test 4: Tensor Caching
class TestTensorCaching:
    """Test that nulllosses cache their zero tensor."""

    def test_cached_tensor_reuse(self, device, losses_config):
        """Test that the same tensor is returned on multiple calls."""
        loss = NullLossSolarV1(
            device=device,
            loss_name="null_loss_solar_v1",
            losses_config=losses_config,
            weight=[1.0, 1.0],
        )

        # Get tensor multiple times
        tensor1 = loss._zero
        tensor2 = loss._zero
        tensor3 = loss._zero

        # All should be the same object (cached)
        assert tensor1 is tensor2
        assert tensor2 is tensor3


# Test 5: Dynamic Loading
class TestDynamicLoading:
    """Test that nulllosses can be dynamically loaded."""

    @pytest.mark.parametrize(
        "loss_name,class_name",
        [
            ("null_loss_aero_v1", "NullLossAeroV1"),
            ("null_loss_solar_v1", "NullLossSolarV1"),
            ("null_loss_bb_v1", "NullLossBbV1"),
            ("null_loss_developability_v0", "NullLossDevelopabilityV0"),
            ("null_loss_eikonal_v0", "NullLossEikonalV0"),
            ("null_loss_smoothness_v0", "NullLossSmoothnessV0"),
        ],
    )
    def test_dynamic_loading(self, loss_name, class_name):  # noqa: ARG002
        """Test that nulllosses can be loaded dynamically from universal file.

        Args:
            loss_name: Loss name for test identification (unused in test body)
            class_name: Class name to dynamically load
        """
        universal_path = str(project_root / "subprojects/wp4_aeroshell_optimizer/v2/src/losses/universal_null_losses.py")

        # Load class dynamically
        loss_cls = load_class_from_path(universal_path, class_name)

        assert loss_cls is not None
        assert loss_cls.__name__ == class_name


# Test 6: Weight=0 Early Return
class TestEarlyReturn:
    """Test that ParentLoss early return works for weight=0."""

    def test_weight_zero_early_return(self, device, losses_config, mock_model):
        """Test that weight=0 causes early return without computation."""
        loss = NullLossSolarV1(
            device=device,
            loss_name="null_loss_solar_v1",
            losses_config=losses_config,
            weight=[0.0, 0.0],  # Disabled
        )

        result = loss.get_loss(mock_model, epoch_name=0)

        assert result.item() == 0.0
        # This should use the ParentLoss early return, not _loss()


# Test 7: ALM Compatibility
class TestALMCompatibility:
    """Test that nulllosses work correctly with ALM."""

    def test_alm_constraint_handling(self, device, losses_config, mock_model):
        """Test that constraint nulllosses work with ALM system."""
        # Create constraint nulllosses (bb, eikonal, smoothness, developability)
        constraint_losses = [
            NullLossBbV1(
                device=device,
                loss_name="null_loss_bb_v1",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
            NullLossEikonalV0(
                device=device,
                loss_name="null_loss_eikonal_v0",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
            NullLossSmoothnessV0(
                device=device,
                loss_name="null_loss_smoothness_v0",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
            NullLossDevelopabilityV0(
                device=device,
                loss_name="null_loss_developability_v0",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
        ]

        # All should return zero
        constraint_values = []
        for loss in constraint_losses:
            result = loss.get_loss(mock_model, epoch_name=0)
            constraint_values.append(result.item())

        # Create ALM constraint vector
        constraint_vector = torch.tensor(constraint_values, device=device)

        # With all zeros, ALM penalty should be zero
        alm_lambda = torch.tensor([0.01, 0.01, 0.01, 0.01], device=device)
        alm_mu = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device)

        # ALM penalty: sum(lambda * c + 0.5 * mu * c^2)
        alm_penalty = torch.sum(alm_lambda * constraint_vector + 0.5 * alm_mu * constraint_vector**2)

        assert alm_penalty.item() == 0.0


# Test 8: Multiple Nulllosses Simultaneously
class TestMultipleNulllosses:
    """Test that multiple nulllosses can be active simultaneously."""

    def test_multiple_active_nulllosses(self, device, losses_config, mock_model):
        """Test that multiple nulllosses can coexist."""
        losses = {
            "aero": NullLossAeroV1(
                device=device,
                loss_name="null_loss_aero_v1",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
            "solar": NullLossSolarV1(
                device=device,
                loss_name="null_loss_solar_v1",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
            "bb": NullLossBbV1(
                device=device,
                loss_name="null_loss_bb_v1",
                losses_config=losses_config,
                weight=[1.0, 1.0],
            ),
        }

        results = {}
        for name, loss in losses.items():
            results[name] = loss.get_loss(mock_model, epoch_name=0)

        # All should return zero
        for name, result in results.items():
            assert result.item() == 0.0, f"{name} did not return zero"


# Test 9: Type Distinctness
class TestTypeDistinctness:
    """Test that each nullloss has a distinct type."""

    def test_distinct_types(self):
        """Test that each nullloss class is distinct."""
        classes = [
            NullLossAeroV1,
            NullLossSolarV1,
            NullLossBbV1,
            NullLossDevelopabilityV0,
            NullLossEikonalV0,
            NullLossSmoothnessV0,
        ]

        # All classes should be distinct
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes):
                if i != j:
                    assert cls1 is not cls2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
