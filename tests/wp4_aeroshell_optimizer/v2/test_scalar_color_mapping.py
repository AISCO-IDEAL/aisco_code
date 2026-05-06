"""Tests for scalar_color_mapping utilities (Plan 010 + Plan 012 extension).

Plan 010 tests: map_scalar_to_rgba, map_boolean_mask_to_rgba (including
configurable endpoint RGBA and validation).

Plan 012 tests: map_scalar_to_rgb_mpl — matplotlib-based colormap mapper
required by CentralVisManager for named colormaps (coolwarm, viridis, etc.).

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_scalar_color_mapping.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.scalar_color_mapping import (
    map_boolean_mask_to_rgba,
    map_scalar_to_rgba,
    map_scalar_to_rgb_mpl,
)

# ---------------------------------------------------------------------------
# Plan 010 tests — map_scalar_to_rgba
# ---------------------------------------------------------------------------


class TestMapScalarToRgba:
    def test_output_shape(self) -> None:
        values = np.array([0.0, 0.5, 1.0])
        result = map_scalar_to_rgba(values)
        assert result.shape == (3, 4)
        assert result.dtype == np.uint8

    def test_auto_range_low_is_blue(self) -> None:
        values = np.array([0.0, 1.0])
        result = map_scalar_to_rgba(values)
        # Low value → blue by default
        assert result[0, 2] > result[0, 0]  # blue > red at low end

    def test_auto_range_high_is_red(self) -> None:
        values = np.array([0.0, 1.0])
        result = map_scalar_to_rgba(values)
        # High value → red by default
        assert result[1, 0] > result[1, 2]  # red > blue at high end

    def test_custom_endpoints(self) -> None:
        values = np.array([0.0, 1.0])
        result = map_scalar_to_rgba(values, low_rgba=(0, 255, 0, 255), high_rgba=(255, 0, 0, 255))
        # Low → green
        assert result[0, 1] == 255
        assert result[0, 0] == 0
        # High → red
        assert result[1, 0] == 255
        assert result[1, 1] == 0

    def test_fixed_min_max(self) -> None:
        values = np.array([0.0, 0.5, 1.0])
        result = map_scalar_to_rgba(values, min_val=0.0, max_val=1.0)
        assert result.shape == (3, 4)

    def test_nan_mapped_to_fallback(self) -> None:
        values = np.array([0.0, float("nan"), 1.0])
        result = map_scalar_to_rgba(values)
        np.testing.assert_array_equal(result[1], [127, 127, 127, 255])

    def test_all_nan_raises(self) -> None:
        values = np.array([float("nan"), float("nan")])
        with pytest.raises(ValueError, match="No finite scalar values"):
            map_scalar_to_rgba(values)


class TestMapBooleanMaskToRgba:
    def test_output_shape(self) -> None:
        mask = np.array([True, False, True])
        result = map_boolean_mask_to_rgba(mask)
        assert result.shape == (3, 4)
        assert result.dtype == np.uint8

    def test_true_and_false_colors_differ(self) -> None:
        mask = np.array([True, False])
        result = map_boolean_mask_to_rgba(mask)
        assert not np.array_equal(result[0], result[1])

    def test_custom_colors(self) -> None:
        mask = np.array([True, False])
        result = map_boolean_mask_to_rgba(mask, true_rgba=(255, 0, 0, 255), false_rgba=(0, 255, 0, 255))
        np.testing.assert_array_equal(result[0], [255, 0, 0, 255])
        np.testing.assert_array_equal(result[1], [0, 255, 0, 255])


# ---------------------------------------------------------------------------
# Plan 012 extension tests — map_scalar_to_rgb_mpl
# ---------------------------------------------------------------------------


class TestMapScalarToRgbMpl:
    """Tests for the matplotlib-based colormap mapper used by CentralVisManager."""

    def test_output_shape_and_dtype(self) -> None:
        values = np.array([0.0, 0.5, 1.0])
        result = map_scalar_to_rgb_mpl(values, "viridis")
        assert result.shape == (3, 3), f"Expected (3, 3), got {result.shape}"
        assert result.dtype == np.float32

    def test_output_in_unit_range(self) -> None:
        rng = np.random.default_rng(42)
        values = rng.uniform(-5.0, 5.0, size=50)
        result = map_scalar_to_rgb_mpl(values, "coolwarm")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_coolwarm_midpoint_is_near_white(self) -> None:
        # coolwarm: blue at low, white at midpoint, red at high
        values = np.array([0.0, 0.5, 1.0])
        result = map_scalar_to_rgb_mpl(values, "coolwarm", vmin=0.0, vmax=1.0)
        mid = result[1]  # midpoint should be near white (all channels high)
        assert mid.min() > 0.7, f"Midpoint of coolwarm should be near white, got {mid}"

    def test_fixed_vmin_vmax_applied(self) -> None:
        # When vmin/vmax are fixed, values at extremes should produce distinct colors
        low_val = np.array([-2.0])
        high_val = np.array([1.0])
        c_low = map_scalar_to_rgb_mpl(low_val, "coolwarm", vmin=-2.0, vmax=1.0)
        c_high = map_scalar_to_rgb_mpl(high_val, "coolwarm", vmin=-2.0, vmax=1.0)
        # Low should be bluish (R < B), high should be reddish (R > B)
        assert c_low[0, 2] > c_low[0, 0], "Low value should be blue-dominant in coolwarm"
        assert c_high[0, 0] > c_high[0, 2], "High value should be red-dominant in coolwarm"

    def test_auto_range_with_none_vmin_vmax(self) -> None:
        values = np.array([10.0, 15.0, 20.0])
        result = map_scalar_to_rgb_mpl(values, "viridis")
        # The full viridis range is spanned: first and last should differ
        assert not np.allclose(result[0], result[-1], atol=0.01)

    def test_nonfinite_values_map_to_grey(self) -> None:
        values = np.array([0.0, float("nan"), float("inf"), 1.0])
        result = map_scalar_to_rgb_mpl(values, "viridis")
        # nan and inf entries should be grey (0.5, 0.5, 0.5)
        np.testing.assert_allclose(result[1], [0.5, 0.5, 0.5], atol=0.01)
        np.testing.assert_allclose(result[2], [0.5, 0.5, 0.5], atol=0.01)

    def test_unknown_colormap_raises(self) -> None:
        values = np.array([0.0, 1.0])
        with pytest.raises((KeyError, ValueError)):
            map_scalar_to_rgb_mpl(values, "not_a_real_colormap_xyzzy")

    def test_flat_input_works(self) -> None:
        scalar = np.float32(0.3)
        result = map_scalar_to_rgb_mpl(scalar, "plasma")
        assert result.shape == (1, 3)

    def test_2d_input_flattened(self) -> None:
        values = np.array([[0.0, 0.5], [1.0, 0.25]])
        result = map_scalar_to_rgb_mpl(values, "viridis")
        assert result.shape == (4, 3)

    def test_plasma_colormap_accepted(self) -> None:
        values = np.linspace(0, 1, 10)
        result = map_scalar_to_rgb_mpl(values, "plasma")
        assert result.shape == (10, 3)

    def test_reds_colormap_accepted(self) -> None:
        values = np.linspace(0, 1, 5)
        result = map_scalar_to_rgb_mpl(values, "Reds")
        assert result.shape == (5, 3)
