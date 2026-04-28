"""Tests for wp2 reg_imp utility functions.

Run:
    pytest tests/wp2_assembler/test_reg_imp.py -v
"""

from __future__ import annotations

import pytest

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm


class TestGetOuterPrismBoundsMm:
    """Tests for get_outer_prism_bounds_mm() — single source of truth for car domain bounds."""

    def test_returns_required_keys(self) -> None:
        bounds = get_outer_prism_bounds_mm()
        assert set(bounds.keys()) == {"x", "y", "z_sampling", "z_mesh"}

    def test_x_bounds(self) -> None:
        """X-axis (length): symmetric around origin, half of 5800mm."""
        bounds = get_outer_prism_bounds_mm()
        assert bounds["x"] == pytest.approx((-2900.0, 2900.0))

    def test_y_bounds(self) -> None:
        """Y-axis (width): symmetric around origin, half of 2300mm."""
        bounds = get_outer_prism_bounds_mm()
        assert bounds["y"] == pytest.approx((-1150.0, 1150.0))

    def test_z_sampling_bounds(self) -> None:
        """Z-axis for sampling: includes below-ground buffer of ground_clearance=100mm."""
        bounds = get_outer_prism_bounds_mm()
        assert bounds["z_sampling"] == pytest.approx((-100.0, 1650.0))

    def test_z_mesh_bounds(self) -> None:
        """Z-axis for mesh extraction: from ground (0) to top (1650mm)."""
        bounds = get_outer_prism_bounds_mm()
        assert bounds["z_mesh"] == pytest.approx((0.0, 1650.0))

    def test_all_values_are_floats(self) -> None:
        bounds = get_outer_prism_bounds_mm()
        for key, (lo, hi) in bounds.items():
            assert isinstance(lo, float), f"{key} lo is not float"
            assert isinstance(hi, float), f"{key} hi is not float"

    def test_lo_lt_hi_for_all_axes(self) -> None:
        bounds = get_outer_prism_bounds_mm()
        for key, (lo, hi) in bounds.items():
            assert lo < hi, f"{key}: lo ({lo}) must be < hi ({hi})"
