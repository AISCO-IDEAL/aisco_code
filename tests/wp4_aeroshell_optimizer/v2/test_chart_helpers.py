"""Tests for loss-competition chart helper robustness in DiversityTrainer.

Covers the `_safe_pie()` nested helper inside `_log_competition_charts()`,
verifying that non-finite and non-positive values are filtered rather than
passed to matplotlib, which would produce broken charts.

Plan reference: 020 — Fix total_loss contamination in loss-competition charts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import matplotlib.figure
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bare_diversity_trainer() -> Any:
    """Return a DiversityTrainer instance with __init__ bypassed.

    Sets only the attributes needed by _log_competition_charts:
    _use_wandb, and the four history lists.
    """
    from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import DiversityTrainer  # noqa: PLC0415

    dt = object.__new__(DiversityTrainer)
    dt._use_wandb = True
    dt._raw_loss_history = []
    dt._normalized_loss_history = []
    dt._weighted_loss_history = []
    dt._optimizer_loss_history = []
    return dt


def _call_log_competition_charts(dt: Any, raw_floats: dict[str, float]) -> dict:
    """Call _log_competition_charts with controlled raw_floats and empty other dicts.

    Mocks wandb.run and wandb.Image so that the method runs to completion and
    returns figures directly (not wrapped in wandb.Image) for easy inspection.
    """
    with patch("wandb.run", new=MagicMock()), patch("wandb.Image", side_effect=lambda fig: fig):
        return dt._log_competition_charts(
            raw_floats=raw_floats,
            normalized_floats={},
            weighted_floats={},
            optimizer_floats={},
            epoch=0,
        )


def _figure_has_no_data_text(fig: matplotlib.figure.Figure) -> bool:
    """Return True if any Axes in the figure contains a 'No data' text artist."""
    for ax in fig.axes:
        for text in ax.texts:
            if "No data" in text.get_text():
                return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSafePieDiversityTrainer:
    """_safe_pie() inside DiversityTrainer._log_competition_charts must filter
    non-finite and non-positive values and render 'No data' instead of passing
    them to ax.pie(), which would raise or produce a broken chart.
    """

    def test_all_nan_values_render_no_data(self) -> None:
        """When all raw_floats values are NaN, _safe_pie must show 'No data'."""
        dt = _bare_diversity_trainer()
        result = _call_log_competition_charts(dt, {"loss_a": float("nan"), "loss_b": float("nan")})

        raw_pie = result.get("loss_competition/raw_pie")
        assert isinstance(raw_pie, matplotlib.figure.Figure), "Expected a Figure, not a wandb.Image wrapper"
        assert _figure_has_no_data_text(raw_pie), "Expected 'No data' text when all values are NaN"

    def test_all_inf_values_render_no_data(self) -> None:
        """When all raw_floats values are Inf, _safe_pie must show 'No data'."""
        dt = _bare_diversity_trainer()
        result = _call_log_competition_charts(dt, {"loss_a": float("inf"), "loss_b": float("inf")})

        raw_pie = result["loss_competition/raw_pie"]
        assert _figure_has_no_data_text(raw_pie), "Expected 'No data' text when all values are Inf"

    def test_all_zero_values_render_no_data(self) -> None:
        """When all raw_floats values are zero, _safe_pie must show 'No data'."""
        dt = _bare_diversity_trainer()
        result = _call_log_competition_charts(dt, {"loss_a": 0.0, "loss_b": 0.0})

        raw_pie = result["loss_competition/raw_pie"]
        assert _figure_has_no_data_text(raw_pie), "Expected 'No data' text when all values are zero"

    def test_all_negative_values_render_no_data(self) -> None:
        """When all raw_floats values are negative, _safe_pie must show 'No data'."""
        dt = _bare_diversity_trainer()
        result = _call_log_competition_charts(dt, {"loss_a": -3.0, "loss_b": -1.0})

        raw_pie = result["loss_competition/raw_pie"]
        assert _figure_has_no_data_text(raw_pie), "Expected 'No data' text when all values are negative"

    def test_mixed_invalid_and_valid_values_renders_pie(self) -> None:
        """When at least one raw_floats value is valid (finite positive), a pie is rendered."""
        dt = _bare_diversity_trainer()
        result = _call_log_competition_charts(
            dt,
            {"loss_nan": float("nan"), "loss_valid": 3.0},
        )

        raw_pie = result["loss_competition/raw_pie"]
        assert isinstance(raw_pie, matplotlib.figure.Figure)
        assert not _figure_has_no_data_text(raw_pie), "Expected a real pie chart when one valid value is present"
        # Verify ax.pie was actually called: the figure should have wedges
        ax = raw_pie.axes[0]
        assert len(ax.patches) > 0, "Expected pie wedges (patches) in the figure"

    def test_does_not_raise_on_nan_inf_mix(self) -> None:
        """_log_competition_charts must not raise when given NaN and Inf values."""
        dt = _bare_diversity_trainer()
        # Should complete without ValueError / RuntimeError from matplotlib
        _call_log_competition_charts(
            dt,
            {"loss_a": float("nan"), "loss_b": float("inf"), "loss_c": -2.0, "loss_d": 0.0},
        )

    @pytest.mark.parametrize(
        "bad_values",
        [
            {"only_nan": float("nan")},
            {"only_inf": float("inf")},
            {"only_neg": -99.0},
            {"only_zero": 0.0},
            {"nan_and_inf": float("nan"), "inf_val": float("inf")},
        ],
    )
    def test_parametrized_bad_inputs_all_show_no_data(self, bad_values: dict[str, float]) -> None:
        """Parametrized guard: any all-invalid input dict shows 'No data'."""
        dt = _bare_diversity_trainer()
        result = _call_log_competition_charts(dt, bad_values)
        raw_pie = result["loss_competition/raw_pie"]
        assert _figure_has_no_data_text(raw_pie), f"Expected 'No data' for input {bad_values}"
