"""Tests for signed relative integrated drag error in ValidationCalculator.

TDD tests for Plan 120: _integrated_drag_errors should return
(abs_errors, rel_errors) from a single integration pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.trainer.validation_and_logging.validation_calculator import (
    _DRAG_ZERO_EPS,
    ValidationCalculator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_calc() -> ValidationCalculator:
    return ValidationCalculator(config={})


def _patched_integrate(calc: ValidationCalculator, return_values: list[float | None]):
    """Patch _integrate_sample to return values from the list in order."""
    side_effects = iter(return_values)
    return patch.object(calc, "_integrate_sample", side_effect=lambda *_args, **_kwargs: next(side_effects))


def _make_batch_and_trainer():
    """Return minimal mocks that satisfy the early-exit guards in _integrated_drag_errors."""
    batch = MagicMock()
    batch.x = MagicMock()  # not None
    batch.c = MagicMock()  # not None

    trainer = MagicMock()
    trainer.device = "cpu"
    trainer.dataset_config = {"v_inf": 25.0, "rho": 1.225}

    pred_normalized = {"field": MagicMock()}
    return batch, trainer, pred_normalized


# ---------------------------------------------------------------------------
# Tests: return type
# ---------------------------------------------------------------------------


def test_integrated_drag_errors_returns_tuple_of_two_lists():
    """_integrated_drag_errors must return (abs_errors, rel_errors)."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [10.0, 20.0]),
    ):
        result = calc._integrated_drag_errors(trainer, batch, pred)

    assert isinstance(result, tuple), "must return a tuple"
    assert len(result) == 2, "tuple must have exactly two elements"
    abs_errors, rel_errors = result
    assert isinstance(abs_errors, list)
    assert isinstance(rel_errors, list)


# ---------------------------------------------------------------------------
# Tests: absolute error (must be unchanged)
# ---------------------------------------------------------------------------


def test_absolute_error_under_prediction():
    """pred=10, target=20 → abs_error = 10."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [10.0, 20.0]),
    ):
        abs_errors, _ = calc._integrated_drag_errors(trainer, batch, pred)

    assert abs_errors == [10.0]


def test_absolute_error_over_prediction():
    """pred=40, target=20 → abs_error = 20."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [40.0, 20.0]),
    ):
        abs_errors, _ = calc._integrated_drag_errors(trainer, batch, pred)

    assert abs_errors == [20.0]


# ---------------------------------------------------------------------------
# Tests: relative error (new behaviour)
# ---------------------------------------------------------------------------


def test_relative_error_under_prediction():
    """pred=10, target=20 → rel_error = 0.5."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [10.0, 20.0]),
    ):
        _, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert pytest.approx(rel_errors[0], rel=1e-6) == 0.5


def test_relative_error_over_prediction():
    """pred=40, target=20 → rel_error = +1.0."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [40.0, 20.0]),
    ):
        _, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert pytest.approx(rel_errors[0], rel=1e-6) == 1.0


def test_relative_error_exact_prediction():
    """pred=20, target=20 → rel_error = 0.0."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [20.0, 20.0]),
    ):
        _, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert pytest.approx(rel_errors[0], abs=1e-9) == 0.0


# ---------------------------------------------------------------------------
# Tests: zero/near-zero guard
# ---------------------------------------------------------------------------


def test_near_zero_target_excluded_from_rel_errors():
    """target near zero must not appear in rel_errors (division guard)."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()
    near_zero = _DRAG_ZERO_EPS * 0.5  # below threshold

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [1.0, near_zero]),
    ):
        abs_errors, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    # abs_error still recorded
    assert len(abs_errors) == 1
    # rel_error skipped
    assert rel_errors == []


def test_zero_target_excluded_from_rel_errors():
    """target exactly zero must not appear in rel_errors."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [5.0, 0.0]),
    ):
        abs_errors, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert len(abs_errors) == 1
    assert rel_errors == []


def test_target_above_eps_included_in_rel_errors():
    """target just above _DRAG_ZERO_EPS must appear in rel_errors."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()
    above_eps = _DRAG_ZERO_EPS * 2.0

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [above_eps * 0.5, above_eps]),
    ):
        _, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert len(rel_errors) == 1


# ---------------------------------------------------------------------------
# Tests: None return from _integrate_sample
# ---------------------------------------------------------------------------


def test_none_pred_excluded_from_both_lists():
    """If _integrate_sample returns None for pred, both lists get nothing."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [None, 20.0]),
    ):
        abs_errors, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert abs_errors == []
    assert rel_errors == []


def test_none_target_excluded_from_both_lists():
    """If _integrate_sample returns None for target, both lists get nothing."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10)]),
        _patched_integrate(calc, [10.0, None]),
    ):
        abs_errors, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert abs_errors == []
    assert rel_errors == []


# ---------------------------------------------------------------------------
# Tests: multiple samples in one batch
# ---------------------------------------------------------------------------


def test_multiple_samples_both_lists_populated():
    """Multiple sample ranges: both lists contain one entry per valid sample."""
    calc = _make_calc()
    batch, trainer, pred = _make_batch_and_trainer()

    # 3 samples: (pred, target) = (10, 20), (40, 20), (5, near_zero)
    near_zero = _DRAG_ZERO_EPS * 0.1
    with (
        patch.object(calc, "_denormalize_field", return_value=MagicMock()),
        patch.object(calc, "_sample_ranges", return_value=[(0, 10), (10, 20), (20, 30)]),
        _patched_integrate(calc, [10.0, 20.0, 40.0, 20.0, 5.0, near_zero]),
    ):
        abs_errors, rel_errors = calc._integrated_drag_errors(trainer, batch, pred)

    assert len(abs_errors) == 3  # all three samples contribute to abs
    assert len(rel_errors) == 2  # third sample skipped (near-zero target)
    assert pytest.approx(rel_errors[0], rel=1e-6) == 0.5
    assert pytest.approx(rel_errors[1], rel=1e-6) == 1.0
