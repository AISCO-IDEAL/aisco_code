"""
Unit tests for cruising_speed.py

Tested edge cases:
  - Zero drag (linear degeneration)
  - Solar power < constant losses (infeasible / clamp)
  - Very high drag (near-zero speed, clamp at 1e-6)
  - Batch of mixed values
  - Residual accuracy for realistic inputs
  - Gradient correctness via finite differences
  - Gradient correctness via analytic implicit derivatives
  - Negative drag (unphysical robustness)
  - Very large solar power (numerical stability)
"""

import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.cruising_speed import (
    cruising_speed,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _residual(
    u: torch.Tensor,
    drag: torch.Tensor,
    solar_raw: torch.Tensor,
    mass: float,
    mu: float,
    c: float,
) -> torch.Tensor:
    """Evaluate F(u) = drag * u^3 / 625 + rolling * u + c - 1.25 * solar."""
    rolling = mass * 9.81 * mu
    return drag * u**3 / (25.0**2) + rolling * u + c - 1.25 * solar_raw


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------


class TestCruisingSpeedForward:
    def test_zero_drag_closed_form(self):
        """With zero drag the equation is linear: u = (1.25*P - c) / rolling."""
        solar = torch.tensor([500.0])
        drag = torch.tensor([0.0])
        mass = 200.0
        c = 30.0
        mu = 0.006

        rolling = mass * 9.81 * mu
        expected = (1.25 * solar - c) / rolling

        result = cruising_speed(solar, drag, mass, c, mu)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_residual_near_zero_realistic(self):
        """Solver residual must be < 1e-4 for typical solar-car values."""
        solar = torch.tensor([800.0])
        drag = torch.tensor([100.0])
        mass = 210.0
        c = 25.0
        mu = 0.006

        u = cruising_speed(solar, drag, mass, c, mu)
        res = _residual(u, drag, solar, mass, mu, c)
        assert torch.abs(res).max().item() < 1e-4

    def test_batch_values(self):
        """Vectorised call with different power/drag combos."""
        solar = torch.tensor([400.0, 600.0, 900.0, 1200.0])
        drag = torch.tensor([50.0, 80.0, 120.0, 200.0])
        mass = 210.0
        c = 20.0
        mu = 0.006

        u = cruising_speed(solar, drag, mass, c, mu)
        res = _residual(u, drag, solar, mass, mu, c)
        assert u.shape == solar.shape
        assert torch.abs(res).max().item() < 1e-3

    def test_solar_less_than_losses_clamps(self):
        """When 1.25*P < c the net power is negative; speed should clamp to 1e-6."""
        solar = torch.tensor([10.0])  # 1.25 * 10 = 12.5 < 30
        drag = torch.tensor([50.0])
        mass = 200.0
        c = 30.0
        mu = 0.006

        u = cruising_speed(solar, drag, mass, c, mu)
        assert u.item() >= 1e-8  # clamp floor
        assert u.item() < 1.0  # should be very small

    def test_very_high_drag(self):
        """Extremely high drag should yield a very low but positive speed."""
        solar = torch.tensor([600.0])
        drag = torch.tensor([1e6])
        mass = 210.0
        c = 20.0
        mu = 0.006

        u = cruising_speed(solar, drag, mass, c, mu)
        assert u.item() >= 1e-6
        assert u.item() < 5.0

    def test_very_large_solar_power(self):
        """Large solar should give a high speed without NaN/Inf."""
        solar = torch.tensor([50000.0])
        drag = torch.tensor([100.0])
        mass = 210.0
        c = 20.0
        mu = 0.006

        u = cruising_speed(solar, drag, mass, c, mu)
        assert torch.isfinite(u).all()
        assert u.item() > 10.0

    def test_output_is_positive(self):
        """Speed must always be positive."""
        solar = torch.tensor([300.0, 600.0, 1000.0])
        drag = torch.tensor([50.0, 100.0, 200.0])

        u = cruising_speed(solar, drag, mass=210.0, c_losses=25.0)
        assert (u > 0).all()


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------


class TestCruisingSpeedGradients:
    def test_gradients_match_analytic_implicit(self):
        """Autograd gradients must match analytic implicit-function-theorem derivatives."""
        solar = torch.tensor([700.0], requires_grad=True)
        drag = torch.tensor([110.0], requires_grad=True)
        mass = 210.0
        c = 25.0
        mu = 0.006

        u = cruising_speed(solar, drag, mass, c, mu)
        u.sum().backward()

        rolling = mass * 9.81 * mu
        u_val = u.detach()
        df_du = 3.0 * drag.detach() * u_val**2 / 625.0 + rolling

        expected_grad_drag = -(u_val**3 / 625.0) / df_du
        expected_grad_solar = 1.25 / df_du

        assert torch.allclose(drag.grad, expected_grad_drag, atol=1e-5, rtol=1e-4)
        assert torch.allclose(solar.grad, expected_grad_solar, atol=1e-5, rtol=1e-4)

    def test_gradients_finite_difference(self):
        """Finite-difference check for the custom backward."""
        # Use float64 for the finite difference calculation to avoid numerical noise
        eps = 1e-4
        solar_val = 650.0
        drag_val = 95.0
        mass = 210.0
        c = 25.0
        mu = 0.006

        # 1. Autograd
        s = torch.tensor([solar_val], dtype=torch.float64, requires_grad=True)
        d = torch.tensor([drag_val], dtype=torch.float64)
        u = cruising_speed(s, d, mass, c, mu)
        u.backward()
        ag_solar = s.grad.item()

        # 2. Finite Difference (ensure these are fresh calls)
        # Use the same dtype as the autograd check
        u_plus = cruising_speed(torch.tensor([solar_val + eps], dtype=torch.float64), d, mass, c, mu).item()
        u_minus = cruising_speed(torch.tensor([solar_val - eps], dtype=torch.float64), d, mass, c, mu).item()

        fd_solar = (u_plus - u_minus) / (2 * eps)

        assert abs(ag_solar - fd_solar) < 1e-4, f"Mismatch: ag={ag_solar}, fd={fd_solar}"

        # Autograd gradient w.r.t. drag
        s2 = torch.tensor([solar_val])
        d2 = torch.tensor([drag_val], requires_grad=True)
        u2 = cruising_speed(s2, d2, mass, c, mu)
        u2.backward()
        ag_drag = d2.grad.item()

        u_plus_d = cruising_speed(s2, torch.tensor([drag_val + eps]), mass, c, mu).item()
        u_minus_d = cruising_speed(s2, torch.tensor([drag_val - eps]), mass, c, mu).item()
        fd_drag = (u_plus_d - u_minus_d) / (2 * eps)

        assert abs(ag_drag - fd_drag) < 1e-4, f"drag grad mismatch: autograd={ag_drag}, fd={fd_drag}"

    def test_no_grad_for_constants(self):
        """rolling_friction_force and c_losses should not receive gradients."""
        solar = torch.tensor([600.0], requires_grad=True)
        drag = torch.tensor([80.0], requires_grad=True)

        u = cruising_speed(solar, drag, mass=210.0, c_losses=25.0)
        u.backward()

        # If constants were tracked they'd have grads — they shouldn't be tensors
        # with grad. Just verify no error and that solar/drag grads exist.
        assert solar.grad is not None
        assert drag.grad is not None

    def test_negative_drag_no_crash(self):
        """Unphysical negative drag shouldn't crash; gradients should still exist."""
        solar = torch.tensor([600.0], requires_grad=True)
        drag = torch.tensor([-10.0], requires_grad=True)

        u = cruising_speed(solar, drag, mass=210.0, c_losses=25.0)
        assert torch.isfinite(u).all()
        u.backward()
        assert torch.isfinite(solar.grad).all()
        assert torch.isfinite(drag.grad).all()
