"""Regression test: OpenFOAM kinematic-pressure → Cp normalization.

simpleFoam stores p_kin = p_static / rho  [m²/s²].  The correct Cp formula is:

    Cp = p_static / (0.5 * rho * v_inf²)
       = p_kin    / (0.5 * v_inf²)          (rho cancels)

Prior to the fix, `compute_pressure_coefficient` was called with `rho=AIR_DENSITY`,
yielding Cp_stored = p_kin / (0.5 * rho * v_inf^2) = Cp_true / rho -- off by 1.225x.

These tests assert that:
1. `compute_pressure_coefficient(p_kin, v_inf, rho=1.0)` gives the correct Cp.
2. The stagnation-point Cp (p_kin = 0.5 * v_inf²) equals exactly 1.0.
3. The old call `compute_pressure_coefficient(p_kin, v_inf, rho=1.225)` gives 1/1.225
   (documents the bug for reference).
"""

import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.dataset.common_preprocessing import (
    AIR_DENSITY,
    V_INF_EULER,
    compute_pressure_coefficient,
)

RHO = AIR_DENSITY  # 1.225 kg/m³
V = V_INF_EULER  # 25.0 m/s
Q_DYN = 0.5 * V**2  # kinematic dynamic pressure [m²/s²] = 312.5


def test_stagnation_cp_correct_with_rho_1() -> None:
    """Kinematic stagnation pressure normalised with rho=1 must yield Cp = 1.0."""
    p_kin_stagnation = torch.tensor([[Q_DYN]], dtype=torch.float32)  # 0.5 * v²
    cp = compute_pressure_coefficient(p_kin_stagnation, V, rho=1.0)
    assert abs(cp.item() - 1.0) < 1e-5, f"Expected Cp=1.0, got {cp.item()}"


def test_stagnation_cp_wrong_with_rho_1225() -> None:
    """Documents the pre-fix bug: using rho=1.225 deflates Cp by that factor."""
    p_kin_stagnation = torch.tensor([[Q_DYN]], dtype=torch.float32)
    cp_bugged = compute_pressure_coefficient(p_kin_stagnation, V, rho=RHO)
    expected_bugged = 1.0 / RHO  # ≈ 0.816
    assert abs(cp_bugged.item() - expected_bugged) < 1e-5, f"Expected bugged Cp={expected_bugged:.4f}, got {cp_bugged.item()}"


def test_arbitrary_kinematic_pressure_roundtrip() -> None:
    """p_kin = p_static / rho → Cp(p_kin, rho=1) == Cp(p_static, rho=RHO)."""
    p_static = torch.tensor([[200.0, -50.0, 0.0]], dtype=torch.float32)  # [Pa]
    p_kin = p_static / RHO  # as stored by simpleFoam

    cp_from_static = compute_pressure_coefficient(p_static, V, rho=RHO)
    cp_from_kin = compute_pressure_coefficient(p_kin, V, rho=1.0)

    torch.testing.assert_close(cp_from_kin, cp_from_static, atol=1e-5, rtol=0)
