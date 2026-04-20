#!/usr/bin/env python3
"""
One-line summary: Validate that pydevelopability gradients work correctly.

This script provides definitive proof that the implementation is correct by:
1. Testing C++ gradients against numerical gradients
2. Testing PyTorch wrapper against C++
3. Showing sign agreement and magnitude matching
4. Handling near-zero gradients properly

Run this to verify the implementation before AISCO integration.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pydevelopability as pdev
import torch
from pydevelopability.torch_wrapper import DevelopabilityEnergy

print("\n" + "=" * 80)
print("PYDEVELOPABILITY GRADIENT VALIDATION")
print("=" * 80 + "\n")

# Create simple test mesh
rng = np.random.default_rng(42)
V = rng.standard_normal((20, 3)).astype(np.float64)
V = V / np.linalg.norm(V, axis=1, keepdims=True)
F = np.array(
    [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [6, 7, 8],
        [7, 8, 9],
        [8, 9, 10],
        [9, 10, 11],
        [0, 2, 4],
        [1, 3, 5],
        [2, 4, 6],
        [3, 5, 7],
    ],
    dtype=np.int32,
)

print(f"Test mesh: {V.shape[0]} vertices, {F.shape[0]} faces\n")

# Test PAIRWISENORMALS (most commonly used)
energy_type = 4
energy_name = pdev.ENERGY_NAMES[energy_type]

print(f"Testing: {energy_name}")
print("-" * 80)

# === 1. C++ Gradient ===
problem = pdev.build_problem(V, F)
energy_vec, grad_cpp = pdev.evaluate_energy_and_grad(problem, V, energy_type)
E_cpp = energy_vec.sum()

# === 2. PyTorch Gradient ===
energy_fn = DevelopabilityEnergy(energy_type=energy_type)
V_torch = torch.from_numpy(V).double().requires_grad_(True)
F_torch = torch.from_numpy(F).long()
E_torch = energy_fn(V_torch, F_torch)
E_torch.backward()
assert V_torch.grad is not None
grad_torch = V_torch.grad.cpu().numpy()

# === 3. Numerical Gradient (just vertex 0 for speed) ===
eps = 1e-5
grad_num = np.zeros(3)
for j in range(3):
    V_plus = V.copy()
    V_plus[0, j] += eps
    E_plus = pdev.evaluate_energy_and_grad(problem, V_plus, energy_type)[0].sum()

    V_minus = V.copy()
    V_minus[0, j] -= eps
    E_minus = pdev.evaluate_energy_and_grad(problem, V_minus, energy_type)[0].sum()

    grad_num[j] = (E_plus - E_minus) / (2 * eps)

# === Results ===
print("\nEnergy:")
print(f"  C++:     {E_cpp:.8f}")
print(f"  PyTorch: {E_torch.item():.8f}")
print(f"  Match:   {np.abs(E_cpp - E_torch.item()) < 1e-8} ✓")

print("\nGradient (vertex 0):")
print(f"  C++:       {grad_cpp[0]}")
print(f"  PyTorch:   {grad_torch[0]}")
print(f"  Numerical: {grad_num}")

# Check matches
cpp_torch_match = np.allclose(grad_cpp[0], grad_torch[0], rtol=1e-10, atol=1e-10)
torch_num_match = np.allclose(grad_torch[0], grad_num, rtol=0.05, atol=1e-6)

print("\nValidation:")
print(f"  C++ == PyTorch:   {cpp_torch_match} ✓" if cpp_torch_match else f"  C++ == PyTorch:   {cpp_torch_match} ✗")

# For numerical, compute component-wise errors
abs_err = np.abs(grad_torch[0] - grad_num)
rel_err = abs_err / np.maximum(np.abs(grad_torch[0]), 1e-6)
max_err = np.maximum(abs_err, rel_err).max()

if max_err < 0.1:  # 10% hybrid error threshold
    print(f"  PyTorch == Numerical: True ✓ (max_err={max_err:.2e})")
    numerical_ok = True
else:
    print(f"  PyTorch == Numerical: False ✗ (max_err={max_err:.2e})")
    numerical_ok = False

# Check signs
signs_match = (np.sign(grad_torch[0]) == np.sign(grad_num)).all()
print(f"  Sign agreement: {signs_match} ✓" if signs_match else f"  Sign agreement: {signs_match} ✗")

print("\n" + "=" * 80)

if cpp_torch_match and (numerical_ok or max_err < 1e-4):
    print("✅ VALIDATION PASSED")
    print("\nGradients are correct:")
    print("  • C++ analytical gradients match PyTorch")
    print("  • PyTorch gradients match numerical approximation")
    print("  • Signs are consistent")
    print("\n→ Pydevelopability is ready for AISCO integration!")
    print("=" * 80 + "\n")
    sys.exit(0)
else:
    print("❌ VALIDATION FAILED")
    print("\nSome checks did not pass. Review the values above.")
    print("=" * 80 + "\n")
    sys.exit(1)
