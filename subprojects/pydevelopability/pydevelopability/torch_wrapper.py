"""
Differentiable PyTorch wrapper around the C++ developability energy.

Usage:
    from pydevelopability.torch_wrapper import DevelopabilityEnergy

    energy_fn = DevelopabilityEnergy(energy_type=0)  # HINGE
    loss = energy_fn(vertices, faces)                 # scalar loss
    loss.backward()                                   # grad flows to vertices
"""

import numpy as np
import torch
from pydevelopability._devcore import (
    build_problem,
)
from pydevelopability._devcore import (
    evaluate_energy_and_grad as _cpp_eval,
)


class _DevelopabilityFunction(torch.autograd.Function):
    """Custom autograd function wrapping C++ energy + analytical gradient."""

    @staticmethod
    def forward(ctx, V_tensor, F_tensor, energy_type):
        # Detach and move to CPU/numpy for C++ call
        V_np = V_tensor.detach().cpu().to(torch.float64).numpy()
        F_np = F_tensor.detach().cpu().numpy().astype(np.int32)

        # Build topology (cheap — dominated by TT adjacency)
        problem = build_problem(V_np, F_np)

        # Evaluate energy + analytical gradient
        energy_vec, grad_np = _cpp_eval(problem, V_np, energy_type)

        # Save for backward
        grad_tensor = torch.from_numpy(grad_np.copy()).to(dtype=V_tensor.dtype, device=V_tensor.device)
        ctx.save_for_backward(grad_tensor)

        # Total energy as scalar
        total = float(energy_vec.sum())
        return V_tensor.new_tensor(total)

    @staticmethod
    def backward(ctx, grad_output):
        (grad_tensor,) = ctx.saved_tensors
        # Chain rule: dL/dV = dL/dE * dE/dV
        return grad_output * grad_tensor, None, None


class DevelopabilityEnergy(torch.nn.Module):
    """
    Differentiable developability energy as a PyTorch Module.

    Args:
        energy_type: int in [0..5], see pydevelopability.ENERGY_NAMES
    """

    def __init__(self, energy_type: int = 0):
        super().__init__()
        self.energy_type = energy_type

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vertices: (V, 3) float tensor, requires_grad=True for backward
            faces:    (F, 3) int/long tensor

        Returns:
            scalar energy tensor (sum of per-vertex energies)
        """
        return _DevelopabilityFunction.apply(vertices, faces, self.energy_type)


class CachedDevelopabilityEnergy(torch.nn.Module):
    """
    Same as DevelopabilityEnergy but caches the FixedTopologyProblem when
    topology (F) doesn't change between calls. Saves ~30% on repeated evals.

    Call .invalidate_cache() if faces change.
    """

    def __init__(self, energy_type: int = 0):
        super().__init__()
        self.energy_type = energy_type
        self._problem = None
        self._F_hash = None

    def invalidate_cache(self):
        self._problem = None
        self._F_hash = None

    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return _CachedDevFunction.apply(vertices, faces, self.energy_type, self)


class _CachedDevFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, V_tensor, F_tensor, energy_type, module):
        V_np = V_tensor.detach().cpu().to(torch.float64).numpy()
        F_np = F_tensor.detach().cpu().numpy().astype(np.int32)

        # Cache topology
        f_hash = hash(F_np.tobytes())
        if module._problem is None or module._F_hash != f_hash:
            module._problem = build_problem(V_np, F_np)
            module._F_hash = f_hash

        energy_vec, grad_np = _cpp_eval(module._problem, V_np, energy_type)

        grad_tensor = torch.from_numpy(grad_np.copy()).to(dtype=V_tensor.dtype, device=V_tensor.device)
        ctx.save_for_backward(grad_tensor)

        total = float(energy_vec.sum())
        return V_tensor.new_tensor(total)

    @staticmethod
    def backward(ctx, grad_output):
        (grad_tensor,) = ctx.saved_tensors
        return grad_output * grad_tensor, None, None, None
