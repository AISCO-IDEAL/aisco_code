import contextlib
from typing import Callable, Generator

import torch
import torch.nn as nn


class AdaNPProjector(nn.Module):
    """
    Simplified adaptive neural projection inspired by the ENFORCE paper (arXiv:2502.06774).

    Iteratively projects SDF outputs toward constraint satisfaction via gradient descent
    on the constraint violation w.r.t. the SDF values. Gradient checkpointing is intentionally
    not used here: the projection steps rely on create_graph=True to keep the computation
    graph intact for backprop through the projection, which is incompatible with torch's
    checkpoint mechanism.
    """

    def __init__(self, max_iters: int = 5, step_size: float = 0.1, tolerance: float = 1e-3):
        super().__init__()
        self.max_iters = max_iters
        self.step_size = step_size
        self.tolerance = tolerance
        self.is_active = False
        self.constraints: dict[str, Callable] = {}

    def register_constraint(self, context: str, fn: Callable) -> None:
        self.constraints[context] = fn

    def _projection_step(self, constraint_fn: Callable, coords: torch.Tensor, sdf_proj: torch.Tensor) -> torch.Tensor:
        violation = constraint_fn(coords, sdf_proj)
        if violation.requires_grad:
            grad = torch.autograd.grad(violation.sum(), sdf_proj, create_graph=True)[0]
            sdf_proj = sdf_proj - self.step_size * grad
        return sdf_proj

    def forward(self, coords: torch.Tensor, sdf: torch.Tensor, constraint_context: str | None) -> torch.Tensor:
        if not self.is_active or constraint_context is None or constraint_context not in self.constraints:
            return sdf

        constraint_fn = self.constraints[constraint_context]
        sdf_proj = sdf

        for _ in range(self.max_iters):
            sdf_proj = self._projection_step(constraint_fn, coords, sdf_proj)

            with torch.no_grad():
                violation = constraint_fn(coords, sdf_proj)
                if violation.abs().max() < self.tolerance:
                    break

        return sdf_proj


@contextlib.contextmanager
def enforce_layer_context(model: nn.Module, context_name: str | None) -> Generator[None, None, None]:
    """
    Temporarily sets model.current_enforce_context so the forward pass routes
    the SDF output through the named constraint's projection step.
    """
    prev_context = getattr(model, "current_enforce_context", None)
    try:
        model.current_enforce_context = context_name
        yield
    finally:
        model.current_enforce_context = prev_context
