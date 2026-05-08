import contextlib
from typing import Callable, Generator  # noqa: UP035

import torch
from torch import nn


class AdaNPProjector(nn.Module):
    """
    Adaptive neural projection following the ENFORCE paper (arXiv:2502.06774).

    Applies a Newton step on the constraint Jacobian dc/dy to project the network's
    SDF output onto the constraint boundary. For linear violation functions (relu),
    a single Newton step achieves exact constraint satisfaction regardless of violation
    magnitude. For nonlinear violations (softplus), max_iters iterations converge.

    The Jacobian is computed on a detached copy of the SDF output so the computation
    graph from the network is not extended by the projection. After projection the
    constraint violation is 0, so the constraint loss and its gradient are both ~0 —
    equivalent to the IFT backward from the paper. The network continues to receive
    gradients from CSE and other non-projected losses.
    """

    def __init__(self, max_iters: int = 5, tolerance: float = 1e-3):
        super().__init__()
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.is_active = False
        self.constraints: dict[str, Callable] = {}

    def register_constraint(self, context: str, fn: Callable) -> None:
        self.constraints[context] = fn

    def _projection_step(self, constraint_fn: Callable, coords: torch.Tensor, sdf_proj: torch.Tensor) -> torch.Tensor:
        sdf_work = sdf_proj.detach().requires_grad_(True)
        violation = constraint_fn(coords, sdf_work)  # [N]

        if violation.sum() < 1e-12:
            return sdf_proj

        dc_dy = torch.autograd.grad(violation.sum(), sdf_work)[0]  # [N, 1]

        active = violation > 0
        correction = torch.where(
            active.unsqueeze(-1),
            violation.unsqueeze(-1) / dc_dy.abs().clamp(min=1e-8) * dc_dy.sign(),
            torch.zeros_like(sdf_proj),
        )
        return sdf_proj - correction.detach()

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
        # nn.Module.__setattr__ is typed only for Tensor/Module values; for plain Python
        # attributes it falls through to object.__setattr__ at runtime. Calling it directly
        # makes that explicit and keeps mypy happy.
        object.__setattr__(model, "current_enforce_context", context_name)
        yield
    finally:
        object.__setattr__(model, "current_enforce_context", prev_context)
