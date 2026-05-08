# eikonal v0
import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.losses_utils import get_points


class EikonalV0(ParentLoss):
    def _setup(self) -> None:
        # Fraction of eikonal points that are biased toward the current zero-crossing.
        # 0.0 (default) = legacy behaviour: fixed uniform prism points allocated once.
        # 0.0 < f <= 1.0 = fresh adaptive sampling: (1-f) uniform + f near-surface each call.
        # Near-surface points are found by a cheap no-grad coarse grid forward pass that
        # identifies the n_coarse points with smallest |SDF|, preventing exterior-only bias.
        self._near_surface_fraction: float = float(
            self.losses_config.get("domain_pts_sampling", {}).get("eikonal_near_surface_fraction", 0.0)
        )
        # Coarse grid used to locate near-surface candidates (4x the eikonal point count).
        self._near_surface_coarse_mult: int = 4

    def _sample_adaptive_points(self, model: object) -> torch.Tensor:
        """Sample eikonal points mixing uniform prism and near-surface candidates.

        A no-grad coarse forward pass on a 4× larger uniform pool selects the
        `near_surface_fraction` fraction closest to the zero-crossing, then fills
        the remaining budget with fresh uniform points.  The result has
        `requires_grad=True` as required for autograd gradient computation.

        Args:
            model: Current SDF model with _active_z already set.

        Returns:
            Tensor of shape (num_domain_points, 3) with requires_grad=True.
        """
        n_total = self._num_domain_points
        n_near = int(n_total * self._near_surface_fraction)
        n_uniform = n_total - n_near

        lo = self._bbox_lo
        hi = self._bbox_hi

        uniform_pts = get_points(n_uniform, lo, hi, self.device) if n_uniform > 0 else None

        if n_near > 0:
            n_coarse = n_near * self._near_surface_coarse_mult
            lo_t = torch.tensor(lo, dtype=torch.float32, device=self.device)
            hi_t = torch.tensor(hi, dtype=torch.float32, device=self.device)
            coarse = lo_t + torch.rand(n_coarse, 3, device=self.device) * (hi_t - lo_t)
            with torch.no_grad():
                sdf_coarse = model(coarse).squeeze()  # type: ignore[operator]
            # Select points closest to zero-crossing (smallest |SDF|)
            _, top_idx = torch.topk(sdf_coarse.abs(), k=n_near, largest=False)
            near_pts = coarse[top_idx].detach().requires_grad_(True)
        else:
            near_pts = None

        parts = [p for p in (uniform_pts, near_pts) if p is not None]
        return torch.cat(parts, dim=0)

    def _loss(self, model) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network
                sdf: the sdf function values for each point
                grad_sdf: the gradients of the sdf function at each point
                    both are torch.Tensors
        Returns:
            Eikonal loss (to be minimized)
                loss: Mean absolute difference between the norm of the gradients and 1
                loss = 1/n * sum from 1 to n of  ( ||grad_sdf||_2 - 1 )^2
                    source: https://arxiv.org/pdf/2002.10099, Implicit Geometric Regularization for Learning Shapes, Page 2, 2. Method l(theta)
        """
        # Use adaptive near-surface sampling when enabled, otherwise fall back to
        # the legacy fixed points allocated once at __init__ (no overhead).
        points = self._sample_adaptive_points(model) if self._near_surface_fraction > 0.0 else self.points

        with torch.enable_grad():
            output = model(points)
            if isinstance(output, tuple):
                sdf, grad_sdf = output
            else:
                sdf = output
                # Compute gradient if not returned
                grad_sdf = torch.autograd.grad(
                    outputs=sdf,
                    inputs=points,
                )[0]

            diff = torch.norm(grad_sdf, p=2, dim=-1) - 1.0
            eikonal_loss = torch.pow(diff, 2).mean()  # in v0 there was no ^2 here, but it should be according to the paper
            return eikonal_loss
