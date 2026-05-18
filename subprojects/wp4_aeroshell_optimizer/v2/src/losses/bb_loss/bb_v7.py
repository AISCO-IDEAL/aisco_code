import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import BbV44


class BbV7(BbV44):
    """BB loss with GINN-style L2-norm reduction.

    Per-point violation vᵢ = softplus(sdf + margin) (from _violation()).
    Returns sqrt(Σvᵢ²) — the L2 norm of the violation vector across all
    inner + outer surface-sampled points. Non-violating points contribute
    ~0 via softplus. ALM receives this scalar, giving gradient ∝ vᵢ/‖v‖
    per point — hot points dominate, magnitude bounded by λ + μ‖v‖.
    """

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            inner_points = self._inner_surface_points.to(self.device)
            outer_points = self._outer_surface_points.to(self.device)

            inner_out = model(inner_points)
            outer_out = model(outer_points)
            sdf_inner = (inner_out[0] if isinstance(inner_out, tuple) else inner_out).squeeze(-1)
            sdf_outer = (outer_out[0] if isinstance(outer_out, tuple) else outer_out).squeeze(-1)

            upper_sq = self._violation(sdf_inner + self._inner_margin).pow(2)
            lower_sq = self._violation(self._outer_margin - sdf_outer).pow(2)

            return torch.sqrt(upper_sq.sum() + lower_sq.sum() + 1e-12)
