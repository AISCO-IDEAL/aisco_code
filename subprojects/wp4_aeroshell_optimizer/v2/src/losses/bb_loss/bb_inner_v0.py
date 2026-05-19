import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import BbV44


class BbInnerV0(BbV44):
    """Inner prism constraint (GINN-style L2-norm reduction).

    Penalises violation of inner-prism containment: model SDF must be ≤ -inner_margin
    at all points sampled on the inner prism's zero-level surface.
    Returns sqrt(Σ vᵢ²) — L2 norm of inner violations only.
    Independent ALM slot from outer prism, so each grows its own λ/μ.
    """

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            inner_points = self._inner_surface_points.to(self.device)
            inner_out = model(inner_points)
            sdf_inner = (inner_out[0] if isinstance(inner_out, tuple) else inner_out).squeeze(-1)

            upper_sq = self._violation(sdf_inner + self._inner_margin).pow(2)

            loss = torch.sqrt(upper_sq.sum() + 1e-12)
            return torch.clamp(loss - self._inner_tolerance, min=0.0)
