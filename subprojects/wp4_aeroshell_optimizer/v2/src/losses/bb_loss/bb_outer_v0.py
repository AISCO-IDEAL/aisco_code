import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import BbV44


class BbOuterV0(BbV44):
    """Outer prism constraint (GINN-style L2-norm reduction).

    Penalises violation of outer-prism fit: model SDF must be ≥ outer_margin
    at all points sampled on the outer prism's zero-level surface.
    Returns sqrt(Σ vᵢ²) — L2 norm of outer violations only.
    Independent ALM slot from inner prism, so each grows its own λ/μ.
    """

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            outer_points = self._outer_surface_points.to(self.device)
            outer_out = model(outer_points)
            sdf_outer = (outer_out[0] if isinstance(outer_out, tuple) else outer_out).squeeze(-1)

            lower_sq = self._violation(self._outer_margin - sdf_outer).pow(2)

            loss = torch.sqrt(lower_sq.sum() + 1e-12)
            return torch.clamp(loss - self._outer_tolerance, min=0.0)
