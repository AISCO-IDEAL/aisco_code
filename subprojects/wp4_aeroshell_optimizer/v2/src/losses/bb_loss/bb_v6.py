import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import BbV44


class BbV6(BbV44):
    """Pure quadratic-violation bb loss — captures the gradient profile of
    ENFORCE phase 2 (where per-point gradient scales with violation magnitude)
    without the projector machinery, combined with topk reduction.

    For each sampled point, computes the softplus violation v, then squares it.
    The squaring gives gradient proportional to v (not bounded like softplus alone),
    so stuck points get a stronger pull than barely-violating ones.

    Reduces via topk: only the worst topk_fraction of points contribute. This
    prevents the .mean() from diluting hot-points with a sea of near-satisfied ones.
    """

    def _setup(self) -> None:
        super()._setup()
        analytic_cfg = self.losses_config["bb_config"].get("analytic_options", {})
        self.__topk_fraction = float(analytic_cfg.get("topk_fraction", 0.1))
        self.__topk_fraction = min(1.0, max(0.01, self.__topk_fraction))
        print(f"[BbV6] topk_fraction={self.__topk_fraction:.2%}")

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

            def topk_mean(t: torch.Tensor) -> torch.Tensor:
                if t.numel() == 0:
                    return torch.tensor(0.0, device=t.device, dtype=t.dtype)
                k = max(1, int(t.numel() * self.__topk_fraction))
                return torch.topk(t, k, largest=True)[0].mean()

            return 0.5 * (topk_mean(upper_sq) + topk_mean(lower_sq))
