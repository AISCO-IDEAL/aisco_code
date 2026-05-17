import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import BbV44


class BbV5(BbV44):
    """BB loss with topk mean reduction instead of full mean.

    Addresses the problem where satisfied points dilute the loss signal: with thousands
    of mostly-satisfied bb points and a handful of stuck hot-spots, the full mean gives
    a weak gradient on the worst violators. Uses topk to focus on the worst-violating
    fraction of points.

    Config parameter `topk_fraction` (default 0.1) controls which fraction of points
    (by violation magnitude) are included in the reduced mean. With topk_fraction=0.1,
    only the top 10% worst points contribute to the loss gradient.
    """

    def _setup(self) -> None:
        super()._setup()
        analytic_cfg = self.losses_config["bb_config"].get("analytic_options", {})
        self.__topk_fraction = float(analytic_cfg.get("topk_fraction", 0.1))
        self.__topk_fraction = min(1.0, max(0.01, self.__topk_fraction))
        print(f"[BbV5] Using topk_fraction={self.__topk_fraction:.2%}")

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            inner_points = self._inner_surface_points.to(self.device)
            outer_points = self._outer_surface_points.to(self.device)

            inner_out = model(inner_points)
            outer_out = model(outer_points)

            sdf_inner_pred = inner_out[0] if isinstance(inner_out, tuple) else inner_out
            sdf_outer_pred = outer_out[0] if isinstance(outer_out, tuple) else outer_out

            sdf_inner_pred = sdf_inner_pred.squeeze(-1)
            sdf_outer_pred = sdf_outer_pred.squeeze(-1)

            # Compute violations (elementwise, all points)
            upper_violation = self._violation(sdf_inner_pred + self._inner_margin)
            lower_violation = self._violation(self._outer_margin - sdf_outer_pred)

            # Reduce via topk: select worst-violating fraction
            def topk_mean(violations: torch.Tensor) -> torch.Tensor:
                if violations.numel() == 0:
                    return torch.tensor(0.0, device=violations.device, dtype=violations.dtype)
                k = max(1, int(violations.numel() * self.__topk_fraction))
                topk_vals = torch.topk(violations, k, largest=True)[0]
                return topk_vals.mean()

            upper_loss = topk_mean(upper_violation)
            lower_loss = topk_mean(lower_violation)

            return 0.5 * (upper_loss + lower_loss)
