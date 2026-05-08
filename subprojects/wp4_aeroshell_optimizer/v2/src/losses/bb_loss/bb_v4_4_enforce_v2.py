import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import BbV44
from subprojects.wp4_aeroshell_optimizer.v2.src.models.enforce_projector import enforce_layer_context


class BbV44EnforceV2(BbV44):
    """
    Two-phase ENFORCE-aware bb loss.

    Phase 1 (enforce projector inactive): standard relu/softplus violation on sdf_raw,
    identical to BbV44 — keeps the constraint pressure on throughout the early epochs so
    the network learns a roughly-feasible shape.

    Phase 2 (projector active, last `1 - activation_epoch_ratio` of training): switches
    to a projection-as-target loss `(sdf_raw - sdf_proj.detach())²`. sdf_proj is the
    Newton-projected (constraint-satisfying) value at each sample point, used as a frozen
    target. The loss directly pulls θ to make sdf_raw match the projected value — i.e.
    nudges the network's zero-crossing onto the inner/outer bb surfaces. Mathematically
    a quadratic violation: stronger gradient near boundary than the linear relu.
    """

    def _setup(self) -> None:
        super()._setup()
        self.__enforce_registered_v2 = False

    def _loss(self, model) -> torch.Tensor:
        if not self.__enforce_registered_v2 and hasattr(model, "register_enforce_constraint"):
            inner_margin = self._BbV44__inner_margin
            outer_margin = self._BbV44__outer_margin
            violation_fn = self._violation

            def inner_constraint(_coords, sdf):
                return violation_fn(sdf.squeeze(-1) + inner_margin)

            def outer_constraint(_coords, sdf):
                return violation_fn(outer_margin - sdf.squeeze(-1))

            model.register_enforce_constraint("bb_inner", inner_constraint)
            model.register_enforce_constraint("bb_outer", outer_constraint)
            self.__enforce_registered_v2 = True

        target_device = next(model.parameters()).device
        self.device = target_device

        inner_points = self._BbV44__inner_surface_points.to(self.device)
        outer_points = self._BbV44__outer_surface_points.to(self.device)

        sdf_inner = model(inner_points).squeeze(-1)
        sdf_outer = model(outer_points).squeeze(-1)

        enforce_layer = getattr(model, "enforce_layer", None)
        if enforce_layer is None or not enforce_layer.is_active:
            upper_violation = self._violation(sdf_inner + self._BbV44__inner_margin)
            lower_violation = self._violation(self._BbV44__outer_margin - sdf_outer)
            return 0.5 * (upper_violation.mean() + lower_violation.mean())

        with enforce_layer_context(model, "bb_inner"):
            target_inner = model(inner_points).squeeze(-1).detach()
        with enforce_layer_context(model, "bb_outer"):
            target_outer = model(outer_points).squeeze(-1).detach()

        inner_loss = (sdf_inner - target_inner).pow(2).mean()
        outer_loss = (sdf_outer - target_outer).pow(2).mean()
        return 0.5 * (inner_loss + outer_loss)
