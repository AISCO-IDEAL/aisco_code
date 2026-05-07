import torch
from typing import Callable

from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5 import ConcatWIREV5
from subprojects.wp4_aeroshell_optimizer.v2.src.models.enforce_projector import AdaNPProjector


class ConcatWIREV5_ENFORCE(ConcatWIREV5):
    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", {})
        if not kwargs.get("config") and len(args) > 4:
            config = args[4] or {}

        super().__init__(*args, **kwargs)

        enforce_cfg = config.get("general_info", {}).get("enforce_projection", {})
        self.enforce_layer = AdaNPProjector(
            max_iters=enforce_cfg.get("max_iters", 5),
            step_size=enforce_cfg.get("step_size", 0.1),
            tolerance=enforce_cfg.get("tolerance", 1e-3),
        )
        self.activation_ratio = enforce_cfg.get("activation_epoch_ratio", 0.9)
        self.current_enforce_context = None

    def register_enforce_constraint(self, context: str, fn: Callable) -> None:
        self.enforce_layer.register_constraint(context, fn)

    def forward(self, coords: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        sdf_raw = super().forward(coords, z)
        return self.enforce_layer(coords, sdf_raw, getattr(self, "current_enforce_context", None))
