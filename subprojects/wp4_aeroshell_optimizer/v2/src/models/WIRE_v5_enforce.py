import torch
from typing import Callable

from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5 import WIREV5
from subprojects.wp4_aeroshell_optimizer.v2.src.models.enforce_projector import AdaNPProjector


class WIREV5_ENFORCE(WIREV5):
    def __init__(self, *args, **kwargs):
        config = kwargs.get("config", {})
        if not kwargs.get("config") and len(args) > 3:
            config = args[3] or {}

        super().__init__(*args, **kwargs)

        enforce_cfg = config.get("general_info", {}).get("enforce_projection", {})
        self.enforce_layer = AdaNPProjector(
            max_iters=enforce_cfg.get("max_iters", 5),
            tolerance=enforce_cfg.get("tolerance", 1e-3),
        )
        self.activation_ratio = enforce_cfg.get("activation_epoch_ratio", 0.9)
        self.current_enforce_context = None

    def register_enforce_constraint(self, context: str, fn: Callable) -> None:
        self.enforce_layer.register_constraint(context, fn)

    def on_epoch_start(self, epoch: int, num_epochs: int) -> None:
        if epoch == int(num_epochs * self.activation_ratio) and not self.enforce_layer.is_active:
            print(f"Activating ENFORCE projection layer at epoch {epoch} / {num_epochs}")
            self.enforce_layer.is_active = True

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        sdf_raw = super().forward(coords)
        return self.enforce_layer(coords, sdf_raw, getattr(self, "current_enforce_context", None))
