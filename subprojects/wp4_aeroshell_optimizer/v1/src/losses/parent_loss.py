"""
Use this class as a Parent-class for all losses.
Do not overwrite the following methods:
- __init__
- get_loss
Overwrite the following methods:
- _setup()
- _loss()
"""

import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.losses_utils import get_points


class ParentLoss:
    def __init__(
        self,
        device: str = "cuda",
        losses_config: dict | None = None,
        weight: list[float] | None = None,
        loss_name: str = "",
    ) -> None:
        # general initialisation
        self.device = device
        self.general_config = load_config()
        self.losses_config = losses_config if losses_config is not None else {}
        self.loss_name = loss_name
        self.training_config = load_config(
            v=self.general_config["general_info"].get("version", "v1"),
            cfg_name=self.general_config["active_configs"]["training"],
        )

        # initialise weight
        self.__weight = weight if weight is not None else [0.0, 0.0]
        if self.__weight == [0.0, 0.0] and loss_name:
            for losses in self.losses_config.get("losses", {}).values():
                if isinstance(losses, dict) and loss_name in losses:
                    self.__weight = losses[loss_name]
                    break

        if self.__weight[0] == 0.0:
            return

        # Initialise points if eikonal or smoothness loss
        if loss_name and ("eikonal" in loss_name.lower() or "smoothness" in loss_name.lower()):
            print(f"DEBUG: Initializing points for {loss_name} on device: {self.device}")
            self.points = get_points(
                num_points=self.losses_config["get_points_params"]["num_points"],
                bbox_size=self.losses_config["get_points_params"]["bbox_size"],
                device=self.device,
            )
            print(f"DEBUG: Points device: {self.points.device}")

        self._setup()

    def _setup(self) -> None:
        return

    def get_loss(self, model, epoch_name) -> torch.Tensor:
        if self.__weight[0] == 0.0:
            return torch.tensor(0.0, device=self.device)
        else:
            # store context for gradient logging
            self.epoch_name = epoch_name
            self._last_model = model
            self._last_loss = self._loss(model)
            return self._last_loss

    def _loss(self, model) -> torch.Tensor:  # noqa: ARG002
        return torch.tensor(0.0, device=self.device)

    def log_gradients(self, epoch_name) -> None:
        """Log gradient norm for this loss in a robust way.

        Prefer parameter gradients computed via autograd if tensor.grad is None.
        """
        # If there is no stored loss or weight is zero, skip
        if not hasattr(self, "_last_loss") or self._last_loss is None:
            return

        # Diagnostics: check if the loss requires gradient
        loss_requires_grad = bool(getattr(self._last_loss, "requires_grad", False))
        # If loss doesn't require grad, don't call autograd.grad to avoid RuntimeError.
        if not loss_requires_grad:
            return

        total_gradient: float = 0.0
        params = [p for p in self._last_model.parameters() if p.requires_grad]

        grads = torch.autograd.grad(
            self._last_loss,
            params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        grads = [g for g in list(grads) if g is not None]
        if grads:
            dev = grads[0].device
            norm_sq = torch.tensor(0.0, device=dev)
            for g in grads:
                norm_sq = norm_sq + g.detach().pow(2).sum()
            total_gradient = torch.sqrt(norm_sq).item()

            wandb.log({f"{self.loss_name}_gradient": total_gradient}, step=epoch_name)
