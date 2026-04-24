"""
Use this class as a Parent-class for all losses.
Do not overwrite the following methods:
- __init__
- get_loss
Overwrite the following methods:
- _setup()
- _loss()
"""

import inspect

import torch

import wandb
from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.losses_utils import get_points


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
            # Get outer-prism bounds from wp2 (single source of truth)
            prism = get_outer_prism_bounds_mm()
            scale = float(self.losses_config["get_points_params"].get("bbox_scale_factor", 1.0))
            bbox_lo = [
                prism["x"][0] * scale,
                prism["y"][0] * scale,
                prism["z_sampling"][0] * scale,
            ]
            bbox_hi = [
                prism["x"][1] * scale,
                prism["y"][1] * scale,
                prism["z_sampling"][1] * scale,
            ]
            self.points = get_points(
                num_points=self.losses_config["get_points_params"]["num_points"],
                bbox_lo=bbox_lo,
                bbox_hi=bbox_hi,
                device=self.device,
            )

        self._setup()

    def _setup(self) -> None:
        return

    def get_loss(self, model, epoch_name, skip_telemetry=False) -> torch.Tensor:
        if self.__weight[0] == 0.0:
            return torch.tensor(0.0, device=self.device)
        else:
            # store context for gradient logging
            self.epoch_name = epoch_name
            self._last_model = model

            sig = inspect.signature(self._loss)
            accepts_skip_telemetry = "skip_telemetry" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            if accepts_skip_telemetry:
                self._last_loss = self._loss(model, skip_telemetry=skip_telemetry)  # type: ignore[call-arg]
            else:
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

        raw_grads = torch.autograd.grad(
            self._last_loss,
            params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        active_grads = [g for g in list(raw_grads) if g is not None]
        if active_grads:
            dev = active_grads[0].device
            norm_sq = torch.tensor(0.0, device=dev)
            for g in active_grads:
                norm_sq = norm_sq + g.detach().pow(2).sum()
            total_gradient = torch.sqrt(norm_sq).item()

            wandb.log({f"{self.loss_name}_gradient": total_gradient}, step=epoch_name)

    def log_gradient_norm(self, norm: float, epoch_name: int) -> None:
        """Log a pre-computed gradient norm for this loss (no autograd involved).

        Called by the training loop after per-loss backward() passes, where the gradient
        norm is computed externally from param.grad deltas. This avoids retain_graph=True.
        """
        if norm > 0.0:
            wandb.log({f"{self.loss_name}_gradient": norm}, step=epoch_name)
