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
            self.points = get_points(
                num_points=self.losses_config["get_points_params"]["num_points"],
                bbox_size=self.losses_config["get_points_params"]["bbox_size"],
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

