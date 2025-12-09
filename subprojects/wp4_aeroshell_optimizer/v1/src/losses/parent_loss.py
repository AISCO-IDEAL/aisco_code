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

from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.losses_utils import get_points


class ParentLoss:
    def __init__(
        self,
        device: str = "cuda",
        losses_config: dict = None,
        weight: list[float] = [0.0, 0.0],
        loss_name: str = None,
    ) -> None:
        # general initialisation
        self.device = device
        self.general_config = load_config()
        self.losses_config = losses_config
        self.training_config = load_config(
            v=self.general_config["general_info"].get("version", "v1"),
            cfg_name=self.general_config["active_configs"]["training"],
        )

        # initialise weight
        self.__weight = weight
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
            )

        self._setup()

    def _setup(self) -> None:
        return

    def get_loss(self, model) -> torch.Tensor:
        if self.__weight[0] == 0.0:
            return torch.tensor(0.0, device=self.device)
        else:
            return self._loss(model) * self.__weight[1]

    def _loss(self, model) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

