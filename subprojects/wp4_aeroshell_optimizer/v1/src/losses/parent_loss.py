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
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import (
    load_general_config,
    load_specific_config,
)


class ParentLoss:
    def __init__(self, device: str = "cuda", weight: list[float] = [0.0, 0.0]) -> None:
        # general
        self.device = device
        self.general_config = load_general_config()
        # losses
        self.losses_config = load_specific_config(
            v=self.general_config["general_info"]["version"],
            cfg_name=self.general_config["active_configs"]["losses"],
        )
        self.weights = self.losses_config.get("losses")
        # training
        self.training_config = load_specific_config(
            v=self.general_config["general_info"]["version"],
            cfg_name=self.general_config["active_configs"]["training"],
        )
        self.device = device
        self.__weight = weight
        if self.__weight[0] == 0.0:
            return
        else:
            self._setup()
            return

    def _setup(self) -> None:
        return

    def get_loss(self, model) -> torch.Tensor:
        if self.__weight[0] == 0.0:
            return torch.tensor(0.0, device=self.device)
        else:
            return self._loss(model) * self.__weight[1]

    def _loss(self, model) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)
