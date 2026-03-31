import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.model_loader import ModelLoader as ModelLoader2
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.model_loader import ModelLoader as ModelLoader1
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class GaotLoss(ParentLoss):
    def _setup(self) -> None:
        version = self.losses_config["cfd"]["version"] if self.losses_config["cfd"]["version"] is not None else 1

        self.gaot: ModelLoader1 | ModelLoader2
        if version == 1:
            self.gaot = ModelLoader1(cfd_config=self.losses_config["cfd"], device=self.device)
        elif version == 2:
            self.gaot = ModelLoader2(cfd_config=self.losses_config["cfd"], device=self.device)
        else:
            raise ValueError(f"Gaot Loss Version {version} is not implemented")

    def _loss(self, model) -> torch.Tensor:
        return self.gaot.forward(sdf_model=model, epoch_name=self.epoch_name)
