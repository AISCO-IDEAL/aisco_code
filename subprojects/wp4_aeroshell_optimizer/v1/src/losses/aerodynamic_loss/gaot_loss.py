import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.model_loader import ModelLoader
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class GaotLoss(ParentLoss):
    def _setup(self) -> None:
        self.gaot = ModelLoader(cfd_config=self.losses_config["cfd"], device=self.device)

    def _loss(self, model) -> torch.Tensor:
        return self.gaot.forward(sdf_model=model, epoch_name=self.epoch_name)
