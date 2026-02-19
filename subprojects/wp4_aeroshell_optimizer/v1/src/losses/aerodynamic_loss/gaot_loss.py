import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.model_loader import ModelLoader
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class GaotLoss(ParentLoss):
    def _setup(self) -> None:
        model_path = self.losses_config["cfd"]["model_path"]
        self.gaot = ModelLoader(model_path=model_path, device=self.device)

    def _loss(self, model) -> torch.Tensor:
        return self.gaot.forward(sdf_model=model)
