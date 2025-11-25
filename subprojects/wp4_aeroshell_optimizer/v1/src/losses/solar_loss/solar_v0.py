

import torch
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp3_solar_estimator.diff.alex_input import dummy_solar_loss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.solar_loss.kaolin_utils_for_dummy_solar_loss import marching_tetrahedra


class SolarLoss(ParentLoss):
    def _setup(self)->None:
        return
    
    def _loss(self, model)-> torch.Tensor:
        resolution: int = 128
        length: float = 8.0
        width: float = 6.0
        """
        Compute top area loss (maximize solar collection area on top surface).
        NOTE: Still WIP, currently running a dummy version of the loss 
        NOTE: Does not work, as long as the kaolin_utils are not implemented
        """
        verts, faces = marching_tetrahedra(model, grid_res=128, dev=self.device)
        return dummy_solar_loss(verts, faces, sun_dir=torch.tensor([0.0, -1.0, 1.0], device=self.device))