

import torch
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp3_solar_estimator.diff.alex_input import dummy_solar_loss
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.differentiability_utils import solar_mesh_deformation, update_from_deformed_mesh
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
    
        verts, faces = get_diff_mesh(model, grid_res=64, device=self.device)
        deformed_verts= solar_mesh_deformation(verts, faces, sun_dir=torch.tensor([0.0, 1.0, 1.0],
                                                device=self.device),
                                                dev=self.device,
                                                slope_param=7.0,
                                                offset_param=3.0)
        loss= update_from_deformed_mesh(deformed_verts, model)
        return dummy_solar_loss(verts, faces, sun_dir=torch.tensor([0.0, -1.0, 1.0], device=self.device))