import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.solar_loss.solar_dummy import (
    solar_mesh_deformation,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.differentiability_utils import (
    loss_from_deformed_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import (
    get_diff_mesh,
)


class SolarV0(ParentLoss):
    def _setup(self) -> None:
        return

    def _loss(self, model: BaseModel, grid_res: int = 64) -> torch.Tensor:
        """
        Compute top area loss (maximize solar collection area on top surface).
        NOTE: Still WIP, currently running a dummy version of the loss
        NOTE: Does not work, as long as the kaolin_utils are not implemented
        """

        verts, faces = get_diff_mesh(model, grid_res=grid_res, device=self.device)
        deformed_verts = solar_mesh_deformation(
            verts,
            faces,
            sun_dir=torch.tensor([0.0, 1.0, 1.0], device=self.device),
            dev=self.device,
            slope_param=7.0,
            offset_param=3.0,
        )
        loss = loss_from_deformed_mesh(deformed_verts, model)
        return loss
