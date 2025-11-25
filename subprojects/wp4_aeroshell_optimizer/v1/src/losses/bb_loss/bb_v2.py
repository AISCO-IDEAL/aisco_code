import cadquery as cq
import torch

from subprojects.wp2_assembler.archive.version1.components.nbz import innerNBZ, outerNBZ
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.cadquery_to_sdf import (
    SDFfromCadQuery,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class BbLoss2(ParentLoss):
    def _setup(self)->None:
        # Create SDFs for inner and outer boxes
        inner_box: cq.Solid = innerNBZ(
            random_magnitude_chassis=0.0,
            random_magnitude_seat=0.0,
            random_magnitude_wheels=0.0,
        )
        self.__inner_SDF: SDFfromCadQuery = SDFfromCadQuery(inner_box)
        outer_box: cq.Solid = outerNBZ()
        self.__outer_SDF: SDFfromCadQuery = SDFfromCadQuery(outer_box)
        self.__pointgrid: list[list[float]] = self.__inner_SDF.pointgrid()
        self.__inner_sdf_values: list[float] = self.__inner_SDF.sdf_at_pointgrid()
        self.__outer_sdf_values: list[float] = self.__outer_SDF.sdf_at_pointgrid()

        self.__inner_bb_points = [
            point
            for point, sdf_value in zip(
                self.__pointgrid, self.__inner_sdf_values, strict=False
            )
            if sdf_value < 0
        ]

        # get all points where sdf is positive (outside the outer box)
        self.__outer_bb_points = [
            point
            for point, sdf_value in zip(
                self.__pointgrid, self.__outer_sdf_values, strict=False
            )
            if sdf_value > 0
        ]

    def _loss(self, model) -> torch.Tensor:
        inner_bb_points_model = model(
            torch.tensor(
                self.__inner_bb_points, dtype=torch.float32, device=self.device
            )
        )[0]
        outer_bb_points_model = model(
            torch.tensor(
                self.__outer_bb_points, dtype=torch.float32, device=self.device
            )
        )[0]

        # compute the loss
        inner_loss = torch.relu(inner_bb_points_model[0]).mean() ** 0.5
        outer_loss = torch.relu(-outer_bb_points_model[0]).mean() ** 0.5
        loss = inner_loss + outer_loss
        return loss

