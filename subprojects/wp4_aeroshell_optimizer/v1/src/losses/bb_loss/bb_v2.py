import cadquery as cq
import torch

from subprojects.wp2_assembler.archive.version1.components.nbz import innerNBZ, outerNBZ
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.s_d_ffrom_cad_query import (
    SDFfromCadQuery,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class BbV2(ParentLoss):
    def _setup(self) -> None:
        # Create SDFs for inner and outer boxes
        inner_box: cq.Solid = innerNBZ(
            random_magnitude_chassis=0.0,
            random_magnitude_seat=0.0,
            random_magnitude_wheels=0.0,
        )
        self.__inner_SDF: SDFfromCadQuery = SDFfromCadQuery(
            shape=inner_box, training_config=self.training_config
        )
        outer_box: cq.Solid = outerNBZ()
        self.__outer_SDF: SDFfromCadQuery = SDFfromCadQuery(
            shape=outer_box, training_config=self.training_config
        )
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
        # Helper to get SDF
        def get_sdf(points):
            out = model(points)
            return out[0] if isinstance(out, tuple) else out

        inner_bb_points_model = get_sdf(
            torch.tensor(
                self.__inner_bb_points, dtype=torch.float32, device=self.device
            )
        )
        outer_bb_points_model = get_sdf(
            torch.tensor(
                self.__outer_bb_points, dtype=torch.float32, device=self.device
            )
        )

        # compute the loss
        inner_loss = torch.relu(inner_bb_points_model[0]).mean() ** 0.5
        outer_loss = torch.relu(-outer_bb_points_model[0]).mean() ** 0.5
        loss = inner_loss + outer_loss
        return loss
