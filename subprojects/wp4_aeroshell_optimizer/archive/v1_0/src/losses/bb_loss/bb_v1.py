import cadquery as cq
import torch

from subprojects.wp2_assembler.archive.version1.components.nbz import innerNBZ, outerNBZ
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.s_d_ffrom_cad_query import SDFfromCadQuery
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class BbV1(ParentLoss):
    def _setup(self) -> None:
        # Create SDFs for inner and outer boxes
        inner_box: cq.Solid = innerNBZ(
            random_magnitude_chassis=0.0,
            random_magnitude_seat=0.0,
            random_magnitude_wheels=0.0,
        )
        self.__inner_SDF: SDFfromCadQuery = SDFfromCadQuery(shape=inner_box, training_config=self.training_config)
        outer_box: cq.Solid = outerNBZ()
        self.__outer_SDF: SDFfromCadQuery = SDFfromCadQuery(shape=outer_box, training_config=self.training_config)
        self.__pointgrid: list[list[float]] = self.__inner_SDF.pointgrid()
        self.__inner_sdf_values: list[float] = self.__inner_SDF.sdf_at_pointgrid()
        self.__outer_sdf_values: list[float] = self.__outer_SDF.sdf_at_pointgrid()

    def _loss(self, model) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network                                              NO Default!!
            inner_box: CadQuery Shape representing the inner bounding box       Defaults to BackupAssembly boxes if None
            outer_box: CadQuery Shape representing the outer bounding box       Defaults to BackupAssembly boxes if None
            pointgrid: List of points at which compatibility is enforced        Defaults to a 21x21x21 grid from -2 to 2 in each axis if None
        Returns:
            Bounding boxes loss (to be minimized)

        Constrains model SDF to lie between outer and inner SDFs:
            sdf_outer(p) <= sdf_model(p) <= sdf_inner(p)
        Loss = mean( relu(sdf_model - sdf_inner) + relu(sdf_outer - sdf_model) )
        """
        with torch.enable_grad():
            P = torch.tensor(self.__pointgrid, dtype=torch.float32, device=self.device)  # [N,3]
            sdf_inner_t = torch.as_tensor(
                self.__inner_sdf_values,
                dtype=torch.float32,
                device=self.device,
            )  # [N]
            sdf_outer_t = torch.as_tensor(self.__outer_sdf_values, dtype=torch.float32, device=self.device)  # [N]

            # Batch forward
            output = model(P)
            if isinstance(output, tuple):
                sdf_pred = output[0]
            else:
                sdf_pred = output

            sdf_pred = sdf_pred.squeeze(-1)  # [N]

            # Box constraints: clamp violations with ReLU
            upper_violation = torch.relu(sdf_pred - sdf_inner_t)
            lower_violation = torch.relu(sdf_outer_t - sdf_pred)

            loss = (upper_violation + lower_violation).mean() ** 0.5
            return loss
            return loss
            loss = (upper_violation + lower_violation).mean() ** 0.5
            return loss
            return loss
            loss = (upper_violation + lower_violation).mean() ** 0.5
            return loss
            return loss
