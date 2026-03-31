import numpy as np
import torch

from subprojects.wp2_assembler.version4.components.nbz import bounding_box_of_assembly, call_assembly, max_assembly_id
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.s_d_ffrom_cad_query import SDFfromCadQuery
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class BbV4(ParentLoss):
    def _setup(self) -> None:
        self.__config = self.losses_config["bb_config"]
        rng = np.random.default_rng()
        self.assembly_id = int(rng.integers(0, max_assembly_id() + 1))
        # Create SDFs for inner and outer boxes
        inner_box, outer_box, __ = call_assembly(id_number=self.assembly_id)
        inner_solids_list = inner_box.toCompound().Solids()
        self.__inner_SDF: SDFfromCadQuery = SDFfromCadQuery(
            shape_list=inner_solids_list,
            pointgrid_config=self.__config["pointgrid_params"],
            assembly_id=self.assembly_id,
            bounding_box_fn=bounding_box_of_assembly,
        )
        self.__outer_SDF: SDFfromCadQuery = SDFfromCadQuery(
            shape=outer_box,
            pointgrid_config=self.__config["pointgrid_params"],
            assembly_id=self.assembly_id,
            bounding_box_fn=bounding_box_of_assembly,
        )
        self.__pointgrid: list[list[float]] = self.__inner_SDF.pointgrid()
        self.__inner_sdf_values: list[float] = self.__inner_SDF.sdf_at_pointgrid()
        self.__outer_sdf_values: list[float] = self.__outer_SDF.sdf_at_pointgrid()

    def _loss(self, model) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network                                              NO Default!!
        Returns:
            Bounding boxes loss (to be minimized)

        Constrains model SDF to lie between outer and inner SDFs:
            sdf_outer(p) <= sdf_model(p) <= sdf_inner(p)
        Loss = mean( relu(sdf_model - sdf_inner) + relu(sdf_outer - sdf_model) )
        """

        with torch.enable_grad():
            # Automatically find where the model is (CPU or GPU)
            target_device = next(model.parameters()).device
            self.device = target_device

            P = torch.tensor(self.__pointgrid, dtype=torch.float32, device=self.device)  # [N,3]
            sdf_inner_t = torch.as_tensor(
                self.__inner_sdf_values,
                dtype=torch.float32,
                device=self.device,
            )  # [N]
            sdf_outer_t = torch.as_tensor(self.__outer_sdf_values, dtype=torch.float32, device=self.device)  # [N]

            # Batch forward
            output = model(P)
            sdf_pred = output[0] if isinstance(output, tuple) else output

            sdf_pred = sdf_pred.squeeze(-1)  # [N]

            # Box constraints: clamp violations with ReLU
            upper_violation = torch.relu(sdf_pred - sdf_inner_t)
            lower_violation = torch.relu(sdf_outer_t - sdf_pred)

            loss = (upper_violation + lower_violation).mean()  # ** 0.5
            return loss
