import torch
from AISCO_code.subprojects.wp2_assembler.version3.components.nbz import call_assembly
from AISCO_code.subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class WheelLossV0(ParentLoss):
    def _setup(self) -> None:
        self.__config = self.losses_config["wheel_loss_config"]
        self.assembly_id = int(self.__config.get("assembly_id", 1))

        inner_assembly, __, __ = call_assembly(id_number=self.assembly_id)

        # get coordinates of wheel interfaces (assuming at least 3 wheels exist)
        # inner_solids[0] is the seat; inner_solids[1+] are wheel solids
        inner_solids = inner_assembly.toCompound().Solids()
        wheels: list[list] = [[], []]  # list of [solid, coordinates]
        for solid in inner_solids[1:]:
            wheels[0].append(solid)
            # get coordinates of each wheel
            wheels[1].append(solid.BoundingBox().center)

    def _loss(self, model) -> torch.Tensor:
        """Placeholder loss function; replace with actual implementation."""
        return torch.tensor(0.0, device=model.device)
