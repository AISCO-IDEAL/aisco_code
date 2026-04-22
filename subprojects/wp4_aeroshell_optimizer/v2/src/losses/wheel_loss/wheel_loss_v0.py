import math
from pathlib import Path

import torch
import yaml
from AISCO_code.subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation
from AISCO_code.subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class WheelLossV0(ParentLoss):
    def _setup(self) -> None:
        self.__config = self.losses_config["wheel_loss_config"]
        self.assembly_id = int(self.losses_config["wheel_loss_config"].get("assembly_id", 1))

        assembly_map_path = Path(__file__).parents[6] / "subprojects/wp2_assembler/version4/config/assembly_map.yaml"
        with assembly_map_path.open("r") as f:
            assembly_map = yaml.safe_load(f) or {}
        params = assembly_map[self.assembly_id]

        # get wheel coordinates (extends width/2 in +y/-y, radius in +z)
        wheel_coords = [(p["x"], p["y"], p["z"]) for key, p in params.items() if key.startswith("wheel_")]

        width = get_regulation("wheel.tire_width")
        radius = get_regulation("wheel.tire_diameter") / 2
        gc = get_regulation("no_build_zones.outer_prism.ground_clearance")
        C = radius - gc - 1
        L = math.sqrt(radius**2 - C**2)

        # points to evaluate
        sampling_density = self.__config.get("sampling_density", 10)  # points per edge of the rectangle
        self.points = [
            (
                (wx + dx * L, wy + dy * width / 2, wz - C)
                for dx in torch.linspace(-1, 1, sampling_density)
                for dy in torch.linspace(-1, 1, sampling_density)
            )
            for wx, wy, wz in wheel_coords
        ]

        print(self.points)

    def _loss(self, model) -> torch.Tensor:
        """Sample points on the wheel interface rectangle and compute loss based on sdf_model values at those points. SDF in this are has to be 0."""
        # Sample points on SDF
        samples = model(self.points)  # shape: (num_wheels, sampling_density^2, 3)

        # Calculate loss as mean squared value of SDF at sampled points
        loss = torch.mean(torch.abs(samples))
        return loss
