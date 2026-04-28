import math
from pathlib import Path

import torch
import yaml

from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation


def main_test():
    id_number = 2
    assembly_map_path = Path(__file__).parents[6] / "subprojects/wp2_assembler/version4/config/assembly_map.yaml"
    with assembly_map_path.open("r") as f:
        assembly_map = yaml.safe_load(f) or {}
    params = assembly_map[id_number]

    # get wheel coordinates (extends width/2 in +y/-y, radius in +z)
    wheel_coords = [(p["x"], p["y"], p["z"]) for key, p in params.items() if key.startswith("wheel_")]
    width = get_regulation("wheel.tire_width")
    radius = get_regulation("wheel.tire_diameter") / 2
    gc = get_regulation("no_build_zones.outer_prism.ground_clearance")
    C = radius - gc - 1
    L = math.sqrt(radius**2 - C**2)

    # points to evaluate
    sampling_density = 2  # points per edge of the rectangle
    points = [
        [
            (wx + dx * L, wy + dy * width / 2, wz - C)
            for dx in torch.linspace(-1, 1, sampling_density)
            for dy in torch.linspace(-1, 1, sampling_density)
        ]
        for wx, wy, wz in wheel_coords
    ]
    print(points)


if __name__ == "__main__":
    main_test()
