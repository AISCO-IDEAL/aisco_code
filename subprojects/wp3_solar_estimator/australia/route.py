import torch

from subprojects.wp3_solar_estimator.australia.routeloader import Routeloader
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class Route:
    def __init__(self):
        loader = Routeloader()
        config = load_config(
            "/cluster/home/aalthaus/AISCO_code/subprojects/wp4_aeroshell_optimizer/v1/configs/australia.yaml"
        )  # fix hardcode sometime
        self.route_points: torch.Tensor  # Shape (N, 3) tensor of [lat, lon, heading] for each point along the route
        # TODO figure out if theta in csv is heading
