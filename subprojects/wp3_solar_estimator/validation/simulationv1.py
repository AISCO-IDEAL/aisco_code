"""
For validation purposes, runs a basic race simulation to predict the total solar output during the race
Assumes a constant speed given in the australia.yaml config file, and a constant solar output per unit area given in the same config file.
"""

from subprojects.wp3_solar_estimator.australia.route_v2 import RouteV2


class SimulationV1:
    def __init__(self, cfg_path: str = "solar_configs/australia.yaml") -> None:
        self.route = RouteV2(cfg_path=cfg_path)

    def run(self):
        pass
