from pathlib import Path

import yaml

from subprojects.wp3_solar_estimator.v3 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class SolarV3(ParentLoss):
    def _setup(self) -> None:
        config_path = Path("subprojects/wp4_aeroshell_optimizer/v1/configs/solar_loss_config.yaml")
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f)

        self.solar_estimator = SolarLossEstimator(config=cfg)

    def _loss(self, model):
        return self.solar_estimator.solar_loss(model=model, epoch_name=self.epoch_name)
