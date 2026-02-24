from subprojects.wp3_solar_estimator.v3 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class SolarV3(ParentLoss):
    def _setup(self) -> None:
        cfg = load_config(v="v1", cfg_name="solar_loss_config.yaml")
        self.solar_estimator = SolarLossEstimator(config=cfg["SolarLossConfigV3"])

    def _loss(self, model):
        return self.solar_estimator.solar_loss(model=model, epoch_name=self.epoch_name)
