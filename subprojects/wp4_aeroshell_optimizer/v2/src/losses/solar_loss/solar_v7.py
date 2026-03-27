from subprojects.wp3_solar_estimator.v7.solar_loss_v7 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config


class SolarV7(ParentLoss):
    def _setup(self) -> None:
        gen_cfg = load_config(v="v1", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        solar_cfg = load_config(v="v1", cfg_name=cfg_path)["SolarLossConfigV7"]
        self.solar_estimator = SolarLossEstimator(config=solar_cfg)

    def _loss(self, model):
        return self.solar_estimator.solar_loss(model=model, epoch_name=self.epoch_name)
