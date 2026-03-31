from subprojects.wp3_solar_estimator.v5.solar_loss_v5 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class SolarV5(ParentLoss):
    def _setup(self) -> None:
        gen_cfg = load_config(v="v1", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        solar_cfg = load_config(v="v1", cfg_name=cfg_path)["SolarLossConfigV5"]
        self.solar_estimator = SolarLossEstimator(config=solar_cfg)

    def _loss(self, model):
        import torch  # noqa: PLC0415

        loss = self.solar_estimator.solar_loss(model=model, epoch_name=self.epoch_name)

        with torch.no_grad():
            self._last_diagnostics = {
                "loss_value": loss.item(),
                "method": self.solar_estimator.method,
                "counter": self.solar_estimator.counter,
                "area_budget_mm2": self.solar_estimator.area_budget_mm2,
                "resolution": self.solar_estimator.resolution,
            }

        return loss
