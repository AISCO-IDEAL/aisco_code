from subprojects.wp3_solar_estimator.v7.solar_loss_v7 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config


class SolarV7(ParentLoss):
    def _setup(self) -> None:
        gen_cfg = load_config(v="v1", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        solar_cfg = load_config(v="v1", cfg_name=cfg_path)["SolarLossConfigV7"]
        self.solar_estimator = SolarLossEstimator(config=solar_cfg)
        self.power_conversion_factor = solar_cfg["power_conversion_factor"]

    def _loss(self, model):
        import torch  # noqa: PLC0415

        raw_loss = self.solar_estimator.solar_loss(model=model, epoch_name=self.epoch_name)
        scaled_loss = raw_loss * self.power_conversion_factor

        with torch.no_grad():
            self._last_diagnostics = {
                "loss_value_raw": raw_loss.item(),
                "loss_value_scaled": scaled_loss.item(),
                "power_conversion_factor": self.power_conversion_factor,
                "method": self.solar_estimator.method,
                "counter": self.solar_estimator.counter,
                "area_budget_mm2": self.solar_estimator.area_budget_mm2,
                "resolution": self.solar_estimator.resolution,
            }

        return scaled_loss
