"""
Solar V3A_2 wrapper: Solar loss with meaningful watt output.

Returns loss = -power_watts, so minimizing loss maximizes power generation.
Gradients flow backward through the power computation.
"""

from subprojects.wp3_solar_estimator.v3A_2.solar_loss_v3A_2 import SolarLossEstimator

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class SolarV3A2(ParentLoss):
    """Solar loss optimizer wrapper for v3A_2 (meaningful watt-based loss)."""

    def _setup(self) -> None:
        gen_cfg = load_config(v="v1", cfg_name="general_config.yaml")
        cfg_path = gen_cfg["active_configs"]["solar"]
        # Uses SolarLossConfigV3 config; can be customized per-version
        solar_cfg = load_config(v="v1", cfg_name=cfg_path)["SolarLossConfigV3"]
        self.solar_estimator = SolarLossEstimator(config=solar_cfg)
        self.power_conversion_factor = solar_cfg.get("power_conversion_factor", 1.0)

    def _loss(self, model):
        import torch  # noqa: PLC0415

        # loss = -power_watts (from solar_mesh_loss_v3)
        raw_loss = self.solar_estimator.solar_loss(disable_logging=skip_telemetry, disable_rendering=skip_telemetry, model=model, epoch_name=self.epoch_name)
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
                "solar_irradiance_W_m2": self.solar_estimator.solar_irradiance_W_m2,
                "efficiency": self.solar_estimator.efficiency,
            }

        return scaled_loss
