from subprojects.wp3_solar_estimator.v2 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class SolarV2(ParentLoss):
    def _setup(self) -> None:
        self.solar_estimator = SolarLossEstimator(device=self.device, area_budget=6e6)

    def _loss(self, model):
        return self.solar_estimator.solar_loss(model=model)
