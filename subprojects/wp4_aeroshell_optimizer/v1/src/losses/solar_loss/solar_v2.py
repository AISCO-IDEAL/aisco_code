from subprojects.wp3_solar_estimator.v2 import solar_loss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class SolarV2(ParentLoss):
    def _setup(self) -> None:
        return

    def _loss(self, model):
        # Delegate to solar_loss, passing the model and correct device
        return solar_loss(model=model, device=self.device)
