from subprojects.wp3_solar_estimator.v1.src.n_diff_solar_loss import solar_loss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class SolarV2(ParentLoss):
    def __init__(self):
        return

    def compute_loss(self, verts, faces):
        loss = solar_loss(verts, faces)
        return loss
