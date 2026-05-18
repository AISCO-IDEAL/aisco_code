import torch

from subprojects.wp3_solar_estimator.australia.route_v3 import RouteV3


class RouteV1(RouteV3):
    """Thin shim — delegates to RouteV3 with mode="time_sweep"."""

    def __init__(self, cfg_path: str = "solar_configs/australia.yaml") -> None:
        super().__init__(cfg_path, mode="time_sweep")

    def get_route_solar_vecs(self, index: int | None = None) -> torch.Tensor:
        """
        Returns sun vectors in the car body frame.

        If index is provided, returns shape (T, 3) for that route point across all time samples.
        Otherwise returns shape (N, T, 3) — all route points × all time samples.
        """
        T = len(self.times)
        all_vecs = self.solar_vectors()  # (N*T, 3)
        if index is not None:
            return all_vecs[index * T : (index + 1) * T]
        return all_vecs.view(self.num_points, T, 3)
