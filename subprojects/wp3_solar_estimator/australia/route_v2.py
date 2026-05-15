import torch

from subprojects.wp3_solar_estimator.australia.route_v3 import RouteV3


class RouteV2(RouteV3):
    """Thin shim — delegates to RouteV3 with mode="race"."""

    def __init__(self, cfg_path: str = "solar_configs/australia.yaml") -> None:
        super().__init__(cfg_path, mode="race")

    def get_route_solar_vecs(self, index: int | None = None) -> torch.Tensor:
        """
        Returns sun vectors in the car body frame.

        If index is provided, returns shape (3,) for that single route point.
        Otherwise returns shape (N, 3) for all route points.
        """
        if index is not None:
            return self.solar_vectors()[index]
        return self.solar_vectors()
