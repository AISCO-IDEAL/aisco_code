import os

import pandas as pd
import torch

_CSV_PATH = os.path.join(os.path.dirname(__file__), "route_preprocessed.csv")


class Routeloader:
    def __init__(self):
        self.road_data = pd.read_csv(_CSV_PATH)  # CSV with 'lat' and 'lon' columns

    def get_position(self, race_distance_km: float | torch.Tensor) -> torch.Tensor:
        """
        Based on the current race distance along the route, find the corresponding latitude, longitude, and heading.

        Returns a tensor of shape (3,) containing [latitude, longitude, heading] for the point along the route corresponding to the given distance
        """
        if race_distance_km > 3030:
            raise ValueError(
                f"Race distance {race_distance_km} km significantly exceeds (by more than a few kms) the total route length of 3027 km found in our csv file. Raising warning since this likely causes unintended behaviour."
            )
        lons = self.road_data["longitude"].tolist()
        lats = self.road_data["latitude"].tolist()
        hdgs = self.road_data["theta"].tolist()
        cumdis = self.road_data["cumDistance"].tolist()  # Cumulative distance

        idx = torch.argmin(torch.abs(torch.tensor(cumdis, dtype=torch.float32) - race_distance_km * 1000))
        return torch.tensor([lats[idx], lons[idx], hdgs[idx]], dtype=torch.float32)
