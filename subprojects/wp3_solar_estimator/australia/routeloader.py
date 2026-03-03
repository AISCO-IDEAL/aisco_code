
import numpy as np
import pandas as pd


class Routeloader:
    def __init__(self):
        self.road_data = pd.read_csv("/cluster/home/aalthaus/AISCO_code/subprojects/wp3_solar_estimator/australia/route_preprocessed.csv")  # CSV with 'lat' and 'lon' columns

    def get_position(self, race_distance_km: float) -> tuple[float, float, float]:
            """
            Based on the current race distance along the route, find the corresponding latitude, longitude, and heading.
            """
            lons = self.road_data['longitude'].tolist()
            lats = self.road_data['latitude'].tolist()
            hdgs = self.road_data['theta'].tolist()
            cumdis = self.road_data['cumDistance'].tolist()  # Cumulative distance

            idx = np.argmin(np.abs(np.array(cumdis) - race_distance_km*1000))
            return lats[idx], lons[idx], hdgs[idx]
