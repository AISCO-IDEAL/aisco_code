import datetime

import pytz
import torch

from subprojects.wp3_solar_estimator.australia.routeloader import Routeloader
from subprojects.wp3_solar_estimator.australia.sun import get_solar_car_vec, get_solar_car_vecs_batch
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class Route:
    def __init__(self, cfg_path: str = "solar_configs/australia.yaml") -> None:
        """
        Loads the route points from the CSV file and stores them in a tensor for efficient access.
        Each point (saved as self.route_points) is represented as an array of (latitude, longitude, heading).
        Use get_route_points() to retrieve the full tensor or a specific point by index.
        """
        loader = Routeloader()
        config = load_config(cfg_name=cfg_path)
        #init vars

        #round up race length and divide by intervals to get number of points
        race_length = config["route_generation"]["race_distance"]  # in km
        interval = config["route_generation"]["intervals"]  # in km
        num_points = int((race_length + 1) // interval)
        self.route_points = torch.zeros((num_points, 3), dtype=torch.float32) # Shape (N, 3) tensor of [lat, lon, heading] for each point along the route

        self.times : list[datetime.datetime] = [] #array of timestamps (datetime format)
        self.num_solar_samples : int = config["sun_generation"]["solar_samples"] # in m/s

        #make timestamps
        start_date = config["sun_generation"]["start_date"]  # in DD/MM/YYYY, Australian time
        racing_window= config["sun_generation"]["racing_window"]  # in hours, e.g. [8, 17.5] for 8:00 to 17:30
        timezone= config["general"]["timezone"]  # e.g. "ACST"
        self.start_time = datetime.datetime.strptime(start_date, "%d/%m/%Y")
        tz = pytz.timezone(timezone)
        self.start_time = tz.localize(self.start_time)
        #uniformly sample timestamps across the racing window based on the number of solar samples
        self.times = [self.start_time + datetime.timedelta(hours=racing_window[0] + i * (racing_window[1] - racing_window[0]) / self.num_solar_samples) for i in range(self.num_solar_samples)]


        # get torch tensor with i * interval race distance values
        distances = torch.tensor([i * interval for i in range(num_points)], dtype=torch.float32)
        # save route points along route based on distances
        for i in range(num_points):
            self.route_points[i] = loader.get_position(distances[i])

    def debug_print(self):
        print("Route_points shape:", self.route_points.shape)

    def get_route_points(self, index: int | None = None) -> torch.Tensor:
        """
        Returns a tensor of shape (N, 3) or (, 3) where each row is [latitude, longitude, heading] for a point along the route.
        If index is provided, returns only that point.
        """
        if index is not None:
            try:
                return self.route_points[index]
            except IndexError:
                raise IndexError(f"Index {index} is out of bounds for route points with length {len(self.route_points)}") from None
        return self.route_points


    def get_route_solar_vecs(self, index: int | None = None) -> torch.Tensor:
        """
        Returns sun vectors (in the car's body frame) sampled across all time samples.

        If index is provided, returns shape (T, 3) for that single route point.
        Otherwise returns shape (N, T, 3) for all route points and all time samples.
        """
        if index is not None:
            try:
                point = self.route_points[index]
            except IndexError:
                raise IndexError(f"Index {index} is out of bounds for route points with length {len(self.route_points)}") from None
            lat, lon, heading = point
            solar_vecs = torch.zeros((len(self.times), 3), dtype=torch.float32)
            for j, dt in enumerate(self.times):
                solar_vecs[j] = get_solar_car_vec(lat, lon, dt, heading)
            return solar_vecs
        #else
        return get_solar_car_vecs_batch(self.route_points, self.times)
