import datetime

import pytz
import torch

from subprojects.wp3_solar_estimator.australia.routeloader import Routeloader
from subprojects.wp3_solar_estimator.australia.sun import get_solar_car_vec, get_solar_car_vecs_batch
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class RouteV2:
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
        num_points = config["simulation"]["num_points"]  # number of route points to sample
        interval = race_length / num_points  # in km, distance between each sampled point
        self.velocity = config["simulation"]["average_speed"]  # in m/s
        time_delta = datetime.timedelta(seconds=(interval * 1000) / self.velocity)
        self.route_points = torch.zeros((num_points, 3), dtype=torch.float32) # Shape (N, 3) tensor of [lat, lon, heading] for each point along the route

        self.times : list[datetime.datetime] = [] #array of timestamps (datetime format)


        #make timestamps
        start_date = config["sun_generation"]["start_date"]  # in DD/MM/YYYY, Australian time
        racing_window: list[float] = config["sun_generation"]["racing_window"]  # in hours, e.g. [8, 17.5] for 8:00 to 17:30
        racing_duration_hours = racing_window[1] - racing_window[0]
        timezone= config["general"]["timezone"]  # e.g. "ACST"
        self.start_time = datetime.datetime.strptime(start_date, "%d/%m/%Y") + datetime.timedelta(
            hours=int(racing_window[0]), minutes=int((racing_window[0] - int(racing_window[0])) * 60)
        )  # set to start of racing window on the start date
        tz = pytz.timezone(timezone)
        self.start_time = tz.localize(self.start_time)
        # account for act we can oly race during window
        race_days = (
            int((race_length * 1000) / self.velocity / 3600 / racing_duration_hours) + 1
        )  # number of days needed to complete
        self.end_time = self.start_time + datetime.timedelta(
            days=race_days, hours=racing_duration_hours
        )  # end time is the last day at the end of the racing window

        #set times to be from start_time to end_time with spacing of time_delta, but only include those within the racing window. Else, loop back to the next day until we have enough samples
        current_time = self.start_time
        for _ in range(num_points):
            # we still are in racing window, logic assume we dont't cross into next racing window within 1 step
            if  (current_time.hour + current_time.minute / 60) <= racing_window[1]:
                self.times.append(current_time)
                current_time += time_delta

            # if overshoot the racing window, loop back to the next day at the start of the racing window and add excess time here
            else:
                excess=(current_time + time_delta).hour + (current_time + time_delta).minute / 60 - racing_window[1]
                # loop to next day, set to start of racing window + excess time
                current_time = current_time + datetime.timedelta(days=1)
                #set to start of racing window that day
                current_time = current_time.replace(
                    hour=int(racing_window[0]),
                    minute=int((racing_window[0] - int(racing_window[0])) * 60),
                    second=0,
                    microsecond=0,
                )  # set seconds to 0
                current_time += datetime.timedelta(seconds=excess * 3600)  # add excess time in seconds

                #make next step
                self.times.append(current_time)
                current_time += time_delta

            if current_time > self.end_time:
                raise ValueError(f"Generated time {current_time} exceeds expected end time {self.end_time}. This likely means a bug with the number of points or average velocity calculations, overshooting the available data range")


        # get torch tensor with i * interval race distance values
        distances = torch.tensor([i * interval for i in range(num_points)], dtype=torch.float32)
        # save route points along route based on distances
        for i in range(num_points):
            self.route_points[i] = loader.get_position(distances[i])

        assert len(self.times) == num_points, f"Number of generated timestamps {len(self.times)} does not match number of route points {num_points}. This likely indicates a bug in the timestamp generation logic."
        assert len(self.route_points) == num_points, f"Number of generated route points {len(self.route_points)} does not match number of timestamps {num_points}. This likely indicates a bug in the route point generation logic."


    def debug_print(self):
        print("Route_points shape:", self.route_points.shape)
        print("Times length:", len(self.times))
        print("Start time:", self.start_time)
        print("End time:", self.end_time)

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
        Returns sun vectors (in the car's body frame) sampled across all route points.

        If index is provided, returns shape (, 3) for that single route point.
        Otherwise returns shape (N, 3) for all route points and their associatedd times.
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
        solar_vecs=get_solar_car_vecs_batch(self.route_points, self.times)
        #remove unnecessary time indexing dim since each route point has a unique timestamp now
        solar_vecs = solar_vecs.squeeze(1)  # shape (N, 3)
        return solar_vecs
