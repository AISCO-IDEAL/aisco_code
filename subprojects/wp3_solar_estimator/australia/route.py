import torch

from subprojects.wp3_solar_estimator.australia.routeloader import Routeloader
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


class Route:
    def __init__(self):
        """
        Loads the route points from the CSV file and stores them in a tensor for efficient access.
        Each point (saved as self.route_points) is represented as an array of (latitude, longitude, heading).
        Use get_route_points() to retrieve the full tensor or a specific point by index.
        """
        loader = Routeloader()
        config = load_config(cfg_name="australia.yaml")  # fix hardcode sometime
        self.route_points: torch.Tensor  # Shape (N, 3) tensor of [lat, lon, heading] for each point along the route
        # TODO figure out if theta in csv is heading
        race_length = config["route_generation"]["race_distance"]  # in km
        interval = config["route_generation"]["intervals"]  # in km
        # round up race length and divide by intervals to get number of points
        num_points = int(race_length + 1 // interval)
        # get torch tensor with i * interval race distance values
        distances = torch.tensor([i * interval for i in range(num_points)], dtype=torch.float32)
        self.route_points = torch.zeros((num_points, 3), dtype=torch.float32)

        for i in range(num_points):
            self.route_points[i] = loader.get_position(distances[i])

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
