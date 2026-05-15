from __future__ import annotations

import datetime
import math

import pytz
import torch

from subprojects.wp3_solar_estimator.australia.routeloader import Routeloader
from subprojects.wp3_solar_estimator.australia.sun import (
    get_solar_car_vecs_batch,
    get_solar_car_vecs_one_to_one,
)
from subprojects.wp4_aeroshell_optimizer import load_config


class RouteV3:
    """
    Unified route + solar vector source.

    mode="race"        — one solar vector per route point (RouteV2 behaviour), with the
                         day-transition excess bug fixed.
    mode="time_sweep"  — T uniformly-spaced timestamps across the racing window, evaluated
                         at all N route points; returns (N*T, 3) flattened (RouteV1 behaviour).
    """

    def __init__(self, cfg_path: str = "solar_configs/australia.yaml", mode: str = "race") -> None:
        if mode not in ("race", "time_sweep"):
            raise ValueError(f"mode must be 'race' or 'time_sweep', got {mode!r}")
        self._mode = mode
        self._solar_vecs_cache: torch.Tensor | None = None

        loader = Routeloader()
        config = load_config(cfg_name=cfg_path)

        # Shared config
        race_length: float = config["route_generation"]["race_distance"]  # km
        timezone: str = config["general"]["timezone"]
        start_date: str = config["sun_generation"]["start_date"]  # DD/MM/YYYY
        racing_window: list[float] = config["sun_generation"]["racing_window"]  # e.g. [8, 17.5]
        racing_duration_hours = racing_window[1] - racing_window[0]
        tz = pytz.timezone(timezone)

        if mode == "race":
            self.num_points: int = config["simulation"]["num_points"]
            velocity: float = config["simulation"]["average_speed"]  # m/s
            interval = race_length / self.num_points
            time_delta = datetime.timedelta(seconds=(interval * 1000) / velocity)

            self.start_time = datetime.datetime.strptime(start_date, "%d/%m/%Y") + datetime.timedelta(
                hours=int(racing_window[0]),
                minutes=int((racing_window[0] - int(racing_window[0])) * 60),
            )
            self.start_time = tz.localize(self.start_time)
            race_days = int((race_length * 1000) / velocity / 3600 / racing_duration_hours) + 1
            self.end_time = self.start_time + datetime.timedelta(days=race_days, hours=racing_duration_hours)

            self.times: list[datetime.datetime] = []
            current_time = self.start_time
            for _ in range(self.num_points):
                if (current_time.hour + current_time.minute / 60) <= racing_window[1]:
                    self.times.append(current_time)
                    current_time += time_delta
                else:
                    # Bug fix vs RouteV2 line 62: excess is relative to the current_time that
                    # overshot, not to (current_time + time_delta) which double-counts one step.
                    excess = current_time.hour + current_time.minute / 60 - racing_window[1]
                    current_time = current_time + datetime.timedelta(days=1)
                    current_time = current_time.replace(
                        hour=int(racing_window[0]),
                        minute=int((racing_window[0] - int(racing_window[0])) * 60),
                        second=0,
                        microsecond=0,
                    )
                    current_time += datetime.timedelta(seconds=excess * 3600)
                    self.times.append(current_time)
                    current_time += time_delta

                if current_time > self.end_time:
                    raise ValueError(f"Generated time {current_time} exceeds expected end time {self.end_time}.")

        else:  # time_sweep
            interval = config["route_generation"]["intervals"]
            self.num_points = int((race_length + 1) // interval)
            num_solar_samples: int = config["route_generation"]["solar_samples"]
            self.start_time = datetime.datetime.strptime(start_date, "%d/%m/%Y")
            self.start_time = tz.localize(self.start_time)
            self.times = [
                self.start_time
                + datetime.timedelta(hours=racing_window[0] + i * (racing_window[1] - racing_window[0]) / num_solar_samples)
                for i in range(num_solar_samples)
            ]

        distances = torch.tensor([i * interval for i in range(self.num_points)], dtype=torch.float32)
        self.route_points = torch.zeros((self.num_points, 3), dtype=torch.float32)
        for i in range(self.num_points):
            self.route_points[i] = loader.get_position(distances[i])

    def solar_vectors(self, subsample: int | None = None) -> torch.Tensor:
        """
        Returns (K, 3) normalised sun vectors in the car body frame.

        For mode="race": K = num_points (one vector per route point at its travel time).
        For mode="time_sweep": K = num_points * num_solar_samples (all (N,T,3) flattened).
        Lazy-computed and cached. subsample reduces K to n evenly-spaced indices.
        """
        if self._solar_vecs_cache is None:
            if self._mode == "race":
                self._solar_vecs_cache = get_solar_car_vecs_one_to_one(self.route_points, self.times)
            else:
                vecs_nt3 = get_solar_car_vecs_batch(self.route_points, self.times)
                self._solar_vecs_cache = vecs_nt3.view(-1, 3)

        if subsample is not None and subsample < len(self._solar_vecs_cache):
            indices = torch.linspace(0, len(self._solar_vecs_cache) - 1, subsample).long()
            return self._solar_vecs_cache[indices]

        return self._solar_vecs_cache

    # --- analysis helpers (operate on full cached vectors) ---

    def sun_elevations_deg(self) -> torch.Tensor:
        """(K,) elevation angles in degrees above horizon."""
        vecs = self.solar_vectors()
        return torch.asin(vecs[:, 2].clamp(-1, 1)) * 180 / math.pi

    def sun_azimuths_car_relative_deg(self) -> torch.Tensor:
        """(K,) azimuth angles in degrees relative to car forward direction."""
        vecs = self.solar_vectors()
        return torch.atan2(vecs[:, 1], vecs[:, 0]) * 180 / math.pi

    def elevation_stats(self) -> dict[str, float]:
        elev = self.sun_elevations_deg()
        return {
            "min": float(elev.min().item()),
            "max": float(elev.max().item()),
            "mean": float(elev.mean().item()),
            "std": float(elev.std().item()),
        }

    def azimuth_stats(self) -> dict[str, float]:
        az = self.sun_azimuths_car_relative_deg()
        return {
            "min": float(az.min().item()),
            "max": float(az.max().item()),
            "mean": float(az.mean().item()),
            "std": float(az.std().item()),
        }

    def below_horizon_count(self) -> int:
        return int((self.sun_elevations_deg() < 0).sum().item())

    def summary(self) -> None:
        vecs = self.solar_vectors()
        elev_stats = self.elevation_stats()
        az_stats = self.azimuth_stats()
        print(f"RouteV3 summary (mode={self._mode!r})")
        print(f"  route points: {self.num_points}, solar vectors: {len(vecs)}")
        print(
            f"  elevation (deg): min={elev_stats['min']:.1f}  max={elev_stats['max']:.1f}  "
            f"mean={elev_stats['mean']:.1f}  std={elev_stats['std']:.1f}"
        )
        print(
            f"  azimuth   (deg): min={az_stats['min']:.1f}  max={az_stats['max']:.1f}  "
            f"mean={az_stats['mean']:.1f}  std={az_stats['std']:.1f}"
        )
        print(f"  below horizon: {self.below_horizon_count()}")

    # --- backward-compat helpers ---

    def debug_print(self) -> None:
        print("Route_points shape:", self.route_points.shape)
        print("Times length:", len(self.times))

    def get_route_points(self, index: int | None = None) -> torch.Tensor:
        if index is not None:
            return self.route_points[index]
        return self.route_points
