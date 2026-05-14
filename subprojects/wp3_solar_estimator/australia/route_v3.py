"""RouteV3: merged RouteV1 + RouteV2 with bug fixes, lazy caching, and analysis helpers."""

from __future__ import annotations

import datetime
import math
from typing import Literal

import pytz
import torch

from subprojects.wp3_solar_estimator.australia.routeloader import Routeloader
from subprojects.wp3_solar_estimator.australia.sun import get_solar_car_vec, get_solar_car_vecs_batch, get_solar_car_vecs_one_to_one
from subprojects.wp4_aeroshell_optimizer import load_config


class RouteV3:
    def __init__(
        self,
        cfg_path: str = "solar_configs/australia.yaml",
        mode: Literal["race", "time_sweep"] = "race",
    ) -> None:
        """
        mode="race"       → one-to-one time/point mapping (V2 behaviour, physically correct).
                            Each route point is assigned the timestamp when the car arrives there.
        mode="time_sweep" → T fixed times across the racing window applied to all N points (V1 behaviour).
                            Useful as a time-of-day survey; not a physical race simulation.
        """
        self.mode = mode
        self._solar_vectors_cache: torch.Tensor | None = None

        loader = Routeloader()
        config = load_config(cfg_name=cfg_path)

        race_length: float = config["route_generation"]["race_distance"]  # km
        timezone: str = config["general"]["timezone"]
        start_date_str: str = config["sun_generation"]["start_date"]
        racing_window: list[float] = config["sun_generation"]["racing_window"]
        racing_duration_hours = racing_window[1] - racing_window[0]
        tz = pytz.timezone(timezone)

        start_dt_naive = datetime.datetime.strptime(start_date_str, "%d/%m/%Y") + datetime.timedelta(
            hours=int(racing_window[0]),
            minutes=int((racing_window[0] - int(racing_window[0])) * 60),
        )
        self.start_time = tz.localize(start_dt_naive)

        if mode == "race":
            self.num_points: int = config["simulation"]["num_points"]
            velocity: float = config["simulation"]["average_speed"]  # m/s
            interval = race_length / self.num_points  # km
            time_delta = datetime.timedelta(seconds=(interval * 1000) / velocity)

            race_days = int((race_length * 1000) / velocity / 3600 / racing_duration_hours) + 1
            self.end_time = self.start_time + datetime.timedelta(days=race_days, hours=racing_duration_hours)

            self.times: list[datetime.datetime] = []
            current_time = self.start_time
            for _ in range(self.num_points):
                if (current_time.hour + current_time.minute / 60) <= racing_window[1]:
                    self.times.append(current_time)
                    current_time += time_delta
                else:
                    # Fix for the V2 excess bug: compute excess from current_time only, not current_time + time_delta
                    excess_hours = current_time.hour + current_time.minute / 60 - racing_window[1]
                    current_time = current_time + datetime.timedelta(days=1)
                    current_time = current_time.replace(
                        hour=int(racing_window[0]),
                        minute=int((racing_window[0] - int(racing_window[0])) * 60),
                        second=0,
                        microsecond=0,
                    )
                    current_time += datetime.timedelta(seconds=excess_hours * 3600)
                    self.times.append(current_time)
                    current_time += time_delta

                if current_time > self.end_time:
                    raise ValueError(
                        f"Generated time {current_time} exceeds expected end time {self.end_time}. "
                        "This likely indicates a bug in timestamp generation."
                    )

            distances = torch.tensor([i * interval for i in range(self.num_points)], dtype=torch.float32)

        else:  # time_sweep
            interval_km: float = config["route_generation"]["intervals"]  # km
            self.num_points = int((race_length + 1) // interval_km)
            self.num_solar_samples: int = config["sun_generation"]["solar_samples"]
            self.times = [
                self.start_time + datetime.timedelta(
                    hours=racing_window[0] + i * racing_duration_hours / self.num_solar_samples
                )
                for i in range(self.num_solar_samples)
            ]
            distances = torch.tensor([i * interval_km for i in range(self.num_points)], dtype=torch.float32)

        self.route_points = torch.zeros((self.num_points, 3), dtype=torch.float32)
        for i in range(self.num_points):
            self.route_points[i] = loader.get_position(distances[i])

    def solar_vectors(self, subsample: int | None = None) -> torch.Tensor:
        """
        Returns (K, 3) solar vectors in car body frame [forward, left, up].

        mode="race":       K = num_points before subsampling
        mode="time_sweep": K = num_points * num_solar_samples (flattened) before subsampling

        Vectors are computed lazily on first call and cached. Subsampling is done by
        evenly-spaced index selection and does not invalidate the cache.
        """
        if self._solar_vectors_cache is None:
            if self.mode == "race":
                self._solar_vectors_cache = get_solar_car_vecs_one_to_one(self.route_points, self.times)
            else:
                full = get_solar_car_vecs_batch(self.route_points, self.times)  # (N, T, 3)
                N, T, _ = full.shape
                self._solar_vectors_cache = full.reshape(N * T, 3)

        vecs = self._solar_vectors_cache
        if subsample is not None and subsample < vecs.shape[0]:
            idx = torch.linspace(0, vecs.shape[0] - 1, subsample).long()
            return vecs[idx]
        return vecs

    def sun_elevations_deg(self) -> torch.Tensor:
        """(K,) sun elevation angles in degrees, computed from the up component (arcsin(v_up))."""
        vecs = self.solar_vectors()
        return torch.rad2deg(torch.asin(vecs[:, 2].clamp(-1.0, 1.0)))

    def sun_azimuths_car_relative_deg(self) -> torch.Tensor:
        """(K,) azimuth of the sun relative to the car heading in degrees (arctan2(v_left, v_forward))."""
        vecs = self.solar_vectors()
        return torch.rad2deg(torch.atan2(vecs[:, 1], vecs[:, 0]))

    def elevation_stats(self) -> dict[str, float]:
        """min/max/mean/std of sun elevation across all K vectors."""
        elevs = self.sun_elevations_deg()
        return {
            "min": float(elevs.min().item()),
            "max": float(elevs.max().item()),
            "mean": float(elevs.mean().item()),
            "std": float(elevs.std().item()),
        }

    def azimuth_stats(self) -> dict[str, float]:
        """min/max/mean/std of car-relative azimuth across all K vectors."""
        azs = self.sun_azimuths_car_relative_deg()
        return {
            "min": float(azs.min().item()),
            "max": float(azs.max().item()),
            "mean": float(azs.mean().item()),
            "std": float(azs.std().item()),
        }

    def below_horizon_count(self) -> int:
        """Number of solar vectors with elevation below 0° (sun below horizon)."""
        return int((self.sun_elevations_deg() < 0).sum().item())

    def summary(self) -> None:
        """Print a human-readable stats block."""
        vecs = self.solar_vectors()
        K = vecs.shape[0]
        elev_s = self.elevation_stats()
        az_s = self.azimuth_stats()
        bh = self.below_horizon_count()
        print(f"RouteV3 summary (mode={self.mode})")
        print(f"  K={K} solar vectors")
        print(f"  Elevation (deg): min={elev_s['min']:.1f}  max={elev_s['max']:.1f}  mean={elev_s['mean']:.1f}  std={elev_s['std']:.1f}")
        print(f"  Azimuth   (deg): min={az_s['min']:.1f}  max={az_s['max']:.1f}  mean={az_s['mean']:.1f}  std={az_s['std']:.1f}")
        print(f"  Below horizon: {bh}/{K}")

    def debug_print(self) -> None:
        """Backward-compat debug print matching RouteV1/V2 interface."""
        print("Route_points shape:", self.route_points.shape)
        print("Times length:", len(self.times))
        print("Start time:", self.start_time)
        if self.mode == "race":
            print("End time:", self.end_time)
