from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from astral import Observer
from astral.sun import azimuth, elevation


class WeatherCalculator:
    def __init__(self):
        self.rng = np.random.default_rng()

    def sun_vector(self, alt_deg, az_deg):
        alt, az = np.radians([alt_deg, az_deg])
        return np.array([
            np.cos(alt) * np.sin(az),
            np.cos(alt) * np.cos(az),
            np.sin(alt)
        ])

    def sun_finder(self, lat, lon, qtime):
        tz = ZoneInfo("Australia/Adelaide")

        dt = datetime.now(tz).replace(
            hour=qtime.hour(),
            minute=qtime.minute(),
            second=qtime.second(),
            microsecond=0,
        )

        obs = Observer(latitude=lat, longitude=lon)
        return elevation(obs, dt), azimuth(obs, dt)

    def compute_shadows(self, alt_deg, az_deg, hdg_deg=0):
        sun = self.sun_vector(alt_deg, az_deg - hdg_deg)
        ray_dir = sun

        _, ray_ids, _ = self.deck.multi_ray_trace(
            self.centers_flat,
            np.repeat(ray_dir[None], self.total_cells, axis=0),
            first_point=False
        )

        shadow = np.zeros(self.total_cells, dtype=bool)
        shadow[ray_ids] = True
        return shadow

    def compute_diffuse_shadows(self, alt_deg, az_deg, hdg_deg=0,
                        sun_angle_deg=0.53, n_samples=15):

        w = self.sun_vector(alt_deg, az_deg - hdg_deg)
        w /= np.linalg.norm(w)

        # Orthonormal basis
        u = np.cross(w, [0, 0, 1])
        if np.linalg.norm(u) < 1e-6:
            u = np.cross(w, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(w, u)

        # Sample solar disk
        sun_r = np.deg2rad(sun_angle_deg / 2)
        r = sun_r * np.sqrt(self.rng.uniform(size=n_samples))
        phi = 2 * np.pi * self.rng.uniform(size=n_samples)

        dirs = (w
                + (r * np.cos(phi))[:, None] * u
                + (r * np.sin(phi))[:, None] * v)
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]

        # Vectorized ray tracing
        origins = np.repeat(self.centers_flat, n_samples, axis=0)
        directions = np.tile(dirs, (self.total_cells, 1))

        _, ray_ids, _ = self.deck.multi_ray_trace(
            origins, directions, first_point=False
        )

        # Count hits per cell
        hit_counts = np.bincount(ray_ids // n_samples,
                                minlength=self.total_cells)

        return hit_counts / n_samples

    def compute_cos_incidence(self, alt_deg, az_deg, hdg_deg=0):
        sun = self.sun_vector(alt_deg, az_deg-hdg_deg)
        cos_inc = np.dot(self.normals_flat, sun)
        return np.clip(cos_inc, 0, 1)

    def compute_illumination(self, alt_deg, az_deg, hdg_deg=0):
        cos_inc = self.compute_cos_incidence(alt_deg, az_deg, hdg_deg)
        shadow = self.compute_shadows(alt_deg, az_deg, hdg_deg=hdg_deg)

        illum = cos_inc.copy()*(1-shadow)
        return illum

    def compute_min_illumination(self, alt_deg, az_deg, hdg_deg=0, area_fraction=0.05):
        illum = self.compute_illumination(alt_deg, az_deg, hdg_deg=hdg_deg)

        mins = []
        i = 0

        for n_cells, mesh in zip(self.n_cells, self.scells, strict=False):
            mesh_illum = illum[i:i+n_cells]
            areas = mesh.compute_cell_sizes()["Area"]

            order = np.argsort(mesh_illum)
            sorted_illum = mesh_illum[order]
            sorted_areas = areas[order]

            cumulative = np.cumsum(sorted_areas)
            threshold = area_fraction * areas.sum()
            idx = np.searchsorted(cumulative, threshold)

            mins.append(sorted_illum[min(idx, len(sorted_illum)-1)])
            i += n_cells

        return np.repeat(mins, self.n_cells)
