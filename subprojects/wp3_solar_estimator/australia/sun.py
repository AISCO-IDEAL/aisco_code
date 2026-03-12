"""
Solar position and sun-vector utilities.

All functions are pure (no side effects, no class state) and accept
standard-library :class:`datetime.datetime` objects rather than Qt types.

Coordinate conventions
-----------------------
World frame (intermediate only)
    East-North-Up (ENU): x = East, y = North, z = Up.
    Used internally; the ``sun_vector_world`` helper returns this frame.

Car body frame
    x = forward  (car's nose direction)
    y = left
    z = up

    The frame is obtained by rotating the ENU frame clockwise by
    *heading_deg* around the Up axis, so that +x aligns with the
    car's heading.  All public functions that include a heading
    return vectors in this ``[forward, left, up]`` convention.

Astral azimuth convention
    ``astral`` returns azimuth as degrees **clockwise from North** (0-360°),
    which is the compass-bearing convention used throughout this module.
"""

from __future__ import annotations

from datetime import datetime

import torch
from astral import Observer
from astral.sun import azimuth as _astral_azimuth
from astral.sun import elevation as _astral_elevation


def _to_float(x: float | torch.Tensor) -> float:
    """Convert a scalar tensor or float to a Python float."""
    return x.item() if isinstance(x, torch.Tensor) else float(x)


def sun_position(lat: float | torch.Tensor, lon: float | torch.Tensor, dt: datetime) -> tuple[float, float]:
    """Return the sun's position for a given observer and moment in time.

    Parameters
    ----------
    lat, lon:
        Observer position in decimal degrees.  May be plain floats or
        scalar :class:`torch.Tensor` values.
    dt:
        A **timezone-aware** :class:`datetime.datetime`.  Astral treats naive
        datetimes as UTC, so passing a naive local time will silently produce
        wrong solar angles.  Use ``pytz.timezone("Australia/Darwin").localize()``
        or ``pytz.timezone("Australia/Adelaide").localize()`` before calling.

    Returns
    -------
    (elevation_deg, azimuth_deg)
        *elevation_deg* - angle above the horizon (negative = below horizon).
        *azimuth_deg*   - bearing from North, clockwise (0-360°).

    Raises
    ------
    ValueError
        If *dt* is timezone-naive.
    """
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise ValueError(
            f"dt must be timezone-aware (got naive datetime {dt!r}). "
            "Astral interprets naive datetimes as UTC, which gives wrong solar angles for local time. "
            "Localize with e.g. pytz.timezone('Australia/Darwin').localize(dt) first."
        )
    obs = Observer(latitude=_to_float(lat), longitude=_to_float(lon))
    return float(_astral_elevation(obs, dt)), float(_astral_azimuth(obs, dt))


def sun_vector_world(elevation_deg: float | torch.Tensor, azimuth_deg: float | torch.Tensor) -> torch.Tensor:
    """Convert solar angles to a unit vector in the world ENU frame.

    This is an intermediate helper used by :func:`car_`.
    For most use-cases you want the car-body-frame version instead.

    Parameters
    ----------
    elevation_deg:
        Sun elevation above the horizon in degrees.
    azimuth_deg:
        Sun azimuth (bearing from North, clockwise) in degrees, as returned
        by ``astral``.

    Returns
    -------
    torch.Tensor, shape (3,)
        Unit vector ``[East, North, Up]``.
    """
    alt = torch.deg2rad(torch.as_tensor(elevation_deg, dtype=torch.float32))
    az = torch.deg2rad(torch.as_tensor(azimuth_deg, dtype=torch.float32))
    return torch.stack(
        [
            torch.cos(alt) * torch.sin(az),  # East
            torch.cos(alt) * torch.cos(az),  # North
            torch.sin(alt),  # Up
        ]
    )


def sun_data_to_car_vec(
    elevation_deg: float | torch.Tensor,
    azimuth_deg: float | torch.Tensor,
    heading_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Returns Sun unit vector in the car's body frame ``[forward, left, up]`` from astral data.

    The body frame is obtained by rotating the world ENU frame clockwise
    by *heading_deg* around the Up axis so that the car's nose (+x) aligns
    with the heading direction.

    Derivation
    ----------
    Let ``rel_az = azimuth - heading`` be the sun's bearing measured
    clockwise from the car's nose.  Projecting directly onto the body axes::

        F =  cos(alt) * cos(rel_az)   # forward
        L = -cos(alt) * sin(rel_az)   # left  (CW is right, so negate)
        U =  sin(alt)                 # up

    This is equivalent to the two-step ENU projection but avoids the
    intermediate world-frame vector.

    Parameters
    ----------
    elevation_deg:
        Sun elevation above the horizon in degrees.
    azimuth_deg:
        Sun azimuth (bearing from North, clockwise) in degrees, as returned
        by ``astral``.
    heading_deg:
        Car heading (bearing from North, clockwise) in degrees.

    Returns
    -------
    torch.Tensor, shape (3,)
        Unit vector ``[forward, left, up]`` in car body frame.
    """
    alt = torch.deg2rad(torch.as_tensor(elevation_deg, dtype=torch.float32))
    rel_az = torch.deg2rad(
        torch.as_tensor(azimuth_deg, dtype=torch.float32) - torch.as_tensor(heading_deg, dtype=torch.float32)
    )  # sun bearing relative to car nose (CW positive)
    return torch.stack(
        [
            torch.cos(alt) * torch.cos(rel_az),  # forward  (rel_az=0   → sun straight ahead)
            -torch.cos(alt) * torch.sin(rel_az),  # left     (rel_az>0 is CW/right, so negate)
            torch.sin(alt),  # up
        ]
    )


def get_solar_car_vec(
    lat: float | torch.Tensor,
    lon: float | torch.Tensor,
    dt: datetime,
    heading_deg: float | torch.Tensor,
) -> torch.Tensor:
    """Return the sun unit vector in the car body frame ``[forward, left, up]``.

    Convenience wrapper that calls :func:`sun_position` and
    :func:`sun_data_to_car_vec` in sequence.

    Parameters
    ----------
    lat, lon:
        Observer position in decimal degrees.  May be plain floats or
        scalar :class:`torch.Tensor` values.
    dt:
        A timezone-aware :class:`datetime.datetime`.  The caller is
        responsible for attaching the correct timezone (e.g. via
        ``zoneinfo.ZoneInfo``).
    heading_deg:
        Car heading (bearing from North, clockwise) in degrees.

    Returns
    -------
    torch.Tensor, shape (3,)
        Unit vector ``[forward, left, up]`` pointing toward the sun in the
        car body frame.  The z-component (up) equals ``sin(elevation)``; a
        negative value means the sun is below the horizon.
    """
    elevation_deg, azimuth_deg = sun_position(lat, lon, dt)
    return sun_data_to_car_vec(elevation_deg, azimuth_deg, heading_deg)


def get_solar_car_vecs_batch(
    route_points: torch.Tensor,
    times: list[datetime],
) -> torch.Tensor:
    """Return sun unit vectors for every route point and time sample.

    Parameters
    ----------
    route_points:
        Tensor of shape ``(N, 3)`` where each row is ``[lat, lon, heading_deg]``.
    times:
        List of ``T`` :class:`datetime.datetime` objects to sample.

    Returns
    -------
    torch.Tensor, shape ``(N, T, 3)``
        Sun unit vectors in the car body frame ``[forward, left, up]``
        for every combination of route point and time.
    """
    N = route_points.shape[0]
    T = len(times)
    solar_vecs = torch.zeros((N, T, 3), dtype=torch.float32)
    for i in range(N):
        lat, lon, heading = route_points[i]
        for j, dt in enumerate(times):
            elev_j, az_j = sun_position(lat, lon, dt)
            solar_vecs[i, j] = sun_data_to_car_vec(elev_j, az_j, heading)
    return solar_vecs


def get_solar_car_vecs_one_to_one(
    route_points: torch.Tensor,
    times: list[datetime],
) -> torch.Tensor:
    """Return sun unit vectors for a one-to-one mapping of route points and times.

    Parameters
    ----------
    route_points:
        Tensor of shape ``(N, 3)`` where each row is ``[lat, lon, heading_deg]``.
    times:
        List of ``N`` :class:`datetime.datetime` objects, one for each route point.

    Returns
    -------
    torch.Tensor, shape ``(N, 3)``
        Sun unit vectors in the car body frame ``[forward, left, up]``.
    """
    if route_points.shape[0] != len(times):
        raise ValueError(f"Number of route points ({route_points.shape[0]}) must match number of times ({len(times)})")
    lats = route_points[:, 0]
    lons = route_points[:, 1]
    headings = route_points[:, 2]
    solar_vecs_list = [
        get_solar_car_vec(lat, lon, dt, heading) for lat, lon, dt, heading in zip(lats, lons, times, headings, strict=False)
    ]
    return torch.stack(solar_vecs_list)
