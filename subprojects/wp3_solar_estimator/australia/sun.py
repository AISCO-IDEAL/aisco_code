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

import numpy as np
from astral import Observer
from astral.sun import azimuth as _astral_azimuth
from astral.sun import elevation as _astral_elevation


def sun_position(lat: float, lon: float, dt: datetime) -> tuple[float, float]:
    """Return the sun's position for a given observer and moment in time.

    Parameters
    ----------
    lat, lon:
        Observer position in decimal degrees.
    dt:
        A timezone-aware :class:`datetime.datetime`.  The caller is
        responsible for attaching the correct timezone (e.g. via
        ``zoneinfo.ZoneInfo``).

    Returns
    -------
    (elevation_deg, azimuth_deg)
        *elevation_deg* - angle above the horizon (negative = below horizon).
        *azimuth_deg*   - bearing from North, clockwise (0-360°).
    """
    obs = Observer(latitude=lat, longitude=lon)
    return float(_astral_elevation(obs, dt)), float(_astral_azimuth(obs, dt))


def sun_vector_world(elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Convert solar angles to a unit vector in the world ENU frame.

    This is an intermediate helper used by :func:`sun_vector_car_frame`.
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
    np.ndarray, shape (3,)
        Unit vector ``[East, North, Up]``.
    """
    alt = np.radians(elevation_deg)
    az = np.radians(azimuth_deg)
    return np.array([
        np.cos(alt) * np.sin(az),   # East
        np.cos(alt) * np.cos(az),   # North
        np.sin(alt),                 # Up
    ])


def sun_vector_car_frame(
    elevation_deg: float,
    azimuth_deg: float,
    heading_deg: float,
) -> np.ndarray:
    """Sun unit vector in the car's body frame ``[forward, left, up]``.

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
    np.ndarray, shape (3,)
        Unit vector ``[forward, left, up]`` in car body frame.
    """
    alt = np.radians(elevation_deg)
    rel_az = np.radians(azimuth_deg - heading_deg)  # sun bearing relative to car nose (CW positive)
    return np.array([
        np.cos(alt) * np.cos(rel_az),    # forward  (rel_az=0   → sun straight ahead)
        -np.cos(alt) * np.sin(rel_az),   # left     (rel_az>0 is CW/right, so negate)
        np.sin(alt),                      # up
    ])


def get_solar_car_vec(lat: float, lon: float, dt: datetime, heading_deg: float) -> np.ndarray:
    """Return the sun unit vector in the car body frame ``[forward, left, up]``.

    Convenience wrapper that calls :func:`sun_position` and
    :func:`sun_vector_car_frame` in sequence.

    Parameters
    ----------
    lat, lon:
        Observer position in decimal degrees.
    dt:
        A timezone-aware :class:`datetime.datetime`.  The caller is
        responsible for attaching the correct timezone (e.g. via
        ``zoneinfo.ZoneInfo``).
    heading_deg:
        Car heading (bearing from North, clockwise) in degrees.

    Returns
    -------
    np.ndarray, shape (3,)
        Unit vector ``[forward, left, up]`` pointing toward the sun in the
        car body frame.  The z-component (up) equals ``sin(elevation)``; a
        negative value means the sun is below the horizon.
    """

    elevation_deg, azimuth_deg = sun_position(lat, lon, dt)
    sun_vec_car = sun_vector_car_frame(elevation_deg, azimuth_deg, heading_deg)
    return sun_vec_car
