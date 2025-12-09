import numpy as np
import pandas as pd
from pvlib import solarposition


def get_sun_direction(latitude, longitude, altitude, time_utc):
    # nserck
    """
    Convert geographic coordinates plus UTC timestamp to a sun direction vector.

    The coordinate system is right-handed with:
        x → East, y → North, z → Zenith (upwards).

    Args:
        latitude (float): Degrees North.
        longitude (float): Degrees East.
        altitude (float): Elevation above sea level in metres.
        time_utc (datetime | pandas.Timestamp): Time in UTC.

    Returns:
        np.ndarray: Normalised vector `[x, y, z]` pointing from the car to the sun.
    """

    # 1. Calculate Sun position
    solpos = solarposition.get_solarposition(
        time_utc,  ##times,
        latitude,
        longitude,
        altitude=altitude,
    )

    azimuth = np.deg2rad(
        solpos["azimuth"].iloc[0]
    )  # convert azimuth from degrees to radians
    elevation = np.deg2rad(solpos["elevation"].iloc[0])

    # 2. Umwandlung von Azimut und Elevation in einen kartesischen Vektor (x, y, z)
    zenith_angle = np.pi / 2 - elevation

    x = np.sin(zenith_angle) * np.sin(azimuth)  # Osten
    y = np.sin(zenith_angle) * np.cos(azimuth)  # Norden
    z = np.cos(zenith_angle)  # Zenit (oben)

    return np.array([x, y, z])
