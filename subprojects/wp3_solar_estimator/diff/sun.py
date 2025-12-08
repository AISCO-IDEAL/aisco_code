import numpy as np
import pandas as pd
from pvlib import solarposition


def get_sun_direction(latitude, longitude, altitude, time_utc):
    #nserck
    """
    Calculated the vector of the sun direction [x,y,z]
    
    x: east
    y: north
    z: zenit (facing upwards!)

    Args:
        latitude (float): [Grad]
        longitude (float):  [Grad]
        altitude (float): [Meter] avobe sea leevel
        time_utc: UTC!

    Rückgabe:
    numpy.array: norm-vec (x, y, z)
    """

    # 1. Calculate Sun position
    solpos = solarposition.get_solarposition(
        time_utc,##times,
        latitude,
        longitude,
        altitude=altitude
    )

    
    azimuth = np.deg2rad(solpos['azimuth'].iloc[0]) # convert azimuth from degrees to radians
    elevation = np.deg2rad(solpos['elevation'].iloc[0]) 

    # 2. Umwandlung von Azimut und Elevation in einen kartesischen Vektor (x, y, z)
    zenith_angle = np.pi/2 - elevation
    
    x = np.sin(zenith_angle) * np.sin(azimuth)  # Osten
    y = np.sin(zenith_angle) * np.cos(azimuth)  # Norden
    z = np.cos(zenith_angle)                    # Zenit (oben)

    return np.array([x, y, z])