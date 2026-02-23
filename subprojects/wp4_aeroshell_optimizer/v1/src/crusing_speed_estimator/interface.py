import numpy as np
from scipy.optimize import fsolve
from aero import get_drag
from solar import get_solar_perf
from mass import get_mass

def energy_balance(speed, m, rho, c_d, a, g, h, susp, braking, electrical, dist_travelled, solar_power):
    """
    Convention:
        - Positive forces/power: contribute to movement (e.g. solar power, potential energy gain)
        - Negative forces/power: resist movement (e.g. drag, rolling friction, etc.)
    """
    # 1. Protect against non-physical solver guesses
    if speed <= 0:
        return 1e6  # Large penalty for zero/negative speed

    # 2. Speed-dependent variables
    c_rolling_friction = (1.28667 + 0.0064*speed + 0.0000533333*speed**2) * 10**(-3)
    f_d = 0.5 * rho * speed**2 * c_d * a
    
    # Force magnitudes 
    f_rolling = c_rolling_friction * m * g
    f_susp = abs(susp)
    f_braking = abs(braking)
    
    # Power magnitudes
    p_elec_loss = abs(electrical)
    race_duration = dist_travelled / speed

    # 3. Energy Balance: Total Energy In - Total Energy Out
    e_in = solar_power * race_duration
    
    # Potential energy:
    e_pot_gain = m * g * h 
    
    e_out = (f_d + f_susp + f_braking + f_rolling) * dist_travelled + (p_elec_loss * race_duration)

    # 4. Kinetic Energy Check
    # Final Energy available for movement
    e_kin_avail = e_in + e_pot_gain - e_out
    
    # We want to find where the car's actual kinetic energy matches what's available
    e_kin_actual = 0.5 * m * speed**2
    
    return e_kin_actual - e_kin_avail

def estimate_cruising_speed():
    # Constants
    m = get_mass()  # kg
    rho = 1.212     # kg/m³
    c_d = 1.5       # -
    a = 10          # m²
    g = 9.8065      # m/s²
    h = 26          # m
    dist_travelled = 3030*10**(3)  # m

    # Losses
    susp = -14      # N
    braking = -20   # N
    electrical = -10 -5 -20 -3  # W

    # Solar power
    solar_power = get_solar_perf()  # W

    # Initial guess
    speed_initial_guess = 25  # m/s

    # Solve for speed
    speed_solution = fsolve(
        energy_balance,
        speed_initial_guess,
        args=(m, rho, c_d, a, g, h, susp, braking, electrical, dist_travelled, solar_power)
    )[0]

    return speed_solution * 3.6  # km/h