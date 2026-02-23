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
    c_rolling_friction = (1.28667 + 0.0064*speed + 0.0000533333*speed**2) * 10**(-3) # TODO: document formula from aCe
    f_d = 0.5 * rho * speed**2 * c_d * a # TODO: prove feasibility of formula
    
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
    e_pot_gain = m * g * h # TODO: prove feasibility of formula
    
    e_out = (f_d + f_susp + f_braking + f_rolling) * dist_travelled + (p_elec_loss * race_duration)

    # 4. Kinetic Energy Check
    # Final Energy available for movement
    e_kin_avail = e_in + e_pot_gain - e_out
    
    # We want to find where the car's actual kinetic energy matches what's available
    e_kin_actual = 0.5 * m * speed**2 # TODO: prove feasibility of formula
    
    return e_kin_actual - e_kin_avail

def estimate_cruising_speed():
    # Constants
    m = get_mass()  # kg TODO: implement mass function to return actual mass of the car
    rho = 1.212     # kg/m³ TODO:prove correctness
    c_d = 1.5       # - TODO: get actual cd
    a = 10          # m² TODO: get actual frontal area
    g = 9.8065      # m/s² TODO: get true gravity
    h = 26          # m TODO: get actual elevation gain of the course
    dist_travelled = 3030*10**(3)  # m TODO: get actual distance of the course

    # Losses
    susp = -14      # N TODO: document measurement from suspension by Markus Strebel
    braking = -20   # N TODO: get actual Fbraking
    electrical = -10 -5 -20 -3  # W TODO: get actual electrical losses (battery, cables, resistors, motor, low voltage system, etc.)

    # Solar power
    solar_power = get_solar_perf()  # W TODO: implement get_solar_perf()

    # Initial guess
    speed_initial_guess = 25  # m/s TODO: document why 25

    # Solve for speed
    speed_solution = fsolve(
        energy_balance,
        speed_initial_guess,
        args=(m, rho, c_d, a, g, h, susp, braking, electrical, dist_travelled, solar_power)
    )[0]

    return speed_solution * 3.6  # km/h