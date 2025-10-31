from components.params import *
from components.seat import Seat
import math

def chassis_constraints(chassis: ChassisParams=ChassisParams, seat: SeatParams=SeatParams, wheels: WheelParams=WheelParams) -> dict:
    return {
        #implement chassis contraints
    }

def seat_constraints(chassis: ChassisParams=ChassisParams, seat: SeatParams=SeatParams, wheels: WheelParams=WheelParams) -> dict:
    #-- calculations --
    s = Seat()
    minus_angle_rad = math.radians(seat.minus_angle)
    in_nose = 0.0

    if chassis.nose_width < seat.width:
        x_1 = (wheels.track - chassis.nose_width)/2
        y_1 = chassis.nose
        x_2 = (seat.width - chassis.nose_width)/2
        in_nose = (-y_1/x_1)*x_2 + y_1
    else:
        in_nose = chassis.nose

    dp_fwd = round(wheels.base/2 + in_nose)
    dp_bwd = round(wheels.base/2 - (s.s_length/2 - math.sin(minus_angle_rad)*seat.radius_top + wheels.radius/math.cos(minus_angle_rad)+math.tan(minus_angle_rad)*chassis.hover))
    
    return {
        'position': (-dp_bwd, dp_fwd)
    }

def wheels_constraints(chassis: ChassisParams=ChassisParams, seat: SeatParams=SeatParams, wheels: WheelParams=WheelParams) -> dict:
    return {
        #implement seat constraints
    }