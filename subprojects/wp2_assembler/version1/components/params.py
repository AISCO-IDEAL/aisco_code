from dataclasses import dataclass

#-- Parameters of the class Seat --
@dataclass
class SeatParams:
    width: float = 450.0 #general width of the seat
    radius_top: float = 835.0 # radius used to build the backrest
    plus_angle: float = 45.0 #angle of the backrest forwards
    minus_angle: float = 27.0 # angle of the backrest backwards

    Lr_l: float = 1000.0 # length of the legroom
    Lr_h: float = 350.0 # height of the legroom

    width_hr: float = 280.0 #Width of the headroom
    height_hr: float = 300.0 #height of the headroom

    position: float = 1000.0 # X-axis distance relative to the middle of the wheelbase

    color: str = "red"

# --- Parameters of the class Wheels ---
@dataclass
class WheelParams:
    radius: float = 225.0
    width: float = 75.0
    base: float = 2000.0
    track: float = 1000.0

    color: str = "black"

# --- Parameters of the class Chassis ---
@dataclass
class ChassisParams:
   nose: float = 500.0
   nose_width: float = 400.0
   hover: float = 140.0
   height: float = 150.0

   color: str = "blue"

