from dataclasses import dataclass
import cadquery as cq
from ocp_vscode import *
from .params import *

# --- Wheels class: ---
class Wheels:
    def __init__(self, params: WheelParams = WheelParams()):
        # -- Parameters: -- 
        self.params = params
        wheel_y_offset = self.params.track/2
        front_wheels_x_pos = self.params.base/2
        wheelpoints_front = [(0,0,wheel_y_offset), (0,0, -wheel_y_offset)]

        # -- building --
        front_wheels = cq.Workplane('XZ', origin= (front_wheels_x_pos, 0, self.params.radius)).pushPoints(wheelpoints_front).cylinder(self.params.width, self.params.radius)

        back_wheel = cq.Workplane('XZ', origin= (-front_wheels_x_pos,0, self.params.radius)).cylinder(self.params.width, self.params.radius)

        self.model = front_wheels.union(back_wheel)

#w = Wheels()

#set_port(3939)
#show_object(w.model)
