from dataclasses import dataclass
import cadquery as cq
from ocp_vscode import *
from .seat import *
from .params import *

# -- Chassis class --
class Chassis:
   def __init__(
        self, 
        chassis_params: ChassisParams = ChassisParams(),
        seat_params: SeatParams = SeatParams(),
        wheel_params: WheelParams = WheelParams(),
        seat: Seat = Seat()
    ):
      # -- Parameters: --
      self.params = chassis_params
      self.seat_params = seat_params
      self.wheel_params = wheel_params
      self.seat = seat

      # --- model building: ---

      # -- Points rear part --
      # we define (0,0) as the center in the front wheel axis.
      
      A = (0,self.wheel_params.track/2) #to the wheel
      B = (A[0]- self.wheel_params.radius, self.wheel_params.track/2) #along the wheel to the base.
      C = (-self.wheel_params.base/2 + self.seat_params.position + seat.critical_point[0], seat.critical_point[1]) #to the seat
      D = (-self.wheel_params.base, self.wheel_params.width) #some addtitonal with, a good point to add further margin for suspention..
      E = (D[0],0) #to the rear wheel

    
      # --- profile of the rear part ---
      profile = (
      cq.Workplane("XY")
      .lineTo(A[0],A[1])
      .lineTo(B[0],B[1])
      .lineTo(C[0], C[1])
      .lineTo(D[0],D[1])
      .lineTo(E[0], E[1])         
      .close()
      )

      # --- Points nose ---
      N=(0, self.wheel_params.track/2)
      O=(self.wheel_params.radius, N[1])
      P=(self.wheel_params.radius+ self.params.nose,self.params.nose_width/2)
      Q=(P[0],0)

      # --- Nose profile ---
      nose_profile = (
      cq.Workplane("XY")
      .lineTo(N[0],N[1])
      .lineTo(O[0],O[1])
      .lineTo(P[0],P[1])
      .lineTo(Q[0],Q[1])        
      .close()
      )

      # --- create symmetrical 3D parts ---
      rear_side1 = profile.extrude(self.params.height, both=False)
      rear_side2 = rear_side1.mirror(mirrorPlane="XZ")
      rear_part = rear_side1.union(rear_side2)

      nose_side1 = nose_profile.extrude(self.params.height, both=False)
      nose_side2 =nose_side1.mirror(mirrorPlane="XZ")
      nose=nose_side1.union(nose_side2)
      chassis = rear_part.union(nose)
    
       # --- set center accordingly to convention (0,0) at wheel_base-center.
      chassis_t_vec= -rear_part.val().BoundingBox().center
      self.model = chassis.translate(chassis_t_vec)

#ch = Chassis()

#set_port(3939)
#show_object(ch.model)
      