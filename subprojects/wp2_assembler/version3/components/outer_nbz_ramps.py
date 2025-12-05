from subprojects.wp2_assembler.version3.components.nbz import call_assembly
import cadquery as cq
import math
from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation

def outer_nbz_ramps(assembly_id):
    #inner, outer = call_assembly(assembly_id)

    
    back_wheel_center = (-1000,278)
    front_wheel_center = (1000,278)
    outer_xmax = 2000
    outer_xmin = -2000
    outer_width = 100

    approach_angle = math.radians(get_regulation("no_build_zones.wheel_based_rules.approach_angle"))
    departure_angle = math.radians(get_regulation("no_build_zones.wheel_based_rules.departure_angle"))
    wheel_radius = (get_regulation("wheel.margin_radial") + get_regulation("wheel.tire_diameter"))/2

    print(departure_angle,approach_angle, wheel_radius)

    A = (front_wheel_center[0] +math.sin(approach_angle)*wheel_radius , front_wheel_center[1] - math.cos(approach_angle)*wheel_radius)
    B = (outer_xmax , A[1] + (abs(outer_xmax - front_wheel_center[0]))*math.tan(approach_angle))
    C = (outer_xmax, front_wheel_center[1]-wheel_radius)
    D = (front_wheel_center[0], front_wheel_center[1]-wheel_radius)

    Front = ( 
        cq.Workplane("XZ")
        .moveTo(*A)                     #Start at the contact point from the approach angle and the wheel
        .lineTo(*B)                     #Line to the end of the bounding box
        .lineTo(*C)                     #Line to the ground
        .lineTo(*D)                     #Line to the contact between the ground and the wheel
        .radiusArc(A,-wheel_radius)     #Wheel arc to the start point
        .close()
        .extrude(outer_width)
    )

    A = (back_wheel_center[0] - math.sin(departure_angle)*wheel_radius , back_wheel_center[1] - math.cos(departure_angle)*wheel_radius)
    B = (outer_xmin , A[1] + (abs(outer_xmin - back_wheel_center[0]))*math.tan(departure_angle))
    C = (outer_xmin, back_wheel_center[1]-wheel_radius)
    D = (back_wheel_center[0], back_wheel_center[1]-wheel_radius)

    Rear = ( 
        cq.Workplane("XZ")
        .moveTo(*A)                     #Start at the contact point from the approach angle and the wheel
        .lineTo(*B)                     #Line to the end of the bounding box
        .lineTo(*C)                     #Line to the ground
        .lineTo(*D)                     #Line to the contact between the ground and the wheel
        .radiusArc(A,wheel_radius)      #Wheel arc to the start point
        .close()
        .extrude(outer_width)
    )
    
    return Front,Rear



if __name__ == "__main__":
    from ocp_vscode import show
    demo_id = 30
    Front, Rear = outer_nbz_ramps(demo_id)
    show (Front, Rear)
