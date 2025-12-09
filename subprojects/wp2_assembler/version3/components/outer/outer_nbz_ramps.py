import cadquery as cq
import math
from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation

def outer_nbz_ramps(inner, outer) -> tuple[cq.Workplane, cq.Workplane]:
    """
    Returns the approach and departure ramps
    
    :param assembly_id: id of assembly
    :return: tuple with both ramps
    :rtype: tuple with two workplanes
    """

    back_wheel_center = (inner["wheel_1"].loc.toTuple()[0][0], inner["wheel_1"].loc.toTuple()[0][2])
    front_wheel_center = (inner["wheel_2"].loc.toTuple()[0][0], inner["wheel_2"].loc.toTuple()[0][2])
    outer_xmax = outer.BoundingBox().xmax
    outer_xmin = outer.BoundingBox().xmin

    approach_angle = math.radians(get_regulation("no_build_zones.wheel_based_rules.approach_angle"))
    departure_angle = math.radians(get_regulation("no_build_zones.wheel_based_rules.departure_angle"))
    wheel_radius = (get_regulation("wheel.margin_radial") + get_regulation("wheel.tire_diameter"))/2

    A = (front_wheel_center[0] +math.sin(approach_angle)*wheel_radius , front_wheel_center[1] - math.cos(approach_angle)*wheel_radius)
    B = (outer_xmax , A[1] + (abs(outer_xmax - front_wheel_center[0]))*math.tan(approach_angle))
    C = (outer_xmax, front_wheel_center[1]-wheel_radius)
    D = (front_wheel_center[0], front_wheel_center[1]-wheel_radius)

    Front = ( 
        cq.Workplane("XZ", origin = (0,outer.BoundingBox().ymax,0))
        .moveTo(*A)                     #Start at the contact point from the approach angle and the wheel
        .lineTo(*B)                     #Line to the end of the bounding box
        .lineTo(*C)                     #Line to the ground
        .lineTo(*D)                     #Line to the contact between the ground and the wheel
        .radiusArc(A,-wheel_radius)     #Wheel arc to the start point
        .close()
        .extrude(outer.BoundingBox().ylen)
    )

    A = (back_wheel_center[0] - math.sin(departure_angle)*wheel_radius , back_wheel_center[1] - math.cos(departure_angle)*wheel_radius)
    B = (outer_xmin , A[1] + (abs(outer_xmin - back_wheel_center[0]))*math.tan(departure_angle))
    C = (outer_xmin, back_wheel_center[1]-wheel_radius)
    D = (back_wheel_center[0], back_wheel_center[1]-wheel_radius)

    Rear = ( 
        cq.Workplane("XZ", origin = (0,outer.BoundingBox().ymax,0))
        .moveTo(*A)                     #Start at the contact point from the approach angle and the wheel
        .lineTo(*B)                     #Line to the end of the bounding box
        .lineTo(*C)                     #Line to the ground
        .lineTo(*D)                     #Line to the contact between the ground and the wheel
        .radiusArc(A,wheel_radius)      #Wheel arc to the start point
        .close()
        .extrude(outer.BoundingBox().ylen)
    )
    
    result = outer.cut(Front.val(), Rear.val()).Solids()[0]

    return result

if __name__ == "__main__":
    '''
    from ocp_vscode import show
    result = outer_nbz_ramps(demo_id)
    inner, _, _ = call_assembly(demo_id)
    show (result, inner)
    print(type(result))
    '''
