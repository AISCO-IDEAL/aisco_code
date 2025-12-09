import math
import random
from pathlib import Path

import cadquery as cq
import yaml

from subprojects.wp2_assembler.version3.components.inner.base_triangle import Triangle
from subprojects.wp2_assembler.version3.components.inner.seat import Seat
from subprojects.wp2_assembler.version3.components.inner.wheels import Wheels
from subprojects.wp2_assembler.version3.components.outer.outer_box import outer_box2
from subprojects.wp2_assembler.version3.components.vision import tracker_box, vision_box
from subprojects.wp2_assembler.version3.utils.distributions import uniform_distribution
from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation
from subprojects.wp2_assembler.version3.components.outer.outer_nbz_ramps import outer_nbz_ramps


class WheelbaseTooSmallError(Exception):
    def __init__(self, dim: str):
        super().__init__(f"Wheelbase {dim} too small")


def inner_nbz_sym_rand():
    """
    New implementation of innerNBZ for version 2
    Args: constraints from regulations
    Returns: cq.shape.solid of inner NBZ shape
    """

    # draw triangle
    triangle = Triangle().build()
    points = Triangle().get_points()

    # add wheels to the corners of the triangle
    wheels = Wheels()
    wheel = wheels.create_wheel()

    seat_class = Seat()
    seat = seat_class.create_seat()

    assembly = (
        cq.Assembly(triangle, name="triangle")
        # move to 0,0,0
        .add(
            wheel,
            loc=cq.Location(points[0].x, points[0].y, points[0].z),
            name="wheel_1",
        )
        .add(
            wheel,
            loc=cq.Location(points[1].x, points[1].y, points[1].z),
            name="wheel_2",
        )
        .add(
            wheel,
            loc=cq.Location(points[2].x, points[2].y, points[2].z),
            name="wheel_3",
        )
        .add(
            seat,
            loc=cq.Location(random.randint(0, float(points[2].x)), 0, -100),
            name="Seat",
        )
    )

    return assembly


def inner_nbz_sym(
    wheel_base_width=1000,
    wheel_base_length=2500,
    tyre_diameter=557,
    seat_position_x=400,
    seat_position_z=-100,
):
    """
    Deterministic implementation of innerNBZ for version 2
    Args: parameters for the inner NBZ
        wheel_base_width: width of the wheel base in mm
        wheel_base_length: length of the wheel base in mm
        tyre_diameter: diameter of the tires in mm
        seat_position: position of the seat in mm
        shift: shift of the whole assembly along x-axis in mm
    Returns: cq.shape.solid of inner NBZ
    """
    """
    Deterministic implementation of innerNBZ for version 2
    Args: parameters for the inner NBZ
        wheel_base_width: width of the wheel base in mm
        wheel_base_length: length of the wheel base in mm
        tyre_diameter: diameter of the tires in mm
        seat_position: position of the seat in mm
        shift: shift of the whole assembly along x-axis in mm
    Returns: cq.shape.solid of inner NBZ
    """
    # draw triangle
    triangle = Triangle().build(wheel_base_length, wheel_base_width)  # triangle as Triangle()
    triangle_wp = triangle.construct()  # triangle as cq.workplane
    points = triangle.get_points()

    # add wheels to the corners of the triangle
    tyre_diameter = get_regulation("wheel.tire_diameter")
    wheel = Wheels().create_wheel(tyre_diameter)

    seat = Seat().create_seat()

    # Calculate shift vector - shift along x-axis and raise by half the tire diameter
    shift_vec = cq.Vector(-wheel_base_length / 2, 0, tyre_diameter / 2)

    assembly = cq.Assembly(name="inner_nbz")
    assembly.add(triangle_wp, name="triangle", loc=cq.Location(shift_vec))
    assembly.add(
        wheel,
        loc=cq.Location(cq.Vector(points[0].x, points[0].y, points[0].z) + shift_vec),
        name="wheel_1",
    )
    assembly.add(
        wheel,
        loc=cq.Location(cq.Vector(points[1].x, points[1].y, points[1].z) + shift_vec),
        name="wheel_2",
    )
    assembly.add(
        wheel,
        loc=cq.Location(cq.Vector(points[2].x, points[2].y, points[2].z) + shift_vec),
        name="wheel_3",
    )
    assembly.add(
        seat,
        loc=cq.Location(
            cq.Vector(seat_position_x, 0, seat_position_z) + shift_vec
        ),  # the 500 shall be removed, after correction in seat.py
        name="Seat",
    )

    return assembly


def outer_nbz(length: float, width: float, height: float):
    """
    New implementation of outerNBZ for version 2
    Args: constraints from regulations
    Returns: cq.shape.solid of outer NBZ shape
    """

    box = outer_box2(length, width, height)
    return box


def vision_nbz(seat_pos_x, seat_pos_z, wheel_base_length, get_box: bool = False) -> list[cq.Solid]:
    """
    Generate a cadquery Assembly of the driver's and the tracker's vision NBZ with given parameters from regulations
    Args: regulations.yaml
    Outputs: cadquery Assembly
    """
    # get parameters from regulations
    t_box_len = get_regulation("tracker_box.dimension.length")
    t_box_margin = get_regulation("tracker_box.margin_to_seat")
    eye_level_z = get_regulation("no_build_zones.vision_box.eye_level_z")
    rad = get_regulation("seat.dimension.radius_top")
    neg_angle = math.radians(get_regulation("seat.dimension.minus_angle"))

    '''
    # print all parameters from above
    print(f"Tracker box length: {t_box_len} mm")
    print(f"Tracker box margin to seat: {t_box_margin} mm")
    print(f"Eye level Z: {eye_level_z} mm")
    print(f"Seat radius top: {rad} mm")
    print(f"Negative angle: {neg_angle} radians")
    '''

    # refence point for the tracker box is the backmost point of the seat + margin of 10cm + half length of the box
    point_ref_t_box = cq.Vector(
        seat_pos_x - math.sin(neg_angle) * rad - t_box_margin - t_box_len / 2, 0,-100+ seat_pos_z + math.cos(neg_angle) * rad
    )
    # reference point for the vision box is the eye level of the driver
    point_ref_v_box = cq.Vector(seat_pos_x, 0, seat_pos_z + eye_level_z)

    # calculate shift vector to move both boxes to origin
    tyre_diameter = get_regulation("wheel.tire_diameter")
    shift_vec = cq.Vector(-wheel_base_length / 2, 0, tyre_diameter / 2)

    box, t_box_cone = tracker_box.tracker_box_nbz(box_and_cone=True)
    v_cone = vision_box.vision_driv_nbz()

    vision_assy = cq.Assembly(name="vision_cones_assembly")
    vision_assy.add(t_box_cone, loc=cq.Location(point_ref_t_box + shift_vec), name="tracker_cone")
    vision_assy.add(v_cone, loc=cq.Location(point_ref_v_box + shift_vec), name="driver_cone")

    if get_box:
        cone_assy = cq.Assembly(name="Tracking_box")
        cone_assy.add(box, loc=cq.Location(point_ref_t_box + shift_vec), name="tracker_box")
        return cone_assy, vision_assy

    return vision_assy


def generate_assembly() -> str:
    """
    Generate a cadquery Assembly of the inner NBZ with random parameters within regulation limits - save parameters in millimeters for later use
    Args: regulations.yaml
    Outputs: parameters for cadquery assembly into file
    """
    # get ranges for wheelbase from regulations
    wheel_ranges = get_range_wheels()
    # generate random wheelbase values within ranges
    wheel_base_width = float(
        uniform_distribution(
            wheel_ranges["wheel_base_width"][0],
            wheel_ranges["wheel_base_width"][1],
            size=1,
        )
    )
    wheel_base_length = float(
        uniform_distribution(
            wheel_ranges["wheel_base_length"][0],
            wheel_ranges["wheel_base_length"][1],
            size=1,
        )
    )
    # print ranges
    print(
        f"Generated wheelbase - Width: from {wheel_ranges['wheel_base_width'][0]} mm until {wheel_ranges['wheel_base_width'][1]} mm, Length: from {wheel_ranges['wheel_base_length'][0]} mm until {wheel_ranges['wheel_base_length'][1]} mm"
    )

    # get ranges for seat and position from regulations
    seat_ranges = get_range_seat(wheel_base_width, wheel_base_length)
    seat_position = float(
        uniform_distribution(
            seat_ranges["seat_position"][0],
            seat_ranges["seat_position"][1],
            size=1,
        )
    )

    tyre_diameter = get_regulation("wheel.tire_diameter")

    # add assembly to assembly_map with unique id-number
    # save parameters to file for later use

    # Convert numpy types to native Python types for YAML compatibility
    parameters = {
        "wheel_base_width": float(wheel_base_width),
        "wheel_base_length": float(wheel_base_length),
        "tyre_diameter": float(tyre_diameter),
        "seat_position": float(seat_position),
    }

    # return assembly and id while saving parameters
    return save_assembly_parameters(parameters)


def save_assembly_parameters(parameters: dict) -> str:
    """
    Save assembly parameters to assembly_map.yaml with the next available ID.

    Args:
        parameters: dict with keys like 'wheel_base_width', 'wheel_base_length', etc.

    Returns:
        str: The ID number assigned (e.g., '042')
    """
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # Read existing data
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}  # Handle empty file

    # Get next ID number
    if assembly_map:
        # Keys are integers like 1, 2, 3 (YAML interprets 001 as 1)
        id_numbers = [int(key) for key in assembly_map]
        next_id = max(id_numbers) + 1
    else:
        next_id = 1

    # Add new entry with integer key (YAML will format it appropriately)
    assembly_map[next_id] = parameters

    # Write back to file
    with config_path.open("w") as file:
        yaml.dump(assembly_map, file, default_flow_style=False, sort_keys=False)

    return str(next_id)


def call_assembly(id_number: int) -> tuple[cq.Solid, cq.Solid]:
    """
    Call the cadquery Assembly of the inner NBZ with given id-number from congif.assembly_map.yaml
    Args: id-number
    Returns: cadquery Assembly
    """
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # Read existing data
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    # YAML loads numeric keys as integers, not strings
    if id_number not in assembly_map:
        raise ValueError(f"ID {id_number} not found in assembly map.")  # noqa: TRY003
    params = assembly_map[id_number]
    seat_pos_z = get_regulation("seat.position.z")

    assembly = inner_nbz_sym(
        params["wheel_base_width"],
        params["wheel_base_length"],
        params["tyre_diameter"],
        params["seat_position"],
        seat_pos_z,
    )

    box_width = get_regulation("no_build_zones.outer_prism.width")
    box_height = get_regulation("no_build_zones.outer_prism.height")

    outer = outer_nbz(params["wheel_base_length"] * 2.2, box_width, box_height)


    box, vision = vision_nbz(params["seat_position"], seat_pos_z, params["wheel_base_length"], get_box=True)
    assembly.add(box)

    outer_with_ramps = outer_nbz_ramps(assembly, outer)

    return assembly, outer_with_ramps, vision


def get_range_wheels() -> dict:
    """
    Get ranges for all parameters for wheelbase from regulations
    Args: regulations.yaml
    Returns: dict of ranges
    """

    seat_length = get_regulation("seat.dimension.lr_l")
    seat_width = get_regulation("seat.dimension.width")
    min_distance = get_regulation("seat.position.min_distance_to_back_wheel")  # some margin to the back wheel

    wheel_ax_marg = get_regulation("wheel.margin_axial")  # some margin for the wheels in axial direction
    wheel_width = get_regulation("wheel.dimension.width")
    wheel_d = get_regulation("wheel.tire_diameter")

    wb_w_min = seat_width + 2 * wheel_ax_marg + wheel_width
    wb_w_max = (
        get_regulation("no_build_zones.outer_prism.width")
        - get_regulation("wheel.margin_axial")
        - get_regulation("wheel.tire_width")
    ) / 2
    wb_l_min = seat_length + min_distance + wheel_d / 2
    # take max box dimension and divide by 2.2 to get max wheelbase length then round off to nearest 10
    wb_l_max = round(get_regulation("no_build_zones.outer_prism.length") / 2.2, -1)

    # ensure min is smaller than max
    if wb_w_min >= wb_w_max:
        raise WheelbaseTooSmallError("width_max")
    if wb_l_min >= wb_l_max:
        raise WheelbaseTooSmallError("length_max")

    ranges = {
        "wheel_base_width": (
            wb_w_min,
            wb_w_max,
        ),  # mm
        "wheel_base_length": (
            wb_l_min,
            wb_l_max,
        ),  # mm
    }
    return ranges


def get_range_seat(wheelbase_width: float, wheelbase_length: float) -> dict:
    """
    After wheelbase is generated, get ranges for seat and position from regulations
    Args: regulations.yaml
    Returns: dict of ranges
    """
    # check if wheelbase length is larger than seat length + min distance to back wheel
    seat_length = get_regulation("seat.dimension.lr_l")
    seat_width = get_regulation("seat.dimension.width")
    min_distance = get_regulation("seat.position.min_distance_to_back_wheel")  # some margin to the back wheel

    wheel_ax_marg = get_regulation("wheel.margin_axial")  # some margin for the wheels in axial direction
    wheel_width = get_regulation("wheel.dimension.width")
    wheel_d = get_regulation("wheel.tire_diameter")

    if wheelbase_length <= (seat_length + min_distance + wheel_d / 2):
        raise WheelbaseTooSmallError("length")
    if wheelbase_width <= (seat_width + 2 * wheel_ax_marg + wheel_width):
        raise WheelbaseTooSmallError("width")

    ranges = {
        "seat_position": (
            min_distance + wheel_d / 2,  # mm places the seat in front of the back wheel
            wheelbase_length - seat_length + wheel_d / 2,  # mm
        ),  # mm
    }
    return ranges

def call_assembly_solids(id_number: int) -> tuple[list[cq.Solid], cq.Solid]:
    """
    Call the cadquery Assembly of the inner NBZ with given id-number from congif.assembly_map.yaml

    Args: id-number

    Returns:  tuple with list[inner solids] and outer solid
    """
    inner, outer, vision = call_assembly(id_number)
    return inner.toCompound().Solids(), outer, vision.toCompound().Solids()


def bounding_box_of_assembly(id_number: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Docstring for bounding_box_of_assembly
    
    :param id_number: id number of the assembly of the wanted bounding box
    :return: tuple with touples with ((x_min, y_min, z_min),(x_max,y_max,z_max))
    :rtype: tuple[tuple[float, float, float], tuple[float, float, float]]
    """
    inner, _ = call_assembly(id_number)
    inner_bounding_box = inner.toCompound().BoundingBox()

    return (
        (inner_bounding_box.xmin,
         inner_bounding_box.ymin,
         inner_bounding_box.zmin
        ),

        (inner_bounding_box.xmax,
         inner_bounding_box.ymax,
         inner_bounding_box.zmax
        )
        )

def create_bb_visualization(BB_coords: tuple[tuple[float, float, float], tuple[float, float, float]]) -> cq.Solid:
    """
    Docstring for create_bb_visualization
    
    :param BB_coords: Tuple with tuples with the coordinates of the bounding box wanted to visualize
    :type BB_coords: tuple[tuple[float, float, float], tuple[float, float, float]]
    :return: solid of Bounding Box wanted to visualize
    :rtype: Solid
    """
    MIN = BB_coords[0]
    MAX = BB_coords[1]

    # Calculate dimensions
    x_len = MAX[0] - MIN[0]
    y_len = MAX[1] - MIN[1]
    z_len = MAX[2] - MIN[2]

    # Calculate center point
    center_x = (MAX[0] + MIN[0]) / 2
    center_y = (MAX[1] + MIN[1]) / 2
    center_z = (MAX[2] + MIN[2]) / 2

    # Create and translate the box
    return (cq.Workplane("XY").box(x_len, y_len, z_len).translate((center_x, center_y, center_z))).val()

if __name__ == "__main__":
    from ocp_vscode import show
    demo_id = 30
    inner, outer, vision = call_assembly_solids(demo_id)
    show(inner, outer, vision)
    print(type(inner), type(outer),type(vision))