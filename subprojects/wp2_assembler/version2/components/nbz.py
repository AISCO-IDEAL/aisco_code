import random
from pathlib import Path

import cadquery as cq
import yaml
from utils.reg_imp import get_regulation

from .inner.base_triangle import Triangle
from .inner.seat import Seat
from .inner.wheels import Wheels
from .outer.outer_box import outerBox


def inner_nbz2_sym_rand():
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


def generate_Assembly():
    """
    Generate a cadquery Assembly of the inner NBZ with random parameters within regulation limits - save parameters in millimeters for later use
    Args: regulations.yaml
    Outputs: parameters for cadquery assembly into file
    """
    # get ranges for wheelbase from regulations
    wheel_ranges = get_range_wheels()
    wheel_base_width = random.randint(
        wheel_ranges["wheel_base_width"][0], wheel_ranges["wheel_base_width"][1]
    )
    wheel_base_length = random.randint(
        wheel_ranges["wheel_base_length"][0], wheel_ranges["wheel_base_length"][1]
    )

    # get ranges for seat and position from regulations
    seat_ranges = get_range_seat_and_pos(wheel_base_width, wheel_base_length)
    seat_position = random.randint(
        seat_ranges["seat_position"][0], seat_ranges["seat_position"][1]
    )

    tyre_diameter = get_regulation("wheels.tire_diameter")

    assembly = inner_nbz2_sym(
        wheel_base_width=wheel_base_width,
        wheel_base_length=wheel_base_length,
        tyre_diameter=tyre_diameter,
        seat_position=seat_position,
    )

    # add assembly to assembly_map with unique id-number
    # save parameters to file for later use

    parameters = {
        "wheel_base_width": wheel_base_width,
        "wheel_base_length": wheel_base_length,
        "tyre_diameter": tyre_diameter,
        "seat_position": seat_position,
    }

    save_assembly_parameters(parameters)

    return assembly


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
        # Keys are strings like '001', '002', etc.
        id_numbers = [float(key) for key in assembly_map]
        next_id = max(id_numbers) + 1
    else:
        next_id = 1

    # Format as 3-digit string: '001', '042', etc.
    id_string = f"{next_id:03d}"

    # Add new entry
    assembly_map[id_string] = parameters

    # Write back to file
    with config_path.open("w") as file:
        yaml.dump(assembly_map, file, default_flow_style=False, sort_keys=False)

    return id_string


def call_assembly(id_number: int) -> cq.Assembly:
    """
    Call the cadquery Assembly of the inner NBZ with given id-number from congif.assembly_map.yaml
    Args: id-number
    Returns: cadquery Assembly
    """
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # Read existing data
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    id_string = f"{id_number:03d}"
    if id_string not in assembly_map:
        raise ValueError(f"ID {id_string} not found in assembly map.")  # noqa: TRY003

    params = assembly_map[id_string]

    assembly = inner_nbz2_sym(
        wheel_base_width=params["wheel_base_width"],
        wheel_base_length=params["wheel_base_length"],
        tyre_diameter=params["tyre_diameter"],
        seat_position=params["seat_position"],
    )

    return assembly


def inner_nbz2_sym(
    wheel_base_width=1000,
    wheel_base_length=2500,
    tyre_diameter=557,
    seat_position=800,
    shift=-500,
):
    """
    Deterministic implementation of innerNBZ for version 2
    Args: constraints from regulations
    Returns: cq.shape.solid of inner NBZ shape
    """
    # draw triangle
    triangle = Triangle().build(
        wheel_base_length, wheel_base_width
    )  # triangle as Triangle()
    triangle_wp = triangle.construct()  # triangle as cq.workplane
    points = triangle.get_points()

    # add wheels to the corners of the triangle
    wheel = Wheels().create_wheel(tyre_diameter)

    seat = Seat().create_seat()

    # Calculate shift vector - shift along x-axis and raise by half the tire diameter
    shift_vector = cq.Vector(shift, 0, tyre_diameter / 2)

    assembly = (
        cq.Assembly(triangle_wp, name="triangle", loc=cq.Location(shift_vector))
        # Add wheels with shift applied
        .add(
            wheel,
            loc=cq.Location(cq.Vector(points[0].x, points[0].y, points[0].z)),
            name="wheel_1",
        )
        .add(
            wheel,
            loc=cq.Location(cq.Vector(points[1].x, points[1].y, points[1].z)),
            name="wheel_2",
        )
        .add(
            wheel,
            loc=cq.Location(cq.Vector(points[2].x, points[2].y, points[2].z)),
            name="wheel_3",
        )
        .add(
            seat,
            loc=cq.Location(cq.Vector(seat_position, 0, -100)),
            name="Seat",
        )
    )

    return assembly


def get_range_wheels() -> dict:
    """
    Get ranges for all parameters for wheelbase from regulations
    Args: regulations.yaml
    Returns: dict of ranges
    """
    ranges = {
        "wheel_base_width": (
            get_regulation("wheel.base.min_width"),
            get_regulation("wheel.base.max_width"),
        ),  # mm
        "wheel_base_length": (
            get_regulation("wheel.base.min_length"),
            get_regulation("wheel.base.max_length"),
        ),  # mm
    }
    return ranges


def get_range_seat_and_pos(wheelbase_width: float, wheelbase_length: float) -> dict:
    """
    After wheelbase is generated, get ranges for seat and position from regulations
    Args: regulations.yaml
    Returns: dict of ranges
    """
    ranges = {
        "seat_position": (
            get_regulation("seat.position.min_distance_to_back_wheel")
            + get_regulation(
                "wheel.tire_diameter"
            ),  # mm places the seat in front of the back wheel
            get_regulation("seat.max_position"),
        ),  # mm
    }
    return ranges


def outer_nbz():
    """
    New implementation of outerNBZ for version 2
    Args: constraints from regulations
    Returns: cq.shape.solid of outer NBZ shape
    """

    box = outerBox()
    return box
