import math
import sys

import cadquery as cq
from ocp_vscode import show

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


def tracker_box(
    position: tuple[float, float, float] = (
        get_regulation("tracker_box.def_position.x"),
        get_regulation("tracker_box.def_position.y"),
        get_regulation("tracker_box.def_position.z"),
    ),
    angle: tuple[float, float] = (
        get_regulation("tracker_box.def_angle.roll"),
        get_regulation("tracker_box.def_angle.pitch"),
    ),
) -> cq.Workplane:
    g_box_len = get_regulation("tracker_box.dimension.length")
    g_box_wid = get_regulation("tracker_box.dimension.width")
    g_box_hei = get_regulation("tracker_box.dimension.height")
    print(f"Creating box with dimensions {g_box_len} x {g_box_wid} x {g_box_hei} mm")

    cone_hei = get_regulation("tracker_box.cone.height")
    cone_apex_angle = get_regulation("tracker_box.cone.apex_angle")
    print(f"Creating cone with height {cone_hei} mm and apex angle {cone_apex_angle}°")

    theta = math.radians(cone_apex_angle) / 2.0
    r = cone_hei * math.tan(theta)
    cone = cq.Solid.makeCone(0.0, r, cone_hei)

    userinput = input("Show cone? (y/n): ")
    if userinput.lower() == "y":
        show(cq.Workplane("XY").add(cone))

    box_w_cone = (
        cq.Workplane("XY")
        .box(g_box_len, g_box_wid, g_box_hei)
        .add(cone)
        .rotate((0, 0, 0), (0, 1, 0), angle[1])
        .translate((position[0], position[1], position[2] + g_box_hei / 2))
    )

    userinput = input("Show tracker box? (y/n): ")
    if userinput.lower() == "y":
        show(box_w_cone)

    userinput = input("Abort process? (y/-): ")
    if userinput.lower() == "y":
        sys.exit()

    return box_w_cone
