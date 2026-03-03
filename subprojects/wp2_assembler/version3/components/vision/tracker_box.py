import math

import cadquery as cq

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


def tracker_box_nbz(
    box_and_cone: bool = False
) -> cq.Workplane:
    """
    Generate a cadquery Assembly of the Tracker Box with its cone on top
    Args: regulations.yaml
    Outputs: cadquery Assembly
    """

    g_box_len = get_regulation("tracker_box.dimension.length")
    g_box_wid = get_regulation("tracker_box.dimension.width")
    g_box_hei = get_regulation("tracker_box.dimension.height")
    #print(f"Creating box with dimensions {g_box_len} x {g_box_wid} x {g_box_hei} mm")

    cone_hei = get_regulation("tracker_box.cone.height")
    cone_apex_angle = get_regulation("tracker_box.cone.apex_angle")
    #print(f"Creating cone with height {cone_hei} mm and apex angle {cone_apex_angle}°")

    theta = math.radians(cone_apex_angle) / 2.0
    r = cone_hei * math.tan(theta)
    cone = cq.Solid.makeCone(0.0, r, cone_hei)

    box_w_cone = (
        cq.Workplane("XY")
        .box(g_box_len, g_box_wid, g_box_hei)
        .add(cone)
        .translate((0, 0, g_box_hei / 2))
    )

    box = (
        cq.Workplane("XY")
        .box(g_box_len, g_box_wid, g_box_hei)
        .translate((0, 0, g_box_hei / 2))
    )

    vision_cone = (
        cq.Workplane("XY")
        .add(cone)
        .translate((0, 0, g_box_hei / 2))
    )
    
    if box_and_cone:
        return box, vision_cone

    return box_w_cone


def tracker_cone_nbz() -> cq.Solid:
    """
    Generate only the cone part of the tracker box NBZ
    Args: regulations.yaml
    Outputs: cadquery Solid
    """

    cone_hei = get_regulation("tracker_box.cone.height")
    cone_apex_angle = get_regulation("tracker_box.cone.apex_angle")
    #print(f"Creating cone with height {cone_hei} mm and apex angle {cone_apex_angle}°")

    theta = math.radians(cone_apex_angle) / 2.0
    r = cone_hei * math.tan(theta)
    cone = cq.Solid.makeCone(0.0, r, cone_hei)

    return cone
