import math

import cadquery as cq

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


def vision_driv_nbz() -> cq.Workplane:
    """
    Generate a cadquery Assembly of the driver's vision NBZ
    Args: regulations.yaml
    Outputs: cadquery Assembly
    """
    radius = get_regulation("no_build_zones.vision_box.radius")
    pos_height = get_regulation("no_build_zones.vision_box.pos_height")
    neg_height = -get_regulation("no_build_zones.vision_box.neg_height")
    pos_angle = math.radians(get_regulation("no_build_zones.vision_box.pos_angle"))
    neg_angle = -math.radians(get_regulation("no_build_zones.vision_box.neg_angle"))
    dist_center = get_regulation("no_build_zones.vision_box.inner_radius")

    if dist_center >= radius:
        raise ValueError("Inner radius must be smaller than outer radius.")  # noqa: TRY003

    if dist_center <= 0:
        raise ValueError("Inner radius must be strictly positive.")  # noqa: TRY003

    point_a = cq.Vector(dist_center, 0, pos_height / radius * dist_center)
    point_b = cq.Vector(dist_center, 0, neg_height / radius * dist_center)
    point_c = cq.Vector(radius, 0, pos_height)
    point_d = cq.Vector(radius, 0, neg_height)

    start_point = cq.Vector(radius * math.cos(pos_angle), radius * math.sin(pos_angle), 0)
    mid_point = cq.Vector(radius, 0, 0)
    end_point = cq.Vector(radius * math.cos(neg_angle), radius * math.sin(neg_angle), 0)

    # print all points
    print(f"Point A: {point_a}")
    print(f"Point B: {point_b}")
    print(f"Point C: {point_c}")
    print(f"Starting Point: {start_point}")
    print(f"Mid Point: {mid_point}")
    print(f"Ending Point: {end_point}")

    arch_pos = (
        cq.Workplane("XY").moveTo(start_point.x, start_point.y).radiusArc(endPoint=(mid_point.x, mid_point.y), radius=radius)
    )
    arch_neg = cq.Workplane("XY").moveTo(mid_point.x, mid_point.y).radiusArc(endPoint=(end_point.x, end_point.y), radius=radius)

    profile_v_box = (
        cq.Workplane("XZ")
        .moveTo(point_a.x, point_a.z)
        .lineTo(point_c.x, point_c.z)
        .lineTo(point_d.x, point_d.z)
        .lineTo(point_b.x, point_b.z)
        .close()
    )

    v_box_sketch = cq.Face.makeFromWires(
        profile_v_box.wires().val()  # noqa: it works like this
    )

    sweep_left = cq.Workplane("YZ").add(v_box_sketch).sweep(arch_pos, isFrenet=True)
    sweep_right = cq.Workplane("XZ").add(v_box_sketch).sweep(arch_neg, isFrenet=True)
    v_box = sweep_left.union(sweep_right)

    # convert to cq.Solid
    v_box_solid = v_box.val()

    return v_box_solid
