import math
from dataclasses import dataclass, field

import cadquery as cq

def create_seat_old(self) -> cq.Workplane:
        """
        Create a CAD model of the seat using hard-coded parameters.
        Returns:
            CadQuery Workplane object representing the seat.
        """
        # convert angles
        minus_angle_rad = math.radians(self.minus_angle)
        plus_angle_rad = math.radians(self.plus_angle)
        aux_l_0e = (
            (self.width_hr / 2) ** 2 + (self.radius_top - self.height_hr) ** 2
        ) ** 0.5

        # coordinates for main seat surfaces
        point_a = (
            -(self.radius_top * math.sin(plus_angle_rad)),
            self.radius_top * math.cos(plus_angle_rad),
        )
        point_b = (
            self.radius_top * math.sin(minus_angle_rad),
            self.radius_top * math.cos(minus_angle_rad),
        )
        point_c = (0, self.radius_top)
        point_d = (
            self.width_hr / 2,
            self.radius_top
            * math.cos(math.asin((self.width_hr / 2) / self.radius_top)),
        )
        point_e = (point_d[0], self.radius_top - self.height_hr)
        point_f = (
            self.width / 2,
            aux_l_0e * math.cos(math.atan((self.width / 2) / aux_l_0e)),
        )
        point_g = (self.width / 2, self.radius_top)

        # --- build main body (wedge + legroom) ---
        profile_wedge = (
            cq.Workplane("XZ")
            .lineTo(point_a[0], point_a[1])
            .radiusArc(endPoint=(point_b[0], point_b[1]), radius=self.radius_top)
            .close()
        )
        solid_wedge = profile_wedge.extrude(self.width).translate(
            (0, self.width / 2, 0)
        )

        legroom_face = cq.Workplane("XZ").rect(self.Lr_l, self.Lr_h)
        solid_legroom = (
            legroom_face.extrude(self.width)
            .translate((0, self.width / 2, 0))
            .translate((self.Lr_l / 2, 0, self.Lr_h / 2))
        )

        seat_base = solid_wedge.union(solid_legroom)

        # --- create head cutout ---
        profile_head = (
            cq.Workplane("YZ")
            .moveTo(point_c[0], point_c[1])
            .radiusArc(endPoint=(point_d[0], point_d[1]), radius=self.radius_top)
            .lineTo(point_e[0], point_e[1])
            .radiusArc(endPoint=(point_f[0], point_f[1]), radius=aux_l_0e)
            .lineTo(point_g[0], point_g[1])
            .close()
        )

        head_face = profile_head.wires().val()
        face1 = cq.Face.makeFromWires(head_face)  # noqa: not used
        face2 = face1.mirror("XZ")
        head_shape = face1.fuse(face2)

        # sweep head cutout forward and backward
        path_fwd = (
            cq.Workplane("XZ")
            .moveTo(point_c[0], point_c[1])
            .radiusArc(endPoint=point_b, radius=self.radius_top)
        )
        path_bwd = (
            cq.Workplane("XZ")
            .moveTo(point_c[0], point_c[1])
            .radiusArc(endPoint=point_a, radius=-self.radius_top)
        )
        sweep_fwd = cq.Workplane("YZ").add(head_shape).sweep(path_fwd, isFrenet=True)
        sweep_bwd = cq.Workplane("YZ").add(head_shape).sweep(path_bwd, isFrenet=True)
        head_cut = sweep_fwd.union(sweep_bwd)

        # subtract cutout from base
        seat = seat_base.cut(head_cut)

        # recenter geometry
        b_box = seat.findSolid().BoundingBox()
        seat = seat.translate(
            (-b_box.center.x, -b_box.center.y, -b_box.center.z)
        ).translate((0, 0, b_box.zlen / 2))
        return seat
