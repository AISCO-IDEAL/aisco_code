import math
from dataclasses import dataclass

import cadquery as cq
from ocp_vscode import *

from .params import *


# -- Seat class: ---
class Seat:
    """The Seat placeholder Component."""

    def __init__(
        self,
        params: SeatParams = SeatParams(),
        chassis_params: ChassisParams = ChassisParams(),
    ):
        # --- Parameters: ---
        self.params = params
        self.chassis_params = chassis_params
        plus_angle_rad = math.radians(self.params.plus_angle)  # plus_angle in radians
        minus_angle_rad = math.radians(
            self.params.minus_angle
        )  # minus_angle in radians
        aux_l_0E: float = (
            (self.params.width_hr / 2) ** 2
            + (self.params.radius_top - self.params.height_hr) ** 2
        ) ** 0.5  # Auxiliary size for building

        # coordinates for the 2D-Wire:
        O = (0, 0)
        A = (
            -(self.params.radius_top * math.sin(minus_angle_rad)),
            self.params.radius_top * math.cos(minus_angle_rad),
        )
        B = (
            self.params.radius_top * math.sin(plus_angle_rad),
            self.params.radius_top * math.cos(plus_angle_rad),
        )
        C = (0, self.params.radius_top)
        D = (
            self.params.width_hr / 2,
            self.params.radius_top
            * math.cos(math.asin((self.params.width_hr / 2) / self.params.radius_top)),
        )
        E = (D[0], self.params.radius_top - self.params.height_hr)
        F = (
            self.params.width / 2,
            aux_l_0E * math.cos(math.atan((self.params.width / 2) / aux_l_0E)),
        )
        G = (self.params.width / 2, self.params.radius_top)

        # --- model building: ---
        # create a 2D-face in the XZ-plane
        profile1 = (  # Wedge structure
            cq.Workplane("XZ")
            .lineTo(A[0], A[1])
            .radiusArc(endPoint=(B[0], B[1]), radius=self.params.radius_top)
            .close()
            .val()
        )
        face1 = cq.Face.makeFromWires(profile1)

        profile2 = (  # legroom
            cq.Workplane("XZ").rect(self.params.Lr_l, self.params.Lr_h).val()
        )

        face2 = cq.Face.makeFromWires(profile2)
        face2 = face2.translate((self.params.Lr_l / 2, 0, self.params.Lr_h / 2))

        face_r = face1.fuse(face2)

        # Extrudiere die 2D Fläche
        solid1 = cq.Workplane("XZ").add(face_r).extrude(self.params.width)

        # Verschiebung des Koordinatensystems
        solid1 = solid1.translate((0, self.params.width / 2, 0))

        # set_port(3939)
        # show_object(solid1)

        # Erstelle das 2D - Profil des Kopfausschnittes in der "ZY"-Ebene
        profile3 = (
            cq.Workplane("YZ")
            .moveTo(C[0], C[1])
            .radiusArc(endPoint=(D[0], D[1]), radius=self.params.radius_top)
            .lineTo(E[0], E[1])
            .radiusArc(endPoint=(F[0], F[1]), radius=aux_l_0E)
            .lineTo(G[0], G[1])
            .close()
            .val()
        )
        face3 = cq.Face.makeFromWires(profile3)

        # Das Profil des halben Kopfausschnittes spiegeln und zusammenfügen
        profile4 = profile3.mirror("XZ")
        face4 = cq.Face.makeFromWires(profile4)

        h_face = face3.fuse(face4)
        h_face = cq.Workplane("YZ").add(h_face)

        # Pfade für .sweep() definieren
        fwd_path = (
            cq.Workplane("XZ")
            .moveTo(C[0], C[1])
            .radiusArc(endPoint=(B[0], B[1]), radius=self.params.radius_top)
        )
        bwd_path = (
            cq.Workplane("XZ")
            .moveTo(C[0], C[1])
            .radiusArc(endPoint=(A[0], A[1]), radius=-self.params.radius_top)
        )

        # .sweep() nach vorne und hinten auf den Kopfausschnitt anwenden und union
        sweep_fwd = h_face.sweep(fwd_path, isFrenet=True)
        sweep_bwd = h_face.sweep(bwd_path, isFrenet=True)

        h_sweep = sweep_fwd.union(sweep_bwd)

        # Soild1 - h_sweep
        result = solid1.cut(h_sweep)

        # Verschiebe das Koordinatensystem ins Zentrum des umhüllenden Quaders
        bBox = result.val().BoundingBox()
        self.s_height = bBox.zlen
        self.s_length = bBox.xlen
        self.s_width = bBox.ylen

        self.model = result.translate(-bBox.center).translate((0, 0, bBox.zlen / 2))

    # Readonly - access
    @property
    def seat_height(self):
        return self.s_height

    @property
    def seat_length(self):
        return self.s_length

    @property
    def seat_width(self):
        return self.s_width

    @property
    def critical_point(self):
        vertices = (
            self.model.faces("<Z")
            .workplane(offset=-self.chassis_params.height)
            .section()
            .vertices()
            .vals()
        )

        critical_point = vertices[1]
        return (critical_point.X, critical_point.Y)


# s = Seat()
# print(s.critical_point)
# set_port(3939)
# show_object(s.model)
