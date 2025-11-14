import math
from dataclasses import dataclass

import cadquery as cq


# --- Seat class (hard-coded parameters) ---
@dataclass
class Seat:
    """Seat specifications and geometry for the solar car."""

    # fixed seat parameters [mm]
    width: float = 450.0            # general width of the seat
    radius_top: float = 835.0       # radius used to build the backrest
    minus_angle: float = 27.0       # angle of the backrest backwards [°]
    plus_angle: float = 45.0        # angle of the backrest forwards [°]
    Lr_l: float = 1000.0            # legroom length
    Lr_h: float = 350.0             # legroom height
    width_hr: float = 280.0         # headroom width
    height_hr: float = 300.0        # headroom height
    position: float = 1000.0        # position relative to the wheelbase midpoint
    color: str = "red"              # visualization color
    margin_axial: float = 10.0      # optional side margin
    margin_radial: float = 10.0     # optional top/bottom margin

    def get_seat_info(self) -> str:
        """Return a short description of the seat configuration."""
        return (
            f"Seat: width={self.width}mm, radius_top={self.radius_top}mm, "
            f"angles (+{self.minus_angle}° / -{self.plus_angle}°), "
            f"legroom={self.Lr_l}×{self.Lr_h}mm, headroom={self.width_hr}×{self.height_hr}mm."
        )

    def create_seat(self) -> cq.Workplane:
        """
        Create a CAD model of the seat using hard-coded parameters.
        Returns:
            CadQuery Workplane object representing the seat.
        """
        # convert angles
        minus_angle_rad = math.radians(self.minus_angle)
        plus_angle_rad = math.radians(self.plus_angle)
        aux_l_0E = ((self.width_hr / 2) ** 2 + (self.radius_top - self.height_hr) ** 2) ** 0.5

        # coordinates for main seat surfaces
        A = (-(self.radius_top * math.sin(plus_angle_rad)),
             self.radius_top * math.cos(plus_angle_rad))
        B = (self.radius_top * math.sin(minus_angle_rad),
             self.radius_top * math.cos(minus_angle_rad))
        C = (0, self.radius_top)
        D = (self.width_hr / 2,
             self.radius_top * math.cos(math.asin((self.width_hr / 2) / self.radius_top)))
        E = (D[0], self.radius_top - self.height_hr)
        F = (self.width / 2,
             aux_l_0E * math.cos(math.atan((self.width / 2) / aux_l_0E)))
        G = (self.width / 2, self.radius_top)

        # --- build main body (wedge + legroom) ---
        profile_wedge = (
            cq.Workplane("XZ")
            .lineTo(A[0], A[1])
            .radiusArc(endPoint=(B[0], B[1]), radius=self.radius_top)
            .close()
        )
        solid_wedge = profile_wedge.extrude(self.width).translate((0, self.width / 2, 0))

        legroom_face = (
            cq.Workplane("XZ")
            .rect(self.Lr_l, self.Lr_h)
            .translate((self.Lr_l / 2, 0, self.Lr_h / 2))
        )
        solid_legroom = legroom_face.extrude(self.width).translate((0, self.width / 2, 0))

        seat_base = solid_wedge.union(solid_legroom)

        # --- create head cutout ---
        profile_head = (
            cq.Workplane("YZ")
            .moveTo(C[0], C[1])
            .radiusArc(endPoint=(D[0], D[1]), radius=self.radius_top)
            .lineTo(E[0], E[1])
            .radiusArc(endPoint=(F[0], F[1]), radius=aux_l_0E)
            .lineTo(G[0], G[1])
            .close()
        )

        head_face = profile_head.val()
        mirrored_face = head_face.mirror("XZ")
        face1 = cq.Face.makeFromWires(head_face)
        face2 = cq.Face.makeFromWires(mirrored_face)
        head_shape = face1.fuse(face2)

        # sweep head cutout forward and backward
        path_fwd = cq.Workplane("XZ").moveTo(C[0], C[1]).radiusArc(endPoint=B, radius=self.radius_top)
        path_bwd = cq.Workplane("XZ").moveTo(C[0], C[1]).radiusArc(endPoint=A, radius=-self.radius_top)
        sweep_fwd = cq.Workplane("YZ").add(head_shape).sweep(path_fwd, isFrenet=True)
        sweep_bwd = cq.Workplane("YZ").add(head_shape).sweep(path_bwd, isFrenet=True)
        head_cut = sweep_fwd.union(sweep_bwd)

        # subtract cutout from base
        seat = seat_base.cut(head_cut)

        # recenter geometry
        bBox = seat.val().BoundingBox()
        seat = seat.translate(-bBox.center).translate((0, 0, bBox.zlen / 2))
        return seat

    def create_seat_with_margin(self) -> cq.Workplane:
        """
        Create the seat with additional margins.
        """
        total_width = self.width + 2 * self.margin_axial
        total_radius = self.radius_top + self.margin_radial

        minus_angle_rad = math.radians(self.minus_angle)
        plus_angle_rad = math.radians(self.plus_angle)

        A = (-(total_radius * math.sin(plus_angle_rad)),
             total_radius * math.cos(plus_angle_rad))
        B = (total_radius * math.sin(minus_angle_rad),
             total_radius * math.cos(minus_angle_rad))

        profile = (
            cq.Workplane("XZ")
            .lineTo(A[0], A[1])
            .radiusArc(endPoint=B, radius=total_radius)
            .close()
        )

        seat_with_margin = profile.extrude(total_width)
        seat_with_margin = seat_with_margin.translate((0, total_width / 2, 0))
        return seat_with_margin
