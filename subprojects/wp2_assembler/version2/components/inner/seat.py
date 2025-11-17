import math
import sys
from dataclasses import dataclass, field

import cadquery as cq
from ocp_vscode import show
from utils.reg_imp import get_regulation


# --- Seat class (hard-coded parameters) ---
@dataclass
class Seat:
    """Seat specifications and geometry for the solar car."""

    # fixed seat parameters [mm]
    width: float = field(
        default_factory=lambda: get_regulation("seat.dimension.width")
    )  # general width of the seat
    radius_top: float = field(
        default_factory=lambda: get_regulation("seat.dimension.radius_top")
    )  # radius used to build the backrest
    minus_angle: float = field(
        default_factory=lambda: get_regulation("seat.dimension.minus_angle")
    )  # angle of the backrest backwards [°]
    plus_angle: float = field(
        default_factory=lambda: get_regulation("seat.dimension.plus_angle")
    )  # angle of the backrest forwards [°]
    Lr_l: float = field(
        default_factory=lambda: get_regulation("seat.dimension.lr_l")
    )  # legroom length
    Lr_h: float = field(
        default_factory=lambda: get_regulation("seat.dimension.lr_h")
    )  # legroom height
    width_hr: float = field(
        default_factory=lambda: get_regulation("seat.dimension.width_hr")
    )  # headroom width
    height_hr: float = field(
        default_factory=lambda: get_regulation("seat.dimension.height_hr")
    )  # headroom height

    def get_seat_info(self) -> str:
        """Return a short description of the seat configuration."""
        return (
            f"Seat: width={self.width}mm, radius_top={self.radius_top}mm, "
            f"angles (+{self.minus_angle}° / -{self.plus_angle}°), "
            f"legroom={self.Lr_l}x{self.Lr_h}mm, headroom={self.width_hr}x{self.height_hr}mm."
        )

    def _create_wedge(  # noqa: PLR0913
        self,
        rad: float,  # radius
        ang_min: float,  # angle backwards
        ang_plus: float,  # angle forwards
        wid: float,  # width of seat
        h_hr: float,  # headroom height
        w_hr: float,  # headroom width
    ) -> cq.Workplane:
        """
        Create the wedge part of the seat.
        Args:
            rad (float): Radius for backrest curve.
            ang_min (float): Backrest angle backwards in radians.
            ang_plus (float): Backrest angle forwards in radians.
            wid (float): Width of the seat.
            h_hr (float): Headroom height.
            w_hr (float): Headroom width.
        """
        # create two rectangles with curved tops, by intersecting a circle with a rectangle with width = wid

        rad_s = rad - h_hr  # radius for headroom curve

        sketch = (
            cq.Sketch("XY")
            .circle(rad_s)
            .push([(rad_s / 2, 0)])
            .rect(rad_s, wid, mode="i")
            .push([(rad / 2, 0)])
            .rect(rad, w_hr, mode="a")
            .push([(0, 0)])
            .circle(rad, mode="i")
        )

        userinput = input("Show sketch? (y/-) ")
        if userinput == "y":
            show(sketch)

        # recreate sketch as workplane object for sweeping
        w_sketch = (
            cq.Workplane("YZ")
            .sketch()
            .circle(rad_s)
            .push([(0, rad_s / 2)])
            .rect(wid, rad_s, mode="i")
            .push([(0, rad / 2)])
            .rect(w_hr, rad, mode="a")
            .push([(0, 0)])
            .circle(rad, mode="i")
            .finalize()
        )
        userinput = input("Show workplane sketch faces? (y/-) ")
        if userinput == "y":
            show(w_sketch)

        start_point = cq.Vector(-math.sin(ang_min) * rad, 0, math.cos(ang_min) * rad)
        end_point = cq.Vector(math.sin(ang_plus) * rad, 0, math.cos(ang_plus) * rad)
        arc = (
            cq.Workplane("XZ")
            .moveTo(start_point.x, start_point.z)
            .radiusArc(endPoint=(end_point.x, end_point.z), radius=rad)
        )

        userinput = input("Show arc? (y/-) ")
        if userinput == "y":
            show(arc)
        # add sketch to YZ-workplane and revolve by corresponding angles
        # Convert sketch to face first, then sweep along the arc
        # Get the wire from the arc path
        arc_wire = arc.val()

        # Get the actual face objects from the finalized sketch
        face_list = w_sketch.faces().vals()

        # Create a new workplane and add the faces, then sweep
        wedge = cq.Workplane("YZ")
        for face in face_list:
            wedge = wedge.add(face)
        wedge = wedge.sweep(arc_wire, isFrenet=True)

        userinput = input("Show wedge? (y/-) ")
        if userinput == "y":
            show(wedge)

        userinput = input("Continue? (y/-) ")
        if userinput != "y":
            sys.exit("Seat creation aborted by user.")

        return wedge

    def create_seat(self) -> cq.Workplane:
        """
        Create a CAD model of the seat using regulation parameters.
        Args:
            None
        Returns:
            CadQuery Workplane object representing the seat.
        """

        wedge = self._create_wedge(
            self.radius_top,
            math.radians(self.minus_angle),
            math.radians(self.plus_angle),
            self.width,
            self.height_hr,
            self.width_hr,
        )

        return wedge

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

        userinput = input("Show wedge from seat? (y/-)")
        if userinput == "y":
            show(solid_wedge)

        legroom_face = (
            cq.Workplane("XZ")
            .rect(self.Lr_l, self.Lr_h)
            .translate((self.Lr_l / 2, 0, self.Lr_h / 2))
        )
        solid_legroom = legroom_face.extrude(self.width).translate(
            (0, self.width / 2, 0)
        )

        userinput = input("Show legroom from seat? (y/-)")
        if userinput == "y":
            show(solid_legroom)

        seat_base = solid_wedge.union(solid_legroom)

        userinput = input("Show union from seat? (y/-)")
        if userinput == "y":
            show(seat_base)

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
        face1 = cq.Face.makeFromWires(head_face)
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
