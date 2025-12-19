import math
from dataclasses import dataclass, field

import cadquery as cq

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


# --- Seat class (hard-coded parameters) ---
@dataclass
class Seat:
    """Seat specifications and geometry for the solar car."""

    # fixed seat parameters [mm]
    width: float = field(default_factory=lambda: get_regulation("seat.dimension.width"))  # general width of the seat
    radius_top: float = field(
        default_factory=lambda: get_regulation("seat.dimension.radius_top")
    )  # radius used to build the backrest
    minus_angle: float = field(
        default_factory=lambda: get_regulation("seat.dimension.minus_angle")
    )  # angle of the backrest backwards [°]
    plus_angle: float = field(
        default_factory=lambda: get_regulation("seat.dimension.plus_angle")
    )  # angle of the backrest forwards [°]
    Lr_l: float = field(default_factory=lambda: get_regulation("seat.dimension.lr_l"))  # legroom length
    Lr_h: float = field(default_factory=lambda: get_regulation("seat.dimension.lr_h"))  # legroom height
    width_hr: float = field(default_factory=lambda: get_regulation("seat.dimension.width_hr"))  # headroom width
    height_hr: float = field(default_factory=lambda: get_regulation("seat.dimension.height_hr"))  # headroom height

    def get_seat_info(self) -> str:
        """Return a short description of the seat configuration."""
        return (
            f"Seat: width={self.width}mm, radius_top={self.radius_top}mm, "
            f"angles (+{self.minus_angle}° / -{self.plus_angle}°), "
            f"legroom={self.Lr_l}x{self.Lr_h}mm, headroom={self.width_hr}x{self.height_hr}mm."
        )

    def _create_wedge(
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
        # create two rectangles with curved tops, by intersecting a circle with a rectangle with width = wid - see documentation for more details

        rad_s = rad - h_hr  # radius for headroom curve

        # the apothem is the height of a triangle formed by radius and half the width - check wikipedia
        apothem_1 = math.sqrt(rad**2 - (w_hr / 2) ** 2)
        apothem_2 = math.sqrt(rad_s**2 - (w_hr / 2) ** 2)
        apothem_3 = math.sqrt(rad_s**2 - (wid / 2) ** 2)

        point_a = (0, rad)
        # point b is the right lower corner of a circular segment with width w_hr and radius rad
        point_b = (w_hr / 2, apothem_1)
        # point c is the right lower corner of a circular segment with width w_hr and radius rad_s
        point_c = (w_hr / 2, apothem_2)
        # point d is the right lower corner of a circular segment with width wid and radius rad_s
        point_d = (wid / 2, apothem_3)
        point_e = (wid / 2, 1)  # avoid exact zero for sweeping - will cause errors
        point_f = (0, 1)

        # profile to sweep the wedge
        profile_head = (
            cq.Workplane("YZ")
            .moveTo(point_a[0], point_a[1])
            .radiusArc(endPoint=point_b, radius=rad)
            .lineTo(point_c[0], point_c[1])
            .radiusArc(endPoint=(point_d[0], point_d[1]), radius=rad_s)
            .lineTo(point_e[0], point_e[1])
            .lineTo(point_f[0], point_f[1])
            .close()
        )

        # make to cq.Face and mirror profile, to sweep around centerline with 1mm margin
        profile = profile_head.wires().val()
        face1 = cq.Face.makeFromWires(
            profile  # noqa: it works like this
        )
        face2 = face1.mirror("XZ")
        w_sketch = face1.fuse(face2)

        # start point is the point the most backward on the wedge
        # end point is the point the most forward on the wedge
        start_point = cq.Vector(-math.sin(ang_min) * rad, 0, math.cos(ang_min) * rad)
        end_point = cq.Vector(math.sin(ang_plus) * rad, 0, math.cos(ang_plus) * rad)

        # create arcs to sweep along
        arc_bwd = (
            cq.Workplane("XZ").moveTo(point_a[0], point_a[1]).radiusArc(endPoint=(start_point.x, start_point.z), radius=-rad)
        )
        arc_fwd = cq.Workplane("XZ").moveTo(end_point.x, end_point.z).radiusArc(endPoint=(point_a[0], point_a[1]), radius=-rad)

        # do the sweeping
        sweep_bwd = cq.Workplane("YZ").add(w_sketch).sweep(arc_bwd, isFrenet=True)
        sweep_fwd = cq.Workplane("YZ").add(w_sketch).sweep(arc_fwd, isFrenet=True)
        wedge = sweep_bwd.union(sweep_fwd)

        return wedge

    def _create_legroom(
        self,
        lr_l: float,  # legroom length
        lr_h: float,  # legroom height
        wid: float,  # width of seat
    ) -> cq.Workplane:
        """
        Create the legroom part of the seat.
        Args:
            Lr_l (float): Legroom length.
            Lr_h (float): Legroom height.
            wid (float): Width of the seat.
        """
        solid_legroom = cq.Workplane("YZ").rect(wid, lr_h, centered=(True, False)).extrude(lr_l)

        return solid_legroom

    def create_seat(
        self,
    ) -> cq.Workplane:  # test failed - sweep function not working as expected
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

        legroom = self._create_legroom(self.Lr_l, self.Lr_h, self.width)

        return wedge.union(legroom)

if __name__ == "__main__":
    import time
    start = time.time()
    seat = Seat()
    s = seat.create_seat()

    bb = s.val().BoundingBox()

    bbox_list = [
        bb.xmin, bb.xmax,
        bb.ymin, bb.ymax,
        bb.zmin, bb.zmax
    ]
    end = time.time()
    print(bbox_list)
    print(end-start)
    from ocp_vscode import show
    show(s)