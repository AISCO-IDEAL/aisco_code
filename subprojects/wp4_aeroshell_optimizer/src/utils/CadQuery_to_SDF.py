"""
This Code is written by written by Clemens Huber
"""
from typing import cast

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON

# The following Code is in mms
rand = 1500
default_pointgrid_ranges = [[int(-5800/2-rand), int(5800/2 + rand)], [int(-1150 - rand),int(1150 + rand)], [int(-rand), int(1650 + rand)]]
default_pointgrid_steps = [200, 150, 50]
default_pointgrid = [[x, y, z] for x in range(default_pointgrid_ranges[0][0], default_pointgrid_ranges[0][1], default_pointgrid_steps[0]) for y in range(default_pointgrid_ranges[1][0], default_pointgrid_ranges[1][1], default_pointgrid_steps[1]) for z in range(default_pointgrid_ranges[2][0], default_pointgrid_ranges[2][1], default_pointgrid_steps[2])]
# print(f"Default pointgrid has {len(default_pointgrid)} points.")
class SDFfromCadQuery:
    """
    Pre Conditions:
    - shape: CadQuery Shape
    - pointgrid (optional): list of points where the SDF is calculated

    Post Conditions:
        If you want to get the SDF value of all Points in the pointgrid use sdf_at_pointgrid() to get the list of the corresponding SDF values
        If you want the SDF value of a single point use sdf(x,y,z) it returns just one value
    """
    def __init__(self, shape: cq.Shape, pointgrid:list[list[float]]=default_pointgrid):  # noqa: B006
        self._shape = shape
        self._pointgrid = pointgrid
        self._sdf_at_pointgrid = [self.fsdf(x, y, z) for x, y, z in self._pointgrid]
    def sdf(self, x: float, y: float, z: float) -> float:
        """
        Signed distance from (x, y, z) to self._shape (CadQuery shape).
        Negative inside, zero on surface, positive outside.
        """

        # Build point and shape
        pnt = gp_Pnt(x, y, z)
        vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex()
        shape_ocp = self._shape.wrapped  # TopoDS_Shape

        # --- Step 1: Unsigned distance ---
        dist_shape_shape = BRepExtrema_DistShapeShape(vertex, shape_ocp)
        dist_shape_shape.Perform()

        if not dist_shape_shape.IsDone() or dist_shape_shape.NbSolution() == 0:
            raise RuntimeError("Distance computation failed (not done or no solutions).")

        d = dist_shape_shape.Value()

        # --- Step 2: Inside/outside check ---
        classifier = BRepClass3d_SolidClassifier(shape_ocp)
        classifier.Perform(pnt, 1e-7)
        state = classifier.State()

        if state == TopAbs_IN:
            return -d
        elif state == TopAbs_ON:
            return 0.0
        else:  # TopAbs_OUT
            return d


    def fsdf(self, x: float, y: float, z: float) -> float:
        # calculates the signed distance form point (x, y, z) to the CadQuery shape boints inside the shape are negative
        pnt = gp_Pnt(x, y, z)
        vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex()
        shape_ocp = self._shape.wrapped
        dist_shape_shape = BRepExtrema_DistShapeShape(vertex, shape_ocp)
        dist_shape_shape.Perform()
        d = dist_shape_shape.Value()
        if d==0 : # Point is either on the surface or inside the shape
            # crate a large shape that goes into negative distance values
            world = cq.Workplane("XY").box(200000, 200000, 200000, centered=True)
            negativ = world.cut(cq.Workplane().newObject([self._shape])).val()
            negativ = cast(cq.Shape, negativ)
            negativ = negativ.wrapped
            dist_shape_shape = BRepExtrema_DistShapeShape(vertex, negativ)
            dist_shape_shape.Perform()
            d = - dist_shape_shape.Value()
        return d

    def sdf_at_pointgrid(self) -> list[float]:
        return self._sdf_at_pointgrid

    def pointgrid(self) -> list[list[float]]:
        return self._pointgrid
class BackupAssembly:
    def __init__(self):
        def m(value: float) -> float:
            return value * 1000.0
        outer_len, outer_wid, outer_ht = 2.0, 1.2, 0.8     # outer cuboid: 2.0 m × 1.2 m × 0.8 m
        inner_len, inner_wid, inner_ht = 1.2, 0.6, 0.4     # inner cuboid: 1.2 m × 0.6 m × 0.4 m
        inner_offset_m = (0.0, 0.0, 0.0)
        self._outer = cast(cq.Solid,
                           cq.Workplane("XY").box(m(outer_len), m(outer_wid), m(outer_ht), centered=True).val())
        self._inner = cast(cq.Solid,
            cq.Workplane("XY")
            .transformed(offset=(m(inner_offset_m[0]), m(inner_offset_m[1]), m(inner_offset_m[2])))
            .box(m(inner_len), m(inner_wid), m(inner_ht), centered=True)
            .val()
        )
    def outer_solid(self) -> cq.Shape:
        return self._outer
    def inner_solid(self) -> cq.Shape:
        return self._inner