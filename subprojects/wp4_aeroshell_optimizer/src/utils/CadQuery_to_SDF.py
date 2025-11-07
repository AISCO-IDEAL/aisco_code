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
dr = 1
default_pointgrid_ranges = [
    [int(-5800 / 2 - rand + dr), int(5800 / 2 + rand)],
    [int(-1150 - rand + dr), int(1150 + rand)],
    [int(-rand+dr), int(1650 + rand)],
]
default_pointgrid_steps = [200, 200, 200]
default_pointgrid = [
    [x, y, z]
    for x in range(
        default_pointgrid_ranges[0][0],
        default_pointgrid_ranges[0][1],
        default_pointgrid_steps[0],
    )
    for y in range(
        default_pointgrid_ranges[1][0],
        default_pointgrid_ranges[1][1],
        default_pointgrid_steps[1],
    )
    for z in range(
        default_pointgrid_ranges[2][0],
        default_pointgrid_ranges[2][1],
        default_pointgrid_steps[2],
    )
]


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

    def __init__(
        self,
        shape: cq.Shape,
        pointgrid: list[list[float]] = default_pointgrid,
        in_mms: bool = True,
    ):
        self.__shape = shape
        self.__pointgrid = pointgrid
        self.__negativ = cq.Workplane("XY").box(100000, 100000, 100000, centered=True).cut(cq.Workplane().newObject([self.__shape])).val()
        self.__negativ = cast(cq.Shape, self.__negativ).wrapped
        self.__sdf_at_pointgrid = [self.sdf(x, y, z) for x, y, z in self.__pointgrid]
        self.__in_mms = in_mms

        if min(self.__sdf_at_pointgrid) > -1:
            self.__sdf_at_pointgrid = [self.sdf2(x, y, z) for x, y, z in self.__pointgrid]
        
    
    def sdf(self, x: float, y: float, z: float) -> float:
        # assumes self.__shape is a CadQuery Shape wrapping a closed solid
        shape = self.__shape.wrapped

        # classify sign
        pnt = gp_Pnt(x, y, z)
        classifier = BRepClass3d_SolidClassifier(shape)
        classifier.Perform(pnt, 1e-6)
        state = classifier.State()

        # unsigned distance
        vtx = BRepBuilderAPI_MakeVertex(pnt).Vertex()
        dss = BRepExtrema_DistShapeShape(vtx, shape)
        dss.Perform()
        if not dss.IsDone():
            return float("inf")  # or raise

        d = dss.Value()

        if state == TopAbs_ON:
            return 0.0
        elif state == TopAbs_IN:
            return -d
        else:
            return d

    def sdf2(self, x: float, y: float, z: float) -> float: # old but gold?
        # calculates the signed distance form point (x, y, z) to the CadQuery shape boints inside the shape are negative
        pnt = gp_Pnt(x, y, z)
        vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex()
        shape_ocp = self.__shape.wrapped
        dist_shape_shape = BRepExtrema_DistShapeShape(vertex, shape_ocp)
        dist_shape_shape.Perform()
        d = dist_shape_shape.Value()
        
        threshold = 0.1  
        
        if d < threshold:  # Point is either on the surface or inside the shape
            dist_shape_shape = BRepExtrema_DistShapeShape(vertex, self.__negativ)
            dist_shape_shape.Perform()
            d = -dist_shape_shape.Value()
            # print(f"Point ({x}, {y}, {z}) is inside the shape, SDF value: {d}")
        return d

    def sdf_at_pointgrid(self) -> list[float]:
        if self.in_mms:
            return self.__sdf_at_pointgrid
        else:
            return [s / 1000.0 for s in self.__sdf_at_pointgrid]  # convert to meters

    def pointgrid(self) -> list[list[float]]:
        if self.__in_mms:
            return self.__pointgrid
        else:
            return [[x / 1000.0, y / 1000.0, z / 1000.0] for x, y, z in self.__pointgrid]

    def volume(self) -> float:
        """Returns the volume of the CadQuery shape in cubic millimeters or cubic meters depending on the in_mms flag."""
        volume = self.__shape.Volume()
        if self.__in_mms:
            return volume  # in cubic millimeters
        else:
            return volume / 1e9  # convert to cubic meters


class BackupAssembly:
    def __init__(self):
        def m(value: float) -> float:
            return value * 1000.0

        outer_len, outer_wid, outer_ht = (
            2.0,
            1.2,
            0.8,
        )  # outer cuboid: 2.0 m × 1.2 m × 0.8 m
        inner_len, inner_wid, inner_ht = (
            1.2,
            0.6,
            0.4,
        )  # inner cuboid: 1.2 m × 0.6 m × 0.4 m
        inner_offset_m = (0.0, 0.0, 0.0)
        self._outer = cast(
            cq.Solid,
            cq.Workplane("XY")
            .box(m(outer_len), m(outer_wid), m(outer_ht), centered=True)
            .val(),
        )
        self._inner = cast(
            cq.Solid,
            cq.Workplane("XY")
            .transformed(
                offset=(
                    m(inner_offset_m[0]),
                    m(inner_offset_m[1]),
                    m(inner_offset_m[2]),
                )
            )
            .box(m(inner_len), m(inner_wid), m(inner_ht), centered=True)
            .val(),
        )

    def outer_solid(self) -> cq.Shape:
        return self._outer

    def inner_solid(self) -> cq.Shape:
        return self._inner
