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


class SDFfromCadQuery:
    """
    Pre Conditions:
    - shape: CadQuery Shape: cq.Solid

    Post Conditions:
        If you want to get the SDF value of all Points in the pointgrid use sdf_at_pointgrid() to get the list of the corresponding SDF values
        If you want the SDF value of a single point use sdf(x,y,z) it returns just one value
    """

    def __init__(
        self,
        pointgrid_config: dict,
        shape: cq.Solid | None = None,
        shape_list: list[cq.Solid] | None = None,
    ):
        if shape is None and shape_list is None:
            raise ValueError("Either shape or shape_list must be provided.")  # noqa: TRY003
        if shape is not None and shape_list is not None:
            raise ValueError("Only one of shape or shape_list must be provided.")  # noqa: TRY003
        self.__shape = shape
        self.__shape_list = shape_list
        self.__pointgrid = self.__pointgrid_geni(pointgrid_config)
        if self.__shape is None:
            l = len(self.__shape_list) if self.__shape_list is not None else 0  # noqa: E741
            self.__sdfs = []
            for x in self.__shape_list or []:
                self.__sdfs.append(SDFfromCadQuery(shape=cast(cq.Solid, x), pointgrid_config=pointgrid_config))
            self.__sdf_at_pointgrid = [
                min(self.__sdfs[i].sdf_at_pointgrid()[j] for i in range(l))
                for j in range(len(self.__pointgrid))  # From all shapes take the minimum SDF at each point
            ]
        else:
            self.__negativ = (
                cq.Workplane("XY")
                .box(100000, 100000, 100000, centered=True)
                .cut(cq.Workplane().newObject([cast(cq.Shape, self.__shape)]))
                .val()
            )
            self.__negativ = cast(cq.Shape, self.__negativ).wrapped
            self.__sdf_at_pointgrid = [self.__sdf(x, y, z) for x, y, z in self.__pointgrid]

            if min(self.__sdf_at_pointgrid) > -1:
                self.__sdf_at_pointgrid = [self.__sdf2(x, y, z) for x, y, z in self.__pointgrid]
            if min(self.__sdf_at_pointgrid) > -1:
                print("SDF from cq.Solid conversion Problem")

    def __pointgrid_geni(self, pointgrid_config) -> list[list[float]]:
        dx, dy, dz, rand = (
            pointgrid_config.get("dx"),
            pointgrid_config.get("dy"),
            pointgrid_config.get("dz"),
            pointgrid_config.get("rand"),
        )
        dr = 1
        ranges = [
            (int(-2900 - rand + dr), int(2900 + rand)),
            (int(-1150 - rand + dr), int(1150 + rand)),
            (int(-rand + dr), int(1650 + rand)),
        ]
        steps = (dx, dy, dz)

        return [
            [x, y, z]
            for x in range(ranges[0][0], ranges[0][1], steps[0])
            for y in range(ranges[1][0], ranges[1][1], steps[1])
            for z in range(ranges[2][0], ranges[2][1], steps[2])
        ]

    def __sdf(self, x: float, y: float, z: float) -> float:
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

    def __sdf2(self, x: float, y: float, z: float) -> float:  # old but gold?
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
        return d

    def sdf_at_pointgrid(self) -> list[float]:
        return self.__sdf_at_pointgrid

    def pointgrid(self) -> list[list[float]]:
        return self.__pointgrid

    def volume(self) -> float:
        if self.__shape is None:
            vol = 0.0
            for sdf in self.__sdfs:
                vol += sdf.volume()
            return vol
        else:
            return self.__shape.Volume()
