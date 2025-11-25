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
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import (
    load_general_config,
    load_specific_config,
)


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
        shape: cq.Solid,
        training_config: dict,
    ):
        self.__shape = shape
        self.__pointgrid = self.__pointgrid_geni(training_config)
        self.__negativ = (
            cq.Workplane("XY")
            .box(100000, 100000, 100000, centered=True)
            .cut(cq.Workplane().newObject([self.__shape]))
            .val()
        )
        self.__negativ = cast(cq.Shape, self.__negativ).wrapped
        self.__sdf_at_pointgrid = [self.sdf(x, y, z) for x, y, z in self.__pointgrid]

        if min(self.__sdf_at_pointgrid) > -1:
            self.__sdf_at_pointgrid = [
                self.sdf2(x, y, z) for x, y, z in self.__pointgrid
            ]
        if min(self.__sdf_at_pointgrid) > -1:
            print("SDF from cq.Solid conversion Problem")

    def __pointgrid_geni(self, training_config) -> list[list[float]]:
        # load pointgrid params
        pointgrid_params = training_config["pointgrid_params"]
        dx, dy, dz, rand = (
            pointgrid_params.get("dx"),
            pointgrid_params.get("dy"),
            pointgrid_params.get("dz"),
            pointgrid_params.get("rand"),
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

    def sdf2(self, x: float, y: float, z: float) -> float:  # old but gold?
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
        return self.__sdf_at_pointgrid

    def pointgrid(self) -> list[list[float]]:
        return self.__pointgrid

    def volume(self) -> float:
        return self.__shape.Volume()
