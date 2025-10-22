"""
In this code, there is a way of transfering a CadQuery shape to a SDF representation using a neural network.
"""
from typing import cast

import cadquery as cq
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex  # type: ignore  # noqa: PGH003
from OCP.BRepExtrema import BRepExtrema_DistShapeShape  # type: ignore  # noqa: PGH003
from OCP.gp import gp_Pnt  # type: ignore  # noqa: PGH003


class SDFfromCadQuery:
    def __init__(self, shape: cq.Shape, pointgrid:list[list[float]]=[[x/10, y/10, z/10] for x in range(-10, 11) for y in range(-10, 11) for z in range(-10, 11)]):  # noqa: B006
        self._shape = shape
        self._pointgrid = pointgrid
        self._sdf_at_pointgrid = [self.sdf(x, y, z) for x, y, z in self._pointgrid]
        
    def sdf(self, x: float, y: float, z: float) -> float:
        # calculates the signed distance form point (x, y, z) to the CadQuery shape boints inside the shape are negative
        pnt = gp_Pnt(x, y, z) 
        vertex = BRepBuilderAPI_MakeVertex(pnt).Vertex() 
        shape_ocp = self._shape.wrapped 
        dist_shape_shape = BRepExtrema_DistShapeShape(vertex, shape_ocp) 
        dist_shape_shape.Perform() 
        d = dist_shape_shape.Value() # check if point is inside the shape 
        if d==0:
            world = cq.Workplane("XY").box(100, 100, 100)
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
