"""
This Code is written by written by Clemens Huber
"""

from typing import cast

import cadquery as cq
import matplotlib.pyplot as plt
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_ON
from subprojects.wp2_assembler.version2.components.nbz import bounding_box_of_assembly, call_assembly_solids


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
        assembly_id: int = 1,
    ):
        if shape is None and shape_list is None:
            raise ValueError("Either shape or shape_list must be provided.")
        if shape is not None and shape_list is not None:
            raise ValueError("Only one of shape or shape_list must be provided.")
        self.__shape = shape
        self.__assembly_id = assembly_id
        self.__shape_list = shape_list
        self.__pointgrid = self.__pointgrid_geni(pointgrid_config)
        if self.__shape is None:
            l = len(self.__shape_list) if self.__shape_list is not None else 0  # noqa: E741
            self.__sdfs = []
            for x in self.__shape_list or []:
                # Pass shapes directly; no redundant casts
                self.__sdfs.append(SDFfromCadQuery(shape=x, pointgrid_config=pointgrid_config))
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
        inner_grid_edges = bounding_box_of_assembly(self.__assembly_id)
        print(inner_grid_edges)
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
        base_steps = (dx, dy, dz)
        fine_steps = (
            max(1, int(dx // 2)),
            max(1, int(dy // 2)),
            max(1, int(dz // 2)),
        )

        # Unpack inner grid edges: ((x_min, y_min, z_min), (x_max, y_max, z_max))
        (ix0, iy0, iz0), (ix1, iy1, iz1) = inner_grid_edges

        points: list[list[float]] = []
        seen = set()

        # Generate coarse grid across entire ranges
        for x in range(ranges[0][0], ranges[0][1], base_steps[0]):
            for y in range(ranges[1][0], ranges[1][1], base_steps[1]):
                for z in range(ranges[2][0], ranges[2][1], base_steps[2]):
                    # Skip points that fall inside inner box (they'll be covered by fine grid)
                    if ix0 <= x <= ix1 and iy0 <= y <= iy1 and iz0 <= z <= iz1:
                        continue
                    key = (x, y, z)
                    if key not in seen:
                        seen.add(key)
                        points.append([x, y, z])

        # Generate fine grid inside inner box with half step size (2x density per axis => 8x overall)
        # Clamp fine grid to overall ranges to avoid stepping outside
        x_start = max(ranges[0][0], int(ix0))
        x_end = min(ranges[0][1], int(ix1))
        y_start = max(ranges[1][0], int(iy0))
        y_end = min(ranges[1][1], int(iy1))
        z_start = max(ranges[2][0], int(iz0))
        z_end = min(ranges[2][1], int(iz1))

        for x in range(x_start, x_end, fine_steps[0]):
            for y in range(y_start, y_end, fine_steps[1]):
                for z in range(z_start, z_end, fine_steps[2]):
                    key = (x, y, z)
                    if key not in seen:
                        seen.add(key)
                        points.append([x, y, z])
        print(f"Pointgrid for BB loss sampels {len(points)} Points")

        return points

    def __sdf(self, x: float, y: float, z: float) -> float:
        # assumes self.__shape is a CadQuery Shape wrapping a closed solid
        assert self.__shape is not None, "Shape must be set when calling __sdf"
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
        assert self.__shape is not None, "Shape must be set when calling __sdf2"
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

    #### From here on, its just Code for testing and debuugging ########################################################
    def visualizr_slice(self, z: float) -> None:
        x_vals = sorted({p[0] for p in self.__pointgrid if p[2] == z})
        y_vals = sorted({p[1] for p in self.__pointgrid if p[2] == z})
        X, Y = np.meshgrid(x_vals, y_vals)  # noqa: N806
        # Create a mapping from coordinates to SDF values
        sdf_dict = {(p[0], p[1], p[2]): self.__sdf_at_pointgrid[i] for i, p in enumerate(self.__pointgrid)}
        Z = np.array([[sdf_dict.get((x, y, z), 0.0) for x in x_vals] for y in y_vals])  # noqa: N806

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(X, Y, Z, levels=10, cmap="RdBu_r")
        plt.colorbar(contour)
        plt.title(f"SDF Slice at z={z}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axis("equal")
        # save the figure
        plt.savefig(f"/cluster/home/clhuber/immages/sdf_slice_z_{z}.png")

    def plot_pointgrid_slice(self, z: float) -> None:
        # Collect actual (x, y) pairs for points with the given z
        pts = [(p[0], p[1]) for p in self.__pointgrid if p[2] == z]
        # Avoid zip strict issues by building lists directly
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, s=10)  # Scatter plot of (x,y) pairs
        plt.title(f"SDF Slice at z={z}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axis("equal")
        plt.savefig(f"/cluster/home/clhuber/immages/sdf_slice_z_{z}.png")


if __name__ == "__main__":
    if False:
        inner_box, outer_box = call_assembly_solids(2)
        sdf_inner = SDFfromCadQuery(shape_list=inner_box, pointgrid_config={"dx": 100, "dy": 50, "dz": 100, "rand": 0})
        sdf_inner.visualizr_slice(z=501)
    if True:
        inner_box, outer_box = call_assembly_solids(2)
        sdf_inner = SDFfromCadQuery(shape_list=inner_box, pointgrid_config={"dx": 100, "dy": 100, "dz": 100, "rand": 0})
        sdf_inner.plot_pointgrid_slice(z=501)
