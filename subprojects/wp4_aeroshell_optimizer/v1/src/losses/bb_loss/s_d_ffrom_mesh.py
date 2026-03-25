from typing import cast

import torch
import trimesh

from subprojects.wp2_assembler.version3.components.nbz import bounding_box_of_assembly as _default_bbox_fn
from subprojects.wp4_aeroshell_optimizer.v1.src.models.blendednet.gpu_sdf import mesh_sdf


class SDFfromMesh:
    """
    Mesh-backed SDF adapter with the same public API as SDFfromCadQuery.

    Pre Conditions:
    - mesh: trimesh.Trimesh

    Post Conditions:
        If you want to get the SDF value of all points in the pointgrid use sdf_at_pointgrid().
        If you want the pointgrid itself use pointgrid().
    """

    def __init__(
        self,
        pointgrid_config: dict,
        mesh: trimesh.Trimesh | None = None,
        mesh_list: list[trimesh.Trimesh] | None = None,
        assembly_id: int = 1,
        bounding_box_fn=None,
        sdf_device: str = "cpu",
    ):
        if mesh is None and mesh_list is None:
            raise ValueError("Either mesh or mesh_list must be provided.")
        if mesh is not None and mesh_list is not None:
            raise ValueError("Only one of mesh or mesh_list must be provided.")

        self.__mesh = mesh
        self.__mesh_list = mesh_list
        self.__assembly_id = assembly_id
        self.__bounding_box_fn = bounding_box_fn or _default_bbox_fn
        self.__sdf_device = torch.device(sdf_device)
        self.__pointgrid = self.__pointgrid_geni(pointgrid_config)

        if self.__mesh is None:
            l = len(self.__mesh_list) if self.__mesh_list is not None else 0  # noqa: E741
            self.__sdfs = []
            for m in self.__mesh_list or []:
                self.__sdfs.append(
                    SDFfromMesh(
                        mesh=m,
                        pointgrid_config=pointgrid_config,
                        assembly_id=assembly_id,
                        bounding_box_fn=bounding_box_fn,
                        sdf_device=sdf_device,
                    )
                )

            self.__sdf_at_pointgrid = [
                min(self.__sdfs[i].sdf_at_pointgrid()[j] for i in range(l)) for j in range(len(self.__pointgrid))
            ]
        else:
            mesh_cast = cast(trimesh.Trimesh, self.__mesh)
            if len(mesh_cast.faces) == 0:
                raise ValueError("Mesh has no faces, cannot compute SDF.")

            query = torch.tensor(self.__pointgrid, dtype=torch.float32, device=self.__sdf_device)
            vertices = torch.tensor(mesh_cast.vertices, dtype=torch.float32, device=self.__sdf_device)
            faces = torch.tensor(mesh_cast.faces, dtype=torch.long, device=self.__sdf_device)
            sdf_values = mesh_sdf(query=query, vertices=vertices, faces=faces)
            self.__sdf_at_pointgrid = sdf_values.detach().cpu().tolist()

    def __pointgrid_geni(self, pointgrid_config) -> list[list[float]]:
        inner_grid_edges = self.__bounding_box_fn(self.__assembly_id)
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

    def sdf_at_pointgrid(self) -> list[float]:
        return self.__sdf_at_pointgrid

    def pointgrid(self) -> list[list[float]]:
        return self.__pointgrid

    def volume(self) -> float:
        if self.__mesh is None:
            vol = 0.0
            for sdf in self.__sdfs:
                vol += sdf.volume()
            return vol
        return float(cast(trimesh.Trimesh, self.__mesh).volume)
