import os

import numpy as np
import torch
import trimesh

from subprojects.wp2_assembler.version4.components.nbz import bounding_box_of_assembly, call_assembly, max_assembly_id
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.bb_loss.s_d_ffrom_mesh import SDFfromMesh
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


def _solid_to_trimesh(solid, tolerance: float, angular_tolerance: float) -> trimesh.Trimesh:
    vertices_cq, faces_cq = solid.tessellate(tolerance, angular_tolerance)
    vertices_np = np.array([[v.x, v.y, v.z] for v in vertices_cq], dtype=np.float32)
    faces_np = np.array(faces_cq, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0:
        raise ValueError("Tessellated mesh has no faces.")
    trimesh.repair.fix_normals(mesh, multibody=False)
    return mesh


class BbV42(ParentLoss):
    """
    Fast BB loss that uses in-memory CadQuery tessellation + mesh-backed SDF.

    No marching cubes and no path-based PLY loading are used. Inner/outer boxes
    are obtained directly from WP2 assembler v4.
    """

    def _setup(self) -> None:
        self.__config = self.losses_config["bb_config"]

        env_assembly_id = os.getenv("AISCO_ASSEMBLY_ID", "").strip()
        configured_assembly_id = self.__config.get("assembly_id")

        if env_assembly_id:
            self.assembly_id = int(env_assembly_id)
        elif configured_assembly_id is not None:
            self.assembly_id = int(configured_assembly_id)
        else:
            rng = np.random.default_rng()
            self.assembly_id = int(rng.integers(0, max_assembly_id() + 1))

        max_id = max_assembly_id()
        if not 0 <= self.assembly_id <= max_id:
            msg = f"assembly_id must be in range 0..{max_id}, got {self.assembly_id}"
            raise ValueError(msg)

        print(f"[BbV42] Using assembly_id={self.assembly_id}")

        inner_box, outer_box, __ = call_assembly(id_number=self.assembly_id)
        inner_solids = list(inner_box.toCompound().Solids())

        tess_tolerance = float(self.__config.get("tessellation_tolerance", 2.0))
        tess_angular_tolerance = float(self.__config.get("tessellation_angular_tolerance", 0.4))
        sdf_device = str(self.__config.get("sdf_device", "cpu"))
        sdf_chunk_size = int(self.__config.get("sdf_chunk_size", 4096))

        inner_mesh_list = [
            _solid_to_trimesh(s, tolerance=tess_tolerance, angular_tolerance=tess_angular_tolerance) for s in inner_solids
        ]
        outer_mesh = _solid_to_trimesh(
            outer_box,
            tolerance=tess_tolerance,
            angular_tolerance=tess_angular_tolerance,
        )

        self.__inner_sdf: SDFfromMesh = SDFfromMesh(
            mesh_list=inner_mesh_list,
            pointgrid_config=self.__config["pointgrid_params"],
            assembly_id=self.assembly_id,
            bounding_box_fn=bounding_box_of_assembly,
            sdf_device=sdf_device,
            sdf_chunk_size=sdf_chunk_size,
        )
        self.__outer_sdf: SDFfromMesh = SDFfromMesh(
            mesh=outer_mesh,
            pointgrid_config=self.__config["pointgrid_params"],
            assembly_id=self.assembly_id,
            bounding_box_fn=bounding_box_of_assembly,
            sdf_device=sdf_device,
            sdf_chunk_size=sdf_chunk_size,
        )

        self.__pointgrid: list[list[float]] = self.__inner_sdf.pointgrid()
        self.__inner_sdf_values: list[float] = self.__inner_sdf.sdf_at_pointgrid()
        self.__outer_sdf_values: list[float] = self.__outer_sdf.sdf_at_pointgrid()

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            p = torch.tensor(self.__pointgrid, dtype=torch.float32, device=self.device)
            sdf_inner_t = torch.as_tensor(self.__inner_sdf_values, dtype=torch.float32, device=self.device)
            sdf_outer_t = torch.as_tensor(self.__outer_sdf_values, dtype=torch.float32, device=self.device)

            output = model(p)
            sdf_pred = output[0] if isinstance(output, tuple) else output
            sdf_pred = sdf_pred.squeeze(-1)

            upper_violation = torch.relu(sdf_pred - sdf_inner_t)
            lower_violation = torch.relu(sdf_outer_t - sdf_pred)

            return (upper_violation + lower_violation).mean()
