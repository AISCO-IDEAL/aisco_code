import numpy as np
import torch
import trimesh

from subprojects.wp2_assembler.version4.components.nbz import call_assembly, max_assembly_id
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


def _as_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


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


def _sample_mesh_surface_points(mesh: trimesh.Trimesh, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    points, _face_idx = trimesh.sample.sample_surface(mesh, count)
    return points.astype(np.float32, copy=False)


class BbV44(ParentLoss):
    """Surface-sampled BB loss from WP2 inner/outer zero-level surfaces.

    Instead of evaluating over the full pointgrid volume, this loss samples points
    directly on the zero-level surfaces of inner and outer BB geometries.

    Constraints on sampled points:
    - inner surface points: sdf_model(p) <= -inner_margin
    - outer surface points: sdf_model(p) >= outer_margin
    """

    def _setup(self) -> None:
        self.__config = self.losses_config["bb_config"]
        self.__analytic_cfg = self.__config.get("analytic_options", {})
        self.__surface_cfg = self.__config.get("surface_sampling", {})

        ### fixed assembly ID ###
        if self.__config.get("random_assembly_id", "False") == "False":
            self.assembly_id = int(self.__config.get("assembly_id", 1))
        ### random assembly ID ###
        elif self.__config["random_assembly_id"] == "True":
            max_id = max_assembly_id()
            self.assembly_id = int(np.random.default_rng().integers(0, max_id + 1))
            if not 0 <= self.assembly_id <= max_id:
                raise ValueError(f"assembly_id must be in range 0..{max_id}, got {self.assembly_id}")
        ### else: config error ###
        else:
            raise ValueError(
                f"Invalid value for random_assembly_id: {self.__config['random_assembly_id']} (expected 'True' or 'False')"
            )

        print(f"[BbV43] Using assembly_id={self.assembly_id}")

        tess_tolerance = float(self.__config.get("tessellation_tolerance", 2.0))
        tess_angular_tolerance = float(self.__config.get("tessellation_angular_tolerance", 0.4))

        inner_assembly, outer_solid, __ = call_assembly(id_number=self.assembly_id)
        inner_solids = list(inner_assembly.toCompound().Solids())

        inner_meshes = [
            _solid_to_trimesh(s, tolerance=tess_tolerance, angular_tolerance=tess_angular_tolerance) for s in inner_solids
        ]
        outer_mesh = _solid_to_trimesh(
            outer_solid,
            tolerance=tess_tolerance,
            angular_tolerance=tess_angular_tolerance,
        )

        target_point_count = int(self.__surface_cfg.get("target_point_count", 20000))
        inner_fraction = float(self.__surface_cfg.get("inner_fraction", 0.5))
        inner_fraction = float(np.clip(inner_fraction, 0.05, 0.95))
        min_points_per_inner_mesh = int(self.__surface_cfg.get("min_points_per_inner_mesh", 256))

        inner_total = max(1, int(target_point_count * inner_fraction))
        outer_total = max(1, target_point_count - inner_total)

        inner_areas = np.array([max(float(m.area), 1e-6) for m in inner_meshes], dtype=np.float64)
        inner_weights = inner_areas / inner_areas.sum()
        inner_counts = np.maximum(
            min_points_per_inner_mesh,
            np.floor(inner_total * inner_weights).astype(int),
        )

        deficit = inner_total - int(inner_counts.sum())
        if deficit > 0:
            order = np.argsort(-inner_weights)
            for idx in order[:deficit]:
                inner_counts[idx] += 1

        inner_points_np = np.concatenate(
            [_sample_mesh_surface_points(m, int(c)) for m, c in zip(inner_meshes, inner_counts, strict=False)], axis=0
        )
        outer_points_np = _sample_mesh_surface_points(outer_mesh, outer_total)

        self.__inner_surface_points = torch.from_numpy(inner_points_np)
        self.__outer_surface_points = torch.from_numpy(outer_points_np)

        self.__inner_margin = float(self.__analytic_cfg.get("inner_margin", 0.0))
        self.__outer_margin = float(self.__analytic_cfg.get("outer_margin", 0.0))
        self.__penalty = str(self.__analytic_cfg.get("penalty", "relu")).lower()
        self.__softplus_beta = float(self.__analytic_cfg.get("softplus_beta", 10.0))

        print(
            f"[BbV44] Surface samples: inner={self.__inner_surface_points.shape[0]}, outer={self.__outer_surface_points.shape[0]}"
        )

    def _violation(self, value: torch.Tensor) -> torch.Tensor:
        if self.__penalty == "softplus":
            return torch.nn.functional.softplus(self.__softplus_beta * value) / self.__softplus_beta
        return torch.relu(value)

    def _loss(self, model) -> torch.Tensor:
        with torch.enable_grad():
            target_device = next(model.parameters()).device
            self.device = target_device

            inner_points = self.__inner_surface_points.to(self.device)
            outer_points = self.__outer_surface_points.to(self.device)

            inner_out = model(inner_points)
            outer_out = model(outer_points)

            sdf_inner_pred = inner_out[0] if isinstance(inner_out, tuple) else inner_out
            sdf_outer_pred = outer_out[0] if isinstance(outer_out, tuple) else outer_out

            sdf_inner_pred = sdf_inner_pred.squeeze(-1)
            sdf_outer_pred = sdf_outer_pred.squeeze(-1)

            # inner surface target: model SDF should be <= -inner_margin
            upper_violation = self._violation(sdf_inner_pred + self.__inner_margin)
            # outer surface target: model SDF should be >= outer_margin
            lower_violation = self._violation(self.__outer_margin - sdf_outer_pred)

            return 0.5 * (upper_violation.mean() + lower_violation.mean())
