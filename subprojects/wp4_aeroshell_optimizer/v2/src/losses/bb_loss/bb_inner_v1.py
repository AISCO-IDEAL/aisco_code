"""Inner BB loss with geometric outward offset and wheel-bottom-cap masking.

Treats the inner solids as if they were inflated along their surface normals by
inner_outward_offset_mm. The aeroshell SDF=0 surface must then stay outside this
inflated geometry — leaving a physical clearance equal to (offset - residual loss)
between the real inner assembly and the trained aeroshell, so small training
residuals no longer cause visible clipping.

The artificial flat bottom cap of each wheel (created by the wheel_clearance cut
in WP2) is masked out: its outward normal is `-ẑ`, so naive offset would push
samples below ground_clearance and collide with the wheel_interface constraint.
The seat bottom (at z≈wheel_clearance+1mm) is NOT masked — it is a real surface
responsible for clipping and must remain in the loss.
"""

import numpy as np
import torch
import trimesh

from subprojects.wp2_assembler.version4.components.nbz import call_assembly
from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_inner_v0 import BbInnerV0
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import _solid_to_trimesh
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.assembly_id_utils import resolve_assembly_id


class BbInnerV1(BbInnerV0):
    def _setup(self) -> None:
        super()._setup()

        bb_config: dict = self.losses_config["bb_config"]
        sampling_cfg: dict = bb_config.get("surface_sampling", {})
        offset_mm: float = float(sampling_cfg.get("inner_outward_offset_mm", 0.0))
        mask_thresh: float = float(sampling_cfg.get("mask_downward_normal_threshold", 0.99))
        cap_z_tol_mm: float = float(sampling_cfg.get("cap_z_tolerance_mm", 0.5))
        target_point_count: int = int(sampling_cfg.get("target_point_count", 20000))
        inner_fraction: float = float(np.clip(float(sampling_cfg.get("inner_fraction", 0.5)), 0.05, 0.95))
        min_per_mesh: int = int(sampling_cfg.get("min_points_per_inner_mesh", 256))

        wheel_clearance_mm: float = float(get_regulation("no_build_zones.outer_prism.wheel_clearance"))

        tess_tol: float = float(bb_config.get("tessellation_tolerance", 2.0))
        tess_ang_tol: float = float(bb_config.get("tessellation_angular_tolerance", 0.4))
        assembly_id: int = resolve_assembly_id(bb_config)

        # Re-load inner meshes; super()._setup already sampled, but without face normals.
        # The double mesh-load is a one-time setup cost — clarity over micro-optimisation.
        inner_assembly, _outer_solid, _ = call_assembly(id_number=assembly_id)
        inner_solids = list(inner_assembly.toCompound().Solids())
        inner_meshes: list[trimesh.Trimesh] = [
            _solid_to_trimesh(s, tolerance=tess_tol, angular_tolerance=tess_ang_tol) for s in inner_solids
        ]

        inner_total = max(1, int(target_point_count * inner_fraction))
        inner_areas = np.array([max(float(m.area), 1e-6) for m in inner_meshes], dtype=np.float64)
        inner_weights = inner_areas / inner_areas.sum()
        inner_counts = np.maximum(min_per_mesh, np.floor(inner_total * inner_weights).astype(int))
        deficit = inner_total - int(inner_counts.sum())
        if deficit > 0:
            order = np.argsort(-inner_weights)
            for idx in order[:deficit]:
                inner_counts[idx] += 1

        kept_raw_chunks: list[np.ndarray] = []
        kept_offset_chunks: list[np.ndarray] = []
        masked_chunks: list[np.ndarray] = []
        total_sampled = 0
        for mesh, count in zip(inner_meshes, inner_counts, strict=False):
            count = int(count)  # noqa: PLW2901
            if count <= 0:
                continue
            points, face_idx = trimesh.sample.sample_surface(mesh, count)
            points = np.asarray(points, dtype=np.float32)
            normals = mesh.face_normals[face_idx].astype(np.float32)
            centroids_z = mesh.triangles[face_idx][..., 2].mean(axis=-1).astype(np.float32)

            # Wheel bottom cap: nearly-pure downward normal AND face centroid at z ≈ wheel_clearance.
            # The seat bottom at z ≈ wheel_clearance + 1mm fails the z test and is kept.
            downward_alignment = -normals[:, 2]  # = normal · (-ẑ)
            is_wheel_bottom_cap = (downward_alignment > mask_thresh) & (centroids_z < wheel_clearance_mm + cap_z_tol_mm)
            keep = ~is_wheel_bottom_cap

            kept_raw_chunks.append(points[keep])
            kept_offset_chunks.append(points[keep] + normals[keep] * offset_mm)
            masked_chunks.append(points[~keep])
            total_sampled += count

        raw_np = np.concatenate(kept_raw_chunks, axis=0) if kept_raw_chunks else np.zeros((0, 3), dtype=np.float32)
        offset_np = np.concatenate(kept_offset_chunks, axis=0) if kept_offset_chunks else np.zeros((0, 3), dtype=np.float32)
        masked_np = np.concatenate(masked_chunks, axis=0) if masked_chunks else np.zeros((0, 3), dtype=np.float32)

        # Loss reads _inner_surface_points; raw + masked are kept for the offset visualizer.
        self._inner_surface_points_raw: torch.Tensor = torch.from_numpy(raw_np)
        self._inner_masked_out_points: torch.Tensor = torch.from_numpy(masked_np)
        self._inner_surface_points = torch.from_numpy(offset_np)

        print(
            f"[BbInnerV1] outward offset=+{offset_mm}mm, masked {masked_np.shape[0]}/{total_sampled} "
            f"wheel-bottom-cap samples, loss-cloud size={offset_np.shape[0]}"
        )
