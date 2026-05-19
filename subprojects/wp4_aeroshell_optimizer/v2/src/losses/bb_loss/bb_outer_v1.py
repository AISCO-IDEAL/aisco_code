"""Outer BB loss with geometric inward offset and separate bottom-face control.

Treats the outer NBZ as if it were shrunk along its surface normals by
outer_inward_offset_mm, so the aeroshell's SDF=0 surface is pushed further inside
the real regulatory envelope. The flat bottom face is offset by outer_bottom_offset_mm
instead, separate because raising the bottom can collide with the wheel_interface
contact-patch constraint and with the inflated inner solid (seat bottom is only
seat_z − ground_clearance ≈ 11mm above the outer NBZ floor).

Face classification (per-sample, from face-normal dot product with -ẑ):
- bottom face: normal ≈ (0, 0, -1), dot product > threshold → outer_bottom_offset_mm
- approach/departure ramps: normal ≈ (±sin10°, 0, -cos10°), dot product ≈ 0.98 → side
- top + ±y walls: dot product ≤ 0 → side
"""

import numpy as np
import torch
import trimesh

from subprojects.wp2_assembler.version4.components.nbz import call_assembly
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_outer_v0 import BbOuterV0
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v4_4 import _solid_to_trimesh
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.assembly_id_utils import resolve_assembly_id


class BbOuterV1(BbOuterV0):
    def _setup(self) -> None:
        super()._setup()

        bb_config: dict = self.losses_config["bb_config"]
        sampling_cfg: dict = bb_config.get("surface_sampling", {})
        side_offset_mm: float = float(sampling_cfg.get("outer_inward_offset_mm", 0.0))
        bottom_offset_mm: float = float(sampling_cfg.get("outer_bottom_offset_mm", 0.0))
        bottom_thresh: float = float(sampling_cfg.get("mask_downward_normal_threshold", 0.99))
        target_point_count: int = int(sampling_cfg.get("target_point_count", 20000))
        inner_fraction: float = float(np.clip(float(sampling_cfg.get("inner_fraction", 0.5)), 0.05, 0.95))

        tess_tol: float = float(bb_config.get("tessellation_tolerance", 2.0))
        tess_ang_tol: float = float(bb_config.get("tessellation_angular_tolerance", 0.4))
        assembly_id: int = resolve_assembly_id(bb_config)

        # Re-load outer mesh to capture per-sample face normals (BbV44 discards them).
        _inner_assembly, outer_solid, _ = call_assembly(id_number=assembly_id)
        outer_mesh: trimesh.Trimesh = _solid_to_trimesh(outer_solid, tolerance=tess_tol, angular_tolerance=tess_ang_tol)

        outer_total = max(1, target_point_count - max(1, int(target_point_count * inner_fraction)))
        points, face_idx = trimesh.sample.sample_surface(outer_mesh, outer_total)
        points = np.asarray(points, dtype=np.float32)
        normals = outer_mesh.face_normals[face_idx].astype(np.float32)

        downward_alignment = -normals[:, 2]
        is_bottom = downward_alignment > bottom_thresh

        # Inward offset = -normal * offset (opposite of outward normal).
        per_sample_offset = np.where(is_bottom, bottom_offset_mm, side_offset_mm).astype(np.float32)
        offset_points = points - normals * per_sample_offset[:, None]

        self._outer_surface_points_raw: torch.Tensor = torch.from_numpy(points.copy())
        self._outer_is_bottom_mask: torch.Tensor = torch.from_numpy(is_bottom.copy())
        self._outer_surface_points = torch.from_numpy(offset_points)

        n_bottom = int(is_bottom.sum())
        print(
            f"[BbOuterV1] inward offsets: sides=-{side_offset_mm}mm, bottom=-{bottom_offset_mm}mm "
            f"({n_bottom} bottom / {points.shape[0]} total samples)"
        )
