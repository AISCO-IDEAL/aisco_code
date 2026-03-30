from __future__ import annotations

from typing import Any

import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.custom_modes.bounding_box_mode_utils import (
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    solid_to_kaolin_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class BoundingBoxesCombinedMode(RenderingMode):
    def __init__(self, visualiser, config):
        super().__init__(visualiser, config)
        self.config = config
        self.completed = False
        self.inner_color = tuple(config.get("inner_color", [0.2, 0.7, 1.0]))
        self.outer_color = tuple(config.get("outer_color", [1.0, 0.35, 0.15]))

    def update(self, optimization_state: dict[str, Any]) -> None:  # noqa: ARG002
        if self.completed:
            return

        assembly_id = int(self.config.get("assembly_id", load_bb_assembly_id()))
        inner_solids, outer_solid = load_inner_outer_solids(assembly_id)

        tol = float(self.config.get("tolerance", 2.0))
        ang_tol = float(self.config.get("angular_tolerance", 0.4))

        outer_mesh = solid_to_kaolin_mesh(outer_solid, device=self.visualiser.device, tolerance=tol, angular_tolerance=ang_tol)
        inner_meshes = [
            solid_to_kaolin_mesh(s, device=self.visualiser.device, tolerance=tol, angular_tolerance=ang_tol) for s in inner_solids
        ]

        vertices, faces, counts = merge_meshes([outer_mesh, *inner_meshes])

        face_colors = torch.zeros((faces.shape[0], 3), dtype=torch.float32, device=torch.device(self.visualiser.device))
        outer_faces = counts[0]
        face_colors[:outer_faces] = torch.tensor(self.outer_color, dtype=torch.float32, device=face_colors.device)
        face_colors[outer_faces:] = torch.tensor(self.inner_color, dtype=torch.float32, device=face_colors.device)

        textures = self.visualiser.face_colors_to_textures(face_colors)

        frame = self._render(
            vertices=vertices,
            faces=faces,
            textures=textures,
            mode=self.render_mode,
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        self.frame_storage.append(frame[0] if hasattr(frame, "__len__") and len(frame) > 0 else frame)
        self.completed = True

    def finalize(self, output_dir, file_name="inner_outer_bb.png", delete: bool = False) -> None:
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frame to save")
            return

        out_path = output_dir / file_name
        self.visualiser.save_image(self.frame_storage[0], out_path)
        print(f"  Saved combined BB frame to {out_path}")

        if delete:
            self.clear_frames()
