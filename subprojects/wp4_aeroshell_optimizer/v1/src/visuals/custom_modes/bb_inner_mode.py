from __future__ import annotations

from typing import Any

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.custom_modes.bounding_box_mode_utils import (
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    solid_to_kaolin_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class InnerBoundingBoxMode(RenderingMode):
    def __init__(self, visualiser, config):
        super().__init__(visualiser, config)
        self.config = config
        self.completed = False
        self.color = tuple(config.get("color", [0.2, 0.7, 1.0]))

    def update(self, optimization_state: dict[str, Any]) -> None:  # noqa: ARG002
        if self.completed:
            return

        assembly_id = int(self.config.get("assembly_id", load_bb_assembly_id()))
        inner_solids, _ = load_inner_outer_solids(assembly_id)

        tol = float(self.config.get("tolerance", 2.0))
        ang_tol = float(self.config.get("angular_tolerance", 0.4))

        meshes = [
            solid_to_kaolin_mesh(s, device=self.visualiser.device, tolerance=tol, angular_tolerance=ang_tol) for s in inner_solids
        ]
        vertices, faces, __ = merge_meshes(meshes)

        textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=faces.shape[0],
            color=self.color,
        )

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

    def finalize(self, output_dir, file_name="inner_bb.png", delete: bool = False) -> None:
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frame to save")
            return

        out_path = output_dir / file_name
        self.visualiser.save_image(self.frame_storage[0], out_path)
        print(f"  Saved inner BB frame to {out_path}")

        if delete:
            self.clear_frames()
