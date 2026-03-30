from __future__ import annotations

from typing import Any

import numpy as np

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.custom_modes.bounding_box_mode_utils import (
    alpha_blend,
    get_mesh_from_optimization_state,
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
        self.inner_alpha = float(config.get("inner_alpha", 0.45))
        self.sdf_color = tuple(config.get("sdf_color", [0.72, 0.72, 0.72]))

    def update(self, optimization_state: dict[str, Any]) -> None:
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

        inner_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=faces.shape[0],
            color=self.color,
        )

        inner_rgb = self._render(
            vertices=vertices,
            faces=faces,
            textures=inner_textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        inner_sil = self._render(
            vertices=vertices,
            faces=faces,
            mode="silhouettes",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        inner_rgb_img = inner_rgb[0] if hasattr(inner_rgb, "__len__") and len(inner_rgb) > 0 else inner_rgb
        inner_mask = inner_sil[0] if hasattr(inner_sil, "__len__") and len(inner_sil) > 0 else inner_sil

        sdf_vertices, sdf_faces = get_mesh_from_optimization_state(optimization_state)
        if sdf_vertices is not None and sdf_faces is not None and int(sdf_faces.numel()) > 0:
            sdf_textures = self.visualiser.create_solid_texture(
                batch_size=1,
                num_faces=sdf_faces.shape[0],
                color=self.sdf_color,
            )
            sdf_rgb = self._render(
                vertices=sdf_vertices,
                faces=sdf_faces,
                textures=sdf_textures,
                mode="rgb",
                camera_position=self.camera_position,
                camera_direction=self.camera_direction,
            )
            base_img = sdf_rgb[0] if hasattr(sdf_rgb, "__len__") and len(sdf_rgb) > 0 else sdf_rgb
        else:
            base_img = np.ones_like(inner_rgb_img, dtype=np.float32)

        composed = alpha_blend(base_img, inner_rgb_img, inner_mask.squeeze(-1), alpha=self.inner_alpha)

        self.frame_storage.append(composed)
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
