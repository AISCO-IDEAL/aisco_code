from __future__ import annotations

from typing import Any

import numpy as np

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.custom_modes.bounding_box_mode_utils import (
    alpha_blend,
    get_mesh_from_optimization_state,
    load_bb_assembly_id,
    load_inner_outer_solids,
    overlay_edges,
    silhouette_edge_mask,
    solid_to_kaolin_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class OuterBoundingBoxMode(RenderingMode):
    def __init__(self, visualiser, config):
        super().__init__(visualiser, config)
        self.config = config
        self.completed = False
        self.color = tuple(config.get("color", [1.0, 0.35, 0.15]))
        self.outer_alpha = float(config.get("outer_alpha", 0.2))
        self.edge_color = tuple(config.get("edge_color", [0.0, 0.0, 0.0]))
        self.edge_strength = float(config.get("edge_strength", 1.0))
        self.sdf_color = tuple(config.get("sdf_color", [0.72, 0.72, 0.72]))

    def update(self, optimization_state: dict[str, Any]) -> None:
        if self.completed:
            return

        assembly_id = int(self.config.get("assembly_id", load_bb_assembly_id()))
        _, outer_solid = load_inner_outer_solids(assembly_id)

        tol = float(self.config.get("tolerance", 2.0))
        ang_tol = float(self.config.get("angular_tolerance", 0.4))
        vertices, faces = solid_to_kaolin_mesh(
            outer_solid,
            device=self.visualiser.device,
            tolerance=tol,
            angular_tolerance=ang_tol,
        )

        outer_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=faces.shape[0],
            color=self.color,
        )

        outer_rgb = self._render(
            vertices=vertices,
            faces=faces,
            textures=outer_textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        outer_sil = self._render(
            vertices=vertices,
            faces=faces,
            mode="silhouettes",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        outer_rgb_img = outer_rgb[0] if hasattr(outer_rgb, "__len__") and len(outer_rgb) > 0 else outer_rgb
        outer_mask = outer_sil[0] if hasattr(outer_sil, "__len__") and len(outer_sil) > 0 else outer_sil

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
            base_img = np.ones_like(outer_rgb_img, dtype=np.float32)

        composed = alpha_blend(base_img, outer_rgb_img, outer_mask.squeeze(-1), alpha=self.outer_alpha)
        edge_mask = silhouette_edge_mask(outer_mask.squeeze(-1))
        composed = overlay_edges(composed, edge_mask, edge_color=self.edge_color, strength=self.edge_strength)

        self.frame_storage.append(composed)
        self.completed = True

    def finalize(self, output_dir, file_name="outer_bb.png", delete: bool = False) -> None:
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frame to save")
            return

        out_path = output_dir / file_name
        self.visualiser.save_image(self.frame_storage[0], out_path)
        print(f"  Saved outer BB frame to {out_path}")

        if delete:
            self.clear_frames()
