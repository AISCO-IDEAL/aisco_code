from __future__ import annotations

from typing import Any

import numpy as np

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.custom_modes.bounding_box_mode_utils import (
    alpha_blend,
    get_mesh_from_optimization_state,
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    overlay_edges,
    silhouette_edge_mask,
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
        self.sdf_color = tuple(config.get("sdf_color", [0.72, 0.72, 0.72]))
        self.inner_alpha = float(config.get("inner_alpha", 0.4))
        self.outer_alpha = float(config.get("outer_alpha", 0.18))
        self.edge_color = tuple(config.get("edge_color", [0.0, 0.0, 0.0]))
        self.edge_strength = float(config.get("edge_strength", 1.0))

    def update(self, optimization_state: dict[str, Any]) -> None:
        if self.completed:
            return

        assembly_id = int(self.config.get("assembly_id", load_bb_assembly_id()))
        inner_solids, outer_solid = load_inner_outer_solids(assembly_id)

        tol = float(self.config.get("tolerance", 2.0))
        ang_tol = float(self.config.get("angular_tolerance", 0.4))

        outer_vertices, outer_faces = solid_to_kaolin_mesh(
            outer_solid,
            device=self.visualiser.device,
            tolerance=tol,
            angular_tolerance=ang_tol,
        )
        inner_meshes = [
            solid_to_kaolin_mesh(s, device=self.visualiser.device, tolerance=tol, angular_tolerance=ang_tol) for s in inner_solids
        ]
        inner_vertices_t, inner_faces_t, __ = merge_meshes(inner_meshes)

        outer_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=outer_faces.shape[0],
            color=self.outer_color,
        )
        inner_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=inner_faces_t.shape[0],
            color=self.inner_color,
        )

        outer_rgb = self._render(
            vertices=outer_vertices,
            faces=outer_faces,
            textures=outer_textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        outer_sil = self._render(
            vertices=outer_vertices,
            faces=outer_faces,
            mode="silhouettes",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        inner_rgb = self._render(
            vertices=inner_vertices_t,
            faces=inner_faces_t,
            textures=inner_textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        inner_sil = self._render(
            vertices=inner_vertices_t,
            faces=inner_faces_t,
            mode="silhouettes",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        outer_rgb_img = outer_rgb[0] if hasattr(outer_rgb, "__len__") and len(outer_rgb) > 0 else outer_rgb
        outer_mask = outer_sil[0] if hasattr(outer_sil, "__len__") and len(outer_sil) > 0 else outer_sil
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
            composed = sdf_rgb[0] if hasattr(sdf_rgb, "__len__") and len(sdf_rgb) > 0 else sdf_rgb
        else:
            composed = np.ones_like(outer_rgb_img, dtype=np.float32)

        composed = alpha_blend(composed, inner_rgb_img, inner_mask.squeeze(-1), alpha=self.inner_alpha)
        composed = alpha_blend(composed, outer_rgb_img, outer_mask.squeeze(-1), alpha=self.outer_alpha)
        edge_mask = silhouette_edge_mask(outer_mask.squeeze(-1))
        composed = overlay_edges(composed, edge_mask, edge_color=self.edge_color, strength=self.edge_strength)

        self.frame_storage.append(composed)
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
