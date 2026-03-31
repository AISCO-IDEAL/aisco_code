from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (
    alpha_blend,
    get_mesh_from_optimization_state,
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    overlay_edges,
    silhouette_edge_mask,
    solid_to_kaolin_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.rendering_mode_base_class import RenderingMode


class MeshAssemblyModeV2(RenderingMode):
    """
    Stable mesh-assembly renderer:
    - Uses optimization_state vertices/faces for SDF mesh (same stable path as mesh_assembly mode).
    - Renders inner assembly + outer BB once as static background.
    - Overlays SDF mesh every render_interval and stores frames for GIF.
    """

    def __init__(self, visualiser, config):
        super().__init__(visualiser, config)
        self.config = config
        self.base_frame = None
        self.base_assembly_id: int | None = None
        self.frame_storage = []
        self.render_interval = int(config.get("render_interval", 1))
        self.make_gif = bool(config.get("make_gif", True))

        self.mesh_color = tuple(config.get("mesh_color", [0.2, 1.0, 1.0]))
        self.mesh_alpha = float(config.get("mesh_alpha", config.get("alpha", 0.5)))

        self.assembly_color = tuple(config.get("assembly_color", [1.0, 0.55, 0.0]))
        self.outer_color = tuple(config.get("outer_color", [1.0, 0.35, 0.15]))
        self.outer_alpha = float(config.get("outer_alpha", 0.2))
        self.outer_edge_color = tuple(config.get("outer_edge_color", [0.0, 0.0, 0.0]))
        self.outer_edge_strength = float(config.get("outer_edge_strength", 1.0))

        self.tolerance = float(config.get("tolerance", 2.0))
        self.angular_tolerance = float(config.get("angular_tolerance", 0.4))

    def _annotate_frame(self, frame: np.ndarray, epoch_value: Any, sdf_ok: bool) -> np.ndarray:
        img_uint8 = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        draw = ImageDraw.Draw(pil_img)

        text = f"epoch: {epoch_value} | sdf: {'ok' if sdf_ok else 'missing'}"
        x = 10
        y = max(0, pil_img.height - 26)
        text_width = max(1, int(len(text) * 7.0))
        text_bg_box: tuple[float, float, float, float] = (
            float(x - 4),
            float(y - 3),
            float(x + text_width + 6),
            float(y + 16),
        )
        draw.rectangle(text_bg_box, fill=(0, 0, 0))
        draw.text((x, y), text, fill=(255, 255, 255))

        return np.asarray(pil_img).astype(np.float32) / 255.0

    def _resolve_assembly_id(self, optimization_state: dict[str, Any] | None = None) -> int:
        if optimization_state is not None:
            runtime_assembly_id = optimization_state.get("assembly_id")
            if runtime_assembly_id is None:
                runtime_assembly_id = optimization_state.get("bb_assembly_id")
            if runtime_assembly_id is not None:
                return int(runtime_assembly_id)

        configured_assembly_id = self.config.get("assembly_id")
        if configured_assembly_id is not None:
            return int(configured_assembly_id)

        return load_bb_assembly_id()

    def _build_base_frame(self, assembly_id: int) -> np.ndarray:
        inner_solids, outer_solid = load_inner_outer_solids(assembly_id)

        inner_meshes = [
            solid_to_kaolin_mesh(
                solid=s,
                device=self.visualiser.device,
                tolerance=self.tolerance,
                angular_tolerance=self.angular_tolerance,
            )
            for s in inner_solids
        ]
        inner_vertices, inner_faces, __ = merge_meshes(inner_meshes)

        outer_vertices, outer_faces = solid_to_kaolin_mesh(
            solid=outer_solid,
            device=self.visualiser.device,
            tolerance=self.tolerance,
            angular_tolerance=self.angular_tolerance,
        )

        inner_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=inner_faces.shape[0],
            color=self.assembly_color,
        )
        outer_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=outer_faces.shape[0],
            color=self.outer_color,
        )

        inner_rgb = self._render(
            vertices=inner_vertices,
            faces=inner_faces,
            textures=inner_textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
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

        inner_img = inner_rgb[0] if hasattr(inner_rgb, "__len__") and len(inner_rgb) > 0 else inner_rgb
        outer_img = outer_rgb[0] if hasattr(outer_rgb, "__len__") and len(outer_rgb) > 0 else outer_rgb
        outer_mask = outer_sil[0] if hasattr(outer_sil, "__len__") and len(outer_sil) > 0 else outer_sil

        base = alpha_blend(inner_img, outer_img, outer_mask.squeeze(-1), alpha=self.outer_alpha)
        edge_mask = silhouette_edge_mask(outer_mask.squeeze(-1))
        base = overlay_edges(base, edge_mask, edge_color=self.outer_edge_color, strength=self.outer_edge_strength)
        return base

    def update(self, optimization_state: dict[str, Any]) -> None:
        iteration = optimization_state.get("iteration")
        if iteration is not None and iteration % self.render_interval != 0:
            return

        current_assembly_id = self._resolve_assembly_id(optimization_state)
        if self.base_frame is None or self.base_assembly_id != current_assembly_id:
            print(f"[MeshAssemblyModeV2] Using assembly_id={current_assembly_id}")
            self.base_assembly_id = current_assembly_id
            self.base_frame = self._build_base_frame(current_assembly_id)

        vertices, faces = get_mesh_from_optimization_state(optimization_state)
        if vertices is None or faces is None or int(faces.numel()) == 0:
            self.frame_storage.append(self._annotate_frame(self.base_frame.copy(), epoch_value=iteration, sdf_ok=False))
            return

        mesh_textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=faces.shape[0],
            color=self.mesh_color,
        )

        mesh_rgb = self._render(
            vertices=vertices,
            faces=faces,
            textures=mesh_textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        mesh_sil = self._render(
            vertices=vertices,
            faces=faces,
            mode="silhouettes",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        mesh_img = mesh_rgb[0] if hasattr(mesh_rgb, "__len__") and len(mesh_rgb) > 0 else mesh_rgb
        mesh_mask = mesh_sil[0] if hasattr(mesh_sil, "__len__") and len(mesh_sil) > 0 else mesh_sil

        blended = alpha_blend(self.base_frame, mesh_img, mesh_mask.squeeze(-1), alpha=self.mesh_alpha)
        self.frame_storage.append(self._annotate_frame(blended, epoch_value=iteration, sdf_ok=True))

    def finalize(self, output_dir, file_name="mesh_assembly_v2.gif", delete: bool = False) -> None:
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        if self.make_gif and len(self.frame_storage) > 1:
            gif_path = output_dir / file_name
            self._frames_to_gif(gif_path, fps=self.fps)
            print(f"  Saved {len(self.frame_storage)} frames to {gif_path}")
        else:
            png_path = output_dir / file_name.replace(".gif", ".png")
            self._save_image(self.frame_storage[-1], png_path)
            print(f"  Saved frame to {png_path}")

        if delete:
            self.clear_frames()
