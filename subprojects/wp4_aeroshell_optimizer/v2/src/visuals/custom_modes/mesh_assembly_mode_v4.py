from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.combined_ply_export import export_combined_bbox_mesh_ply
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


class MeshAssemblyModeV4(RenderingMode):
    """
    Stable mesh-assembly renderer:
    - Uses optimization_state vertices/faces for SDF mesh (same stable path as mesh_assembly mode).
    - Renders inner assembly + outer BB once as static background.
    - Overlays SDF mesh every render_interval and stores frames for GIF.
    """

    def __init__(self, visualiser, config):
        # Collect camera positions from numbered keys (camera_position_1, camera_position_2, ...)
        # or fall back to a single camera_position for backward compatibility.
        camera_positions: list[list[float]] = []
        idx = 1
        while f"camera_position_{idx}" in config:
            camera_positions.append(config[f"camera_position_{idx}"])
            idx += 1
        if not camera_positions:
            camera_positions.append(config["camera_position"])  # raises KeyError with clear message if absent

        # Base class requires camera_position; inject first position to satisfy its validation.
        patched_config = dict(config)
        patched_config.setdefault("camera_position", camera_positions[0])
        super().__init__(visualiser, patched_config)

        self.config = config
        self.camera_positions: list[list[float]] = camera_positions
        # Per-camera look_at: look_at_1, look_at_2, ... fall back to the shared look_at.
        look_at_targets: list[list[float]] = []
        for i in range(len(camera_positions)):
            key = f"look_at_{i + 1}"
            if key in config:
                look_at_targets.append(config[key])
                print(
                    f"[MeshAssemblyModeV4] cam{i}: camera_position={camera_positions[i]}, look_at=config['{key}']={config[key]}"
                )
            else:
                look_at_targets.append(self.look_at)
                print(
                    f"[MeshAssemblyModeV4] cam{i}: camera_position={camera_positions[i]}, look_at=default={self.look_at} ('{key}' not in config, available keys: {[k for k in config if 'look' in k.lower()]})"
                )
        self.camera_directions: list[list[float]] = [
            [target[j] - pos[j] for j in range(3)] for pos, target in zip(camera_positions, look_at_targets, strict=True)
        ]
        print(f"[MeshAssemblyModeV4] camera_directions={self.camera_directions}")
        # One cached base frame and assembly-id per camera position.
        self.base_frames: list[np.ndarray | None] = [None] * len(camera_positions)
        self.base_assembly_ids: list[int | None] = [None] * len(camera_positions)
        # Cached bbox mesh tensors keyed by assembly_id for PLY export.
        self.cached_bbox_tensors: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # PLAN-026: frame_storage is a list[list[Any]] here — one inner list per camera.
        # This is a deliberate LSP deviation: RenderingMode declares frame_storage as
        # list[Any] (flat).  MeshAssemblyModeV4 overrides it with a nested structure so
        # that finalize() can write one GIF per camera into cam{idx}/ sub-directories.
        #
        # IMPORTANT FOR FUTURE MODE AUTHORS: any code that reads frame_storage via a
        # RenderingMode reference (e.g. visual_wrapper.py) MUST guard against the nested
        # layout before treating it as a flat list of frames.  Check whether the first
        # element is itself a list (and that storage is non-empty) before doing so.
        # Failing to guard will produce wrong W&B upload keys and potentially an
        # IndexError on empty storage.
        # See also: AISCO-docs/40_Aeroshell_Optimiser/02 Code Documentation/
        #           Multi-Camera Visualiser Guide.md — "frame_storage Contract" section.
        self.frame_storage: list[list[Any]] = [[] for _ in camera_positions]
        self.render_interval = int(config.get("render_interval", 1))
        self.make_gif = bool(config.get("make_gif", True))

        self.ply_export_enabled = bool(config.get("ply_export_enabled", False))
        self.ply_export_interval = int(config.get("ply_export_interval", self.render_interval))
        self._ply_missing_mesh_dir_warned = False
        # Holds the last optimization_state that contained a valid mesh, so finalize()
        # can write at least one combined PLY if no interval export fired yet.
        self._last_ply_state: dict | None = None
        self._ply_export_count: int = 0

        self.mesh_color = tuple(config.get("mesh_color", [0.2, 1.0, 1.0]))
        self.mesh_alpha = float(config.get("mesh_alpha", config.get("alpha", 0.5)))

        self.assembly_color = tuple(config.get("assembly_color", [1.0, 0.55, 0.0]))
        self.outer_color = tuple(config.get("outer_color", [1.0, 0.35, 0.15]))
        self.outer_alpha = float(config.get("outer_alpha", 0.2))
        self.outer_edge_color = tuple(config.get("outer_edge_color", [0.0, 0.0, 0.0]))
        self.outer_edge_strength = float(config.get("outer_edge_strength", 1.0))

        self.tolerance = float(config.get("tolerance", 2.0))
        self.angular_tolerance = float(config.get("angular_tolerance", 0.4))

    # ------------------------------------------------------------------
    # RenderingMode overrides required by the nested frame_storage layout
    # ------------------------------------------------------------------

    def clear_frames(self) -> None:
        """Reset each per-camera buffer to empty without destroying the outer list.

        The base-class implementation calls ``self.frame_storage.clear()``, which
        would collapse the nested structure to a flat empty list.  This override
        empties each inner list individually so the n_cams slot structure is
        preserved for the next training run.
        """
        for inner in self.frame_storage:
            inner.clear()

    def num_stored_frames(self) -> int:
        """Return total frame count summed across all cameras.

        The base-class implementation returns ``len(self.frame_storage)`` which,
        on the nested list, would return n_cams instead of the total number of
        frames.  This override sums the inner list lengths.
        """
        return sum(len(inner) for inner in self.frame_storage)

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

    def _render_look(self, **kwargs) -> Any:
        """Render in 'look' mode so camera_direction is respected.

        The shared renderer uses camera_mode='look_at' by default, which always
        points at the world origin ([0,0,0]) and ignores camera_direction.
        Temporarily switching to 'look' mode makes it use camera_direction instead,
        which we compute as (look_at_target - camera_position) per camera.
        The original mode is always restored, even on error.
        """
        renderer = self.visualiser.renderer
        original_mode = renderer.camera_mode
        renderer.camera_mode = "look"
        try:
            return self._render(**kwargs)
        finally:
            renderer.camera_mode = original_mode

    def _build_base_frame(self, assembly_id: int, camera_position: list[float], camera_direction: list[float]) -> np.ndarray:
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

        # Cache raw bbox tensors once per assembly_id so update() can use them
        # for PLY export without re-tessellating.
        if assembly_id not in self.cached_bbox_tensors:
            self.cached_bbox_tensors[assembly_id] = (
                inner_vertices.detach().cpu(),
                inner_faces.detach().cpu(),
                outer_vertices.detach().cpu(),
                outer_faces.detach().cpu(),
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

        inner_rgb = self._render_look(
            vertices=inner_vertices,
            faces=inner_faces,
            textures=inner_textures,
            mode="rgb",
            camera_position=camera_position,
            camera_direction=camera_direction,
        )
        outer_rgb = self._render_look(
            vertices=outer_vertices,
            faces=outer_faces,
            textures=outer_textures,
            mode="rgb",
            camera_position=camera_position,
            camera_direction=camera_direction,
        )
        outer_sil = self._render_look(
            vertices=outer_vertices,
            faces=outer_faces,
            mode="silhouettes",
            camera_position=camera_position,
            camera_direction=camera_direction,
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
        vertices, faces = get_mesh_from_optimization_state(optimization_state)

        for cam_idx, (cam_pos, cam_dir) in enumerate(zip(self.camera_positions, self.camera_directions, strict=True)):
            try:
                if self.base_frames[cam_idx] is None or self.base_assembly_ids[cam_idx] != current_assembly_id:
                    if cam_idx == 0:
                        print(f"[MeshAssemblyModeV4] Using assembly_id={current_assembly_id}")
                    self.base_assembly_ids[cam_idx] = current_assembly_id
                    self.base_frames[cam_idx] = self._build_base_frame(current_assembly_id, cam_pos, cam_dir)

                base_frame = self.base_frames[cam_idx]
                assert base_frame is not None  # guaranteed by the block above

                if vertices is None or faces is None or int(faces.numel()) == 0:
                    self.frame_storage[cam_idx].append(
                        self._annotate_frame(base_frame.copy(), epoch_value=iteration, sdf_ok=False)
                    )
                    continue

                mesh_textures = self.visualiser.create_solid_texture(
                    batch_size=1,
                    num_faces=faces.shape[0],
                    color=self.mesh_color,
                )

                mesh_rgb = self._render_look(
                    vertices=vertices,
                    faces=faces,
                    textures=mesh_textures,
                    mode="rgb",
                    camera_position=cam_pos,
                    camera_direction=cam_dir,
                )
                mesh_sil = self._render_look(
                    vertices=vertices,
                    faces=faces,
                    mode="silhouettes",
                    camera_position=cam_pos,
                    camera_direction=cam_dir,
                )

                mesh_img = mesh_rgb[0] if hasattr(mesh_rgb, "__len__") and len(mesh_rgb) > 0 else mesh_rgb
                mesh_mask = mesh_sil[0] if hasattr(mesh_sil, "__len__") and len(mesh_sil) > 0 else mesh_sil

                blended = alpha_blend(base_frame, mesh_img, mesh_mask.squeeze(-1), alpha=self.mesh_alpha)
                self.frame_storage[cam_idx].append(self._annotate_frame(blended, epoch_value=iteration, sdf_ok=True))

            except Exception as e:
                print(f"[MeshAssemblyModeV4] cam{cam_idx} rendering failed: {e}")
                # Append a black fallback frame so all cameras always contribute
                # equal frame counts (prevents 3:2 imbalance in the GIF/W&B upload).
                fallback = (
                    np.zeros_like(self.base_frames[cam_idx])
                    if self.base_frames[cam_idx] is not None
                    else np.zeros((1, 1, 3), dtype=np.float32)
                )
                self.frame_storage[cam_idx].append(self._annotate_frame(fallback, epoch_value=iteration, sdf_ok=False))

        # PLY export — once per ply_export_interval (independent of camera loop).
        if self.ply_export_enabled and vertices is not None and faces is not None and int(faces.numel()) > 0:
            # Always keep the most recent valid state so finalize() can flush it.
            self._last_ply_state = {
                "iteration": iteration,
                "assembly_id": current_assembly_id,
                "vertices": vertices.detach().cpu(),
                "faces": faces.detach().cpu(),
                "mesh_dir": optimization_state.get("mesh_dir"),
            }

        if (
            self.ply_export_enabled
            and iteration is not None
            and iteration % self.ply_export_interval == 0
            and vertices is not None
            and faces is not None
            and int(faces.numel()) > 0
        ):
            self._write_ply(
                assembly_id=current_assembly_id,
                iteration=iteration,
                mesh_dir=optimization_state.get("mesh_dir"),
                sdf_verts=vertices.detach().cpu(),
                sdf_faces=faces.detach().cpu(),
            )

    def _write_ply(
        self,
        assembly_id: int,
        iteration: int,
        mesh_dir: Any,
        sdf_verts: torch.Tensor,
        sdf_faces: torch.Tensor,
    ) -> None:
        """Write a combined bbox+SDF PLY for the given epoch. Called from update() and finalize()."""
        if mesh_dir is None:
            if not self._ply_missing_mesh_dir_warned:
                print("[MeshAssemblyModeV4] PLY export skipped: 'mesh_dir' not found in optimization_state")
                self._ply_missing_mesh_dir_warned = True
            return
        bbox = self.cached_bbox_tensors.get(assembly_id)
        if bbox is None:
            return
        inner_v, inner_f, outer_v, outer_f = bbox
        ply_path = mesh_dir / f"combined_bb_mesh_epoch{iteration:04d}.ply"
        try:
            export_combined_bbox_mesh_ply(
                inner_verts=inner_v,
                inner_faces=inner_f,
                outer_verts=outer_v,
                outer_faces=outer_f,
                sdf_verts=sdf_verts,
                sdf_faces=sdf_faces,
                output_path=ply_path,
                inner_color=self.assembly_color,
                outer_color=self.outer_color,
                sdf_color=self.mesh_color,
                outer_alpha=self.outer_alpha,
                sdf_alpha=self.mesh_alpha,
            )
            print(f"[MeshAssemblyModeV4] Combined PLY exported: epoch={iteration}, path={ply_path}, components=inner+outer+sdf")
            self._ply_export_count += 1
        except Exception as ply_exc:
            print(f"[MeshAssemblyModeV4] PLY export failed at epoch={iteration}: {ply_exc}")

    def finalize(self, output_dir, file_name="mesh_assembly_v4.gif", delete: bool = False) -> None:
        # If PLY export was enabled but no interval export fired (e.g. ply_export_interval
        # exceeds num_epochs), write the last valid state now as a final snapshot.
        if self.ply_export_enabled and self._ply_export_count == 0 and self._last_ply_state is not None:
            state = self._last_ply_state
            iteration = state["iteration"] if state["iteration"] is not None else 0
            self._write_ply(
                assembly_id=state["assembly_id"],
                iteration=iteration,
                mesh_dir=state.get("mesh_dir"),
                sdf_verts=state["vertices"],
                sdf_faces=state["faces"],
            )

        # Early-exit when every per-camera buffer is empty.
        if not any(self.frame_storage):
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        # Write one GIF (or PNG) per camera directly into output_dir, prefixed
        # with cam{idx}_ so all perspectives are visible in one folder.
        for cam_idx, cam_frames in enumerate(self.frame_storage):
            if not cam_frames:
                continue
            if self.make_gif and len(cam_frames) > 1:
                gif_path = output_dir / f"cam{cam_idx}_{file_name}"
                self._frames_to_gif(gif_path, fps=self.fps, frame_list=cam_frames)
                print(f"  cam{cam_idx}: saved {len(cam_frames)} frames → {gif_path}")
            else:
                png_path = output_dir / f"cam{cam_idx}_{file_name.replace('.gif', '.png')}"
                self._save_image(cam_frames[-1], png_path)
                print(f"  cam{cam_idx}: saved frame → {png_path}")

        if delete:
            self.clear_frames()
