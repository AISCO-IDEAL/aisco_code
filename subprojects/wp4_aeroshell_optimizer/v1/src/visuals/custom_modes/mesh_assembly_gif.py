import os

import torch

# Import both versions of the assembler to support BbV3 and BbV4
from subprojects.wp2_assembler.version3.components.nbz import call_assembly as call_assembly_v3
from subprojects.wp2_assembler.version4.components.nbz import call_assembly as call_assembly_v4
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import fetch_bounding_box_ply
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser_utils import assembly_to_kaolin_mesh


class MeshAssemblyGIF(RenderingMode):
    """
    Rendering mode that blends mesh evolution over a static assembly background.

    - Renders assembly once (RGB) as the base layer.
    - Each update renders mesh (RGB) + silhouette mask.
    - Finalize blends: assembly + mesh overlaid where silhouette > 0.
    - Alpha config controls mesh opacity (0=invisible, 1=opaque).
    """

    def __init__(self, visualiser, config):
        super().__init__(visualiser, config)
        self.config = config
        self.rendered_assembly = False
        self.assembly_frame = None
        self.frame_storage = []
        self.silhouettes = []
        self.store_wandb = config.get("store_wandb", False)  # Whether to upload the final GIF to wandb
        self.alpha = config.get("alpha", 0.5)  # Transparency factor (mesh opacity)
        self.assembly_color = tuple(config.get("assembly_color", [0.8, 0.8, 0.8]))  # Default assembly color
        self.mesh_color = tuple(config.get("mesh_color", [0.8, 0.8, 0.8]))  # Default mesh color (light gray)

    def update(self, optimization_state: dict) -> None:
        """
        Update the rendering mode with the current optimization state.
        """
        # 1. Extract and validate data
        vertices, faces = self._get_geometry(optimization_state)
        if vertices is None or faces is None:
            print("Warning: No vertices/faces in optimization_state, skipping update")
            return

        # 2. Render assembly once on first update
        if not self.rendered_assembly:
            self._setup_background_assembly()

        # 3. Perform the iteration renderings
        self._render_iteration_frame(vertices, faces)

    def _get_geometry(self, state: dict):
        """Handles the naming variations in the optimization state dictionary."""
        vertices = state.get("vertices") or state.get("verts")
        faces = state.get("faces") or state.get("faces_in")
        return vertices, faces

    def _setup_background_assembly(self) -> None:
        """Resolves config and renders the static background assembly."""
        assembly_cfg = load_config(cfg_name="general_config.yaml")
        active_losses_cfg = load_config(cfg_name=assembly_cfg["active_configs"]["losses"])
        bb_config = active_losses_cfg.get("bb_config", {})

        # Route logic: Static PLY vs CAD Assembly
        if "inner_ply_location" in bb_config:
            inner_mesh, _ = fetch_bounding_box_ply(
                inner_ply_location=bb_config["inner_ply_location"], outer_ply_location=bb_config.get("outer_ply_location", "")
            )
            assembly_vertices = torch.as_tensor(inner_mesh.vertices, dtype=torch.float32)
            assembly_faces = torch.as_tensor(inner_mesh.faces, dtype=torch.long)
        else:
            assembly_vertices, assembly_faces = self._get_cad_assembly_geometry(bb_config)

        # Render the background frame
        textures = self.visualiser.create_solid_texture(
            batch_size=1, num_faces=assembly_faces.shape[0], color=self.assembly_color
        )
        render = self._render(
            vertices=assembly_vertices,
            faces=assembly_faces,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
            textures=textures,
        )

        self.assembly_frame = render[0] if hasattr(render, "__len__") and len(render) > 0 else render
        self.rendered_assembly = True

    def _get_cad_assembly_geometry(self, bb_config: dict):
        """Handles CAD specific assembly fetching logic."""
        env_id = os.getenv("AISCO_ASSEMBLY_ID", "").strip()
        assembly_id = int(env_id) if env_id else bb_config.get("assembly_id", 1)

        if os.getenv("AISCO_WP2_VERSION", "v3") == "v4":
            assembly, _, _ = call_assembly_v4(assembly_id)
        else:
            assembly, _, _ = call_assembly_v3(assembly_id)

        return assembly_to_kaolin_mesh(
            assembly, tolerance=self.config.get("tolerance", 0.1), angular_tolerance=self.config.get("angular_tolerance", 0.1)
        )

    def _render_iteration_frame(self, vertices, faces) -> None:
        """Renders both the RGB mesh and the silhouette mask for the current step."""
        # RGB Rendering
        textures = self.visualiser.create_solid_texture(batch_size=1, num_faces=faces.shape[0], color=self.mesh_color)
        rgb = self._render(
            vertices=vertices,
            faces=faces,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
            textures=textures,
        )
        self.frame_storage.append(rgb[0] if hasattr(rgb, "__len__") and len(rgb) > 0 else rgb)

        # Silhouette Rendering
        sil = self._render(
            vertices=vertices,
            faces=faces,
            mode="silhouettes",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        self.silhouettes.append(sil[0] if hasattr(sil, "__len__") and len(sil) > 0 else sil)

    def finalize(self, output_dir, file_name="mesh_assembly.gif", delete: bool = False) -> None:
        """
        Combine assembly background with mesh frames using silhouette-based alpha blending.

        Blending formula:
            blended = mesh_rgb * (alpha * silhouette) + assembly_rgb * (1 - alpha * silhouette)

        Where silhouette is the binary mask (0 or 1, shape H*W*1).
        """
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No mesh frames to save")
            return

        if self.assembly_frame is None:
            print(f"  {self.__class__.__name__}: No assembly frame, cannot blend")
            return

        if len(self.silhouettes) != len(self.frame_storage):
            print(
                f"  {self.__class__.__name__}: Silhouette count mismatch ({len(self.silhouettes)} vs {len(self.frame_storage)})"
            )
            return

        # Combine assembly + mesh frames using silhouette masks
        combined_frames = []
        for mesh_frame, silhouette in zip(self.frame_storage, self.silhouettes, strict=False):
            # silhouette: (H, W, 1) with values in [0, 1]
            # mesh_frame, assembly_frame: (H, W, 3) with values in [0, 1]

            # Scale silhouette by alpha for mesh opacity control
            mask = silhouette * self.alpha  # (H, W, 1)

            # Alpha blend: mesh where mask > 0, assembly elsewhere
            blended = mesh_frame * mask + self.assembly_frame * (1.0 - mask)
            combined_frames.append(blended)

        # Temporarily replace frame_storage with combined frames for GIF creation
        original_frames = self.frame_storage
        self.frame_storage = combined_frames

        try:
            gif_path = output_dir / file_name
            self._frames_to_gif(gif_path, fps=self.fps)
            print(f"  Saved {len(combined_frames)} blended frames to {gif_path}")
        finally:
            # Restore original frames
            self.frame_storage = original_frames
        if delete:
            self.clear_frames()
            print("  Deleted stored frames after saving GIF.")
