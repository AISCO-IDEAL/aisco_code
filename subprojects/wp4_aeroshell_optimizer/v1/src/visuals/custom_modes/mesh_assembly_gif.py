from subprojects.wp2_assembler.version3.components.nbz import call_assembly
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
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

        First update: render assembly (RGB) as static background.
        Every update: render mesh (RGB) + silhouette mask, store both.

        Args:
            optimization_state: Dictionary with 'vertices' and 'faces' (or 'verts'/'faces_in')
        """
        # Get vertices and faces from optimization state
        vertices = optimization_state.get("vertices")
        if vertices is None:
            vertices = optimization_state.get("verts")

        faces = optimization_state.get("faces")
        if faces is None:
            faces = optimization_state.get("faces_in")

        if vertices is None or faces is None:
            print("Warning: No vertices/faces in optimization_state, skipping update")
            return

        # Render assembly once on first update (use CadQuery assembly, not the current mesh)
        if not self.rendered_assembly:
            # Resolve assembly id from configs (mirrors AssemblyMode behavior)
            assembly_cfg = load_config(cfg_name="general_config.yaml")
            active_losses_cfg = load_config(cfg_name=assembly_cfg["active_configs"]["losses"])
            assembly_id = active_losses_cfg["bb_config"]["assembly_id"]

            assembly, _, _ = call_assembly(assembly_id)

            tolerance = self.config.get("tolerance", 0.1)
            angular_tolerance = self.config.get("angular_tolerance", 0.1)

            assembly_vertices, assembly_faces = assembly_to_kaolin_mesh(
                assembly,
                tolerance=tolerance,
                angular_tolerance=angular_tolerance,
            )

            # Solid texture for assembly so renderer gets expected texture shape
            textures = self.visualiser.create_solid_texture(
                batch_size=1,
                num_faces=assembly_faces.shape[0],
                color=self.assembly_color,
            )

            assembly_render = self._render(
                vertices=assembly_vertices,
                faces=assembly_faces,
                mode="rgb",
                camera_position=self.camera_position,
                camera_direction=self.camera_direction,
                textures=textures,
            )
            # Remove batch dimension: (1, H, W, C) → (H, W, C)
            if hasattr(assembly_render, "__len__") and len(assembly_render) > 0:
                self.assembly_frame = assembly_render[0]
            else:
                self.assembly_frame = assembly_render

            self.rendered_assembly = True

        # Render mesh RGB frame

        # Make texture for mesh rendering

        textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=faces.shape[0],
            color=self.mesh_color,
        )

        mesh_rgb = self._render(
            vertices=vertices,
            faces=faces,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        # Remove batch dimension
        if hasattr(mesh_rgb, "__len__") and len(mesh_rgb) > 0:
            self.frame_storage.append(mesh_rgb[0])
        else:
            self.frame_storage.append(mesh_rgb)

        # Render mesh silhouette (binary mask)
        silhouette = self._render(
            vertices=vertices,
            faces=faces,
            mode="silhouettes",  # Note: plural
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )
        # Remove batch dimension
        if hasattr(silhouette, "__len__") and len(silhouette) > 0:
            self.silhouettes.append(silhouette[0])
        else:
            self.silhouettes.append(silhouette)

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
