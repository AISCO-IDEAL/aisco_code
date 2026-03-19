from typing import Any

from subprojects.wp2_assembler.version3.components.nbz import call_assembly
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser_utils import assembly_to_kaolin_mesh


class AssemblyMode(RenderingMode):
    """
    Rendering mode for CadQuery assemblies, converting them to mesh format.

    This mode automatically:
    - Converts CadQuery Assembly to mesh data (vertices and faces)
    - Renders from fixed camera position every update() call
    - Stores all frames
    - Creates image on finalize()

    Configuration (set in YAML config under this mode's "config"):
        - camera_position: [x, y, z] (default: [4000, 0, 0])
        - look_at: [x, y, z] (default: [0, 0, 0])
        - render_mode: "rgb", "normal", "depth" (default: "rgb")

        For mesh conversion:
        - tolerance: Tessellation tolerance in model units (default: 0.1)
        - angular_tolerance: Angular tolerance in radians (default: 0.1)
    """

    def __init__(self, visualiser, config):
        super().__init__(visualiser, config)  # Initialize base class as usual
        self.config = config
        self.completed = False
        self.color = tuple(config.get("color", [0.8, 0.8, 0.8]))  # Default gray color

    def update(self, optimization_state: dict[str, Any]) -> None:  # noqa: ARG002
        """
        Update the rendering mode with the current assembly state.

        Args:
            assembly: The current optimization state (not used here)
        """
        if self.completed:
            return  # No further updates, we are done
        assembly_id = load_config(cfg_name=load_config(cfg_name="general_config.yaml")["active_configs"]["losses"])["bb_config"][
            "assembly_id"
        ]
        assembly, _, _ = call_assembly(assembly_id)
        # Get tessellation tolerances from config
        tolerance = self.config.get("tolerance", 0.1)
        angular_tolerance = self.config.get("angular_tolerance", 0.1)

        # Convert CadQuery Assembly to Kaolin-compatible mesh data
        vertices, faces = assembly_to_kaolin_mesh(assembly, tolerance=tolerance, angular_tolerance=angular_tolerance)
        # Create a solid-color texture matching the (batched) face count expected by the renderer.
        # At this point, `faces` is likely (F, 3). The renderer will add a batch dim, so set batch_size=1
        # and num_faces to the number of faces F.
        textures = self.visualiser.create_solid_texture(
            batch_size=1,
            num_faces=faces.shape[0],
            color=self.color,
        )

        # Render the current assembly state
        # Note: The base class _render method expects vertices and faces
        rendered_frame = self._render(
            vertices=vertices,
            faces=faces,
            mode=self.config.get("render_mode", "rgb"),
            camera_position=self.config.get("camera_position"),
            camera_direction=self.camera_direction,
            textures=textures,
        )

        # Store the frame (handle batch dimension like base class does)
        # rendered_frame comes as (1, H, W, C) numpy array, we need (H, W, C)
        if hasattr(rendered_frame, "__len__") and len(rendered_frame) > 0:
            self.frame_storage.append(rendered_frame[0])
        else:
            self.frame_storage.append(rendered_frame)

        self.completed = True  # Mark as completed after first render
        return
        # Frame is automatically saved by the base class render method

    def finalize(self, output_dir, file_name="assembly.png", delete: bool = False) -> None:
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No assembly frame to save, length of storage {len(self.frame_storage)}")
            return

        # Create jpg image
        jpg_path = output_dir / file_name
        self.visualiser.save_image(self.frame_storage[0], jpg_path)

        print(f"  Saved Assembly frame to {jpg_path}")
        if delete:
            self.clear_frames()
            print("  Deleted stored frames after saving image.")
