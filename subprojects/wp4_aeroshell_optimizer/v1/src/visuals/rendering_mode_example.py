"""
Example rendering mode implementation.

This demonstrates how to create a custom rendering mode by:
1. Inheriting from RenderingMode
2. Implementing update() to render every N iterations
3. Implementing finalize() to save results

Each team member should create their own file following this pattern.
"""

from pathlib import Path

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base import RenderingMode


class ExampleRenderingMode(RenderingMode):
    """
    Example rendering mode that creates RGB image animations.

    Configuration parameters (set in YAML config under this mode's "config"):
        - render_interval: Render every N iterations (default: 10)
        - camera_position: Camera position [x, y, z] (default: [4000, 0, 0])
        - look_at: Camera look-at point [x, y, z] (default: [0, 0, 0])
        - fps: GIF frame rate (default: 15)
    """

    def __init__(self, visualiser, config: dict):
        """
        Initialize the example rendering mode.

        Args:
            visualiser: Shared Visualiser instance
            config: Mode configuration from YAML
        """
        super().__init__(visualiser, config)
        self.render_interval = config.get("render_interval", 10)
        self.camera_position = config.get("camera_position", [4000, 0, 0])
        self.look_at = config.get("look_at", [0, 0, 0])
        self.fps = config.get("fps", 15)

    def update(self, optimization_state: dict) -> None:
        """
        Render every N iterations and store frames locally.

        Only renders when (iteration % render_interval == 0) to avoid
        filling memory with unnecessary renders.

        Args:
            optimization_state: Current optimization state
        """
        iteration = optimization_state.get("iteration", 0)

        # Only render at specified intervals
        if iteration % self.render_interval != 0:
            return

        vertices = optimization_state.get("vertices")
        faces = optimization_state.get("faces")

        # Skip if mesh data not available
        if vertices is None or faces is None:
            return

        # Compute camera direction from position and look-at point
        direction = [
            self.look_at[i] - self.camera_position[i] for i in range(3)
        ]

        # Render the mesh
        image = self._render(
            vertices=vertices,
            faces=faces,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=direction,
        )

        # Store the rendered image (image[0] if batch dimension exists)
        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir: Path) -> None:
        """
        Save all rendered frames as GIF and sample PNGs.

        Args:
            output_dir: Directory for saving results (mode-specific subdirectory)
        """
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        # Create GIF animation
        gif_path = output_dir / "rendering.gif"
        self._frames_to_gif(gif_path, fps=self.fps)
        print(f"    Saved GIF: {gif_path}")

        # Save first frame as reference
        first_frame_path = output_dir / "first_frame.png"
        self._save_image(self.frame_storage[0], first_frame_path)

        # Save last frame as reference
        last_frame_path = output_dir / "final_frame.png"
        self._save_image(self.frame_storage[-1], last_frame_path)

        print(f"    Saved sample frames to {output_dir}")
