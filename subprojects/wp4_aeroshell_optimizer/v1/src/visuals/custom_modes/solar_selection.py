"""
Advanced example: Custom rendering mode with interval-based rendering.

This shows how to override update() for custom logic while still
using the base class conveniences.
"""

from pathlib import Path

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class SolarSelectionMode(RenderingMode):
    """
    Renders which faces were selected for solar performance, black/white difference, no softmax



    Configuration:
        - camera_position: [x, y, z] (default: [4000, 0, 0])
        - look_at: [x, y, z] (default: [0, 0, 0])
        - fps: GIF frame rate (default: 15)
        - render_interval: Render every N iterations (default: 10)
        - make_gif: false saves seperate image every interval iterations
        - fps: if make_gif is true, this sets the fps of the resulting gif, if false this is ignored
        - solar_color:  # Color for faces selected for solar performance
        - rest_color:  # Color for faces not selected for solar performance
    """

    def __init__(self, visualiser, config: dict):
        """Initialize with custom parameter."""
        super().__init__(visualiser, config)
        self.render_interval = config.get("render_interval", 1)

    def update(self, optimization_state: dict) -> None:
        """
        Args:
            optimization_state: Dictionary containing:
                - 'vertices': Current mesh vertices (required)
                - 'faces': Current mesh faces (required)
                - 'iteration': Current iteration number (optional)
                - 'loss': Current loss value (optional)
                - 'model': The current neural network model (optional)
                - Any other custom keys
        """


        iteration = optimization_state.get("iteration", 0)
        # Only render at specified intervals
        if iteration % self.render_interval != 0:
            return


        vertices = optimization_state.get("vertices")
        faces = optimization_state.get("faces")

        if vertices is None or faces is None:
            raise Warning("Warning: No vertices/faces in optimization_state, skipping update")

        # Render from fixed camera position

    # TODO
    raise NotImplementedError("SolarSelectionMode.update() is not implemented yet. This is a placeholder to show how to structure the code for interval-based rendering. Implement your custom rendering logic here using self._render() and manage frame storage as needed.")

    def finalize(self, output_dir: Path, filename: str = "solar_selection.gif", delete: bool = False) -> None:
        """Save GIF plus first and last frames as PNGs."""
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        # Use base class to create GIF
        super().finalize(output_dir, filename=filename, delete=delete)

        # Additionally save first and last frames
        self._save_image(self.frame_storage[0], output_dir / "first_frame.png")
        self._save_image(self.frame_storage[-1], output_dir / "final_frame.png")

        print(f"  Also saved first/last frames to {output_dir}")
