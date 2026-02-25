"""
Example rendering mode implementation.

This demonstrates how to create a custom rendering mode by:
1. Inheriting from RenderingMode
2. Implementing update() to render every N iterations
3. Implementing finalize() to save results

Each team member should create their own file following this pattern.
"""

from pathlib import Path

import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class SolarV3Selection(RenderingMode):
    """
    Example rendering mode that creates RGB image animations.

    Configuration parameters (set in YAML config under this mode's "config"):
        - render_interval: Render every N iterations (default: 10)
        - camera_position: Camera position [x, y, z] (default: [4000, 0, 0])
        - look_at: Camera look-at point [x, y, z] (default: [0, 0, 0])
        - fps: GIF frame rate (default: 15)
    """

    def __init__(self, visualiser):
        """
        Initialize the example rendering mode.

        Args:
            visualiser: Shared Visualiser instance
            config: Mode configuration from YAML
        """

        config = load_config(v="v1", cfg_name="neural_renderer_v0.yaml")["detached_modes"]["solar_v3_selection"]["config"]
        super().__init__(visualiser, config)
        self.camera_position = config["camera_position"]
        self.look_at = config["look_at"]
        self.fps = config["fps"]

    def update(self, optimization_state: dict) -> None:
        # use vertices/faces to render (don't track gradients for performance)
        vertices = optimization_state["vertices"].detach()  # is torch tensor of shape (num_vertices, 3)
        faces = optimization_state["faces"].detach()  # is torch tensor of shape (num_faces, 3) with vertex indices for each face
        threshhold = optimization_state[
            "threshold"
        ]  # is a scalar tensor with the current threshold value, under which faces are considered not selected, you can use this to make not selected faces light red for example
        selection_weights = optimization_state[
            "selection_weights"
        ].detach()  # this is the tensor you want to visualize, implement your texture based on this

        colors = torch.zeros((faces.shape[0], 3), dtype=torch.float32, device=faces.device)  # Shape: (num_faces, 3)
        # make selected ones shades of blue on a spectrum from light blue (low weight) to dark blue (high weight)
        colors[:, 0] = 0  # No red
        colors[:, 1] = 0  # No green
        colors[:, 2] = selection_weights  # Blue intensity based on selection weights

        # make not selected faces light red
        if (
            threshhold > 0
        ):  # Only do this if there is a meaningful threshhold, otherwise all faces are selected and it just looks bad with some light red faces
            not_selected = selection_weights < threshhold
            colors[not_selected, 0] = 1.0  # Red intensity
            colors[not_selected, 1] = 0.5  # Green intensity
            colors[not_selected, 2] = 0.5  # Blue intensity
        else:
            colors[:, 0] = 1.0  # No red
            colors[:, 1] = 1.0  # No green
            colors[:, 2] = selection_weights  # Blue intensity based on selection weights

        # Convert face colors to proper texture format
        textures = self.visualiser.face_colors_to_textures(colors)

        # Render the mesh
        image = self._render(
            vertices=vertices,
            faces=faces,
            textures=textures,
            mode="rgb",
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
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
