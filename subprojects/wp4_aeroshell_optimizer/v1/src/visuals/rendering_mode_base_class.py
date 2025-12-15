"""
Abstract base class for custom rendering modes.

Base mode automatically:
- Renders from a fixed camera position every update() call
- Stores frames automatically
- Creates GIF on finalize()

Each team member should:
1. Create a new file in custom_modes/: rendering_mode_<yourname>.py
2. Inherit from RenderingMode
3. Set camera parameters in config
4. Optionally override update() for custom logic
5. Register your mode in the config under "rendering_modes"

Minimal Example:
    from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base import RenderingMode

    class MyMode(RenderingMode):
        pass  # That's it! Base class handles everything

    # Config:
    # my_mode:
    #   enabled: true
    #   class_path: "custom_modes.rendering_mode_myname:MyMode"
    #   config:
    #     camera_position: [4000, 0, 0]
    #     look_at: [0, 0, 0]
    #     render_mode: "rgb"
    #     fps: 15
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


class MissingConfigKeyError(KeyError):
    """Raised when a required rendering mode config key is missing."""

    def __init__(self, missing_key: str, available_keys: list[str]):
        self.missing_key = missing_key
        self.available_keys = available_keys
        super().__init__(f"Missing required key '{missing_key}' in rendering mode config. Available: {available_keys}")


class RenderingMode:
    """
    Base class for custom rendering modes with automatic rendering.

    The base implementation automatically renders from a fixed camera position
    and creates a GIF. Override update() only if you need custom behavior.
    """

    def __init__(self, visualiser: Visualiser, config: dict):
        """
        Initialize the rendering mode.

        Args:
            visualiser: The Visualiser instance (shared across all modes)
            config: Mode-specific configuration dictionary from YAML
                Required keys:
                - camera_position: [x, y, z] camera location
                - look_at: [x, y, z] point to look at
                - render_mode: "rgb", "normal", "depth", etc.
                - fps: GIF frame rate
        """

        self.visualiser = visualiser
        self.config = config

        # Validate required config keys
        required_keys = ["camera_position", "look_at", "render_mode", "fps"]
        for key in required_keys:
            if key not in config:
                raise MissingConfigKeyError(key, list(config.keys()))

        # Camera settings
        self.camera_position = config["camera_position"]
        self.look_at = config["look_at"]
        self.render_mode = config["render_mode"]
        self.fps = config["fps"]

        # Compute camera direction from position and look_at
        self.camera_direction = [self.look_at[i] - self.camera_position[i] for i in range(3)]

        # Each mode manages its own frame storage
        self.frame_storage = []

    def update(self, optimization_state: dict[str, Any]) -> None:
        """
        Render from fixed camera position and store frame.

        Override this method if you need custom rendering logic.
        Default behavior: renders every call and stores the frame.

        Args:
            optimization_state: Dictionary containing:
                - 'vertices': Current mesh vertices (required)
                - 'faces': Current mesh faces (required)
                - 'iteration': Current iteration number (optional)
                - 'loss': Current loss value (optional)
                - 'model': The current neural network model (optional)
                - Any other custom keys
        """
        vertices = optimization_state.get("vertices")
        faces = optimization_state.get("faces")

        if vertices is None or faces is None:
            return

        # Render from fixed camera position
        image = self._render(
            vertices=vertices,
            faces=faces,
            mode=self.render_mode,
            camera_position=self.camera_position,
            camera_direction=self.camera_direction,
        )

        # Store the frame (handle batch dimension if present)
        if hasattr(image, "__len__") and len(image) > 0:
            self.frame_storage.append(image[0])
        else:
            self.frame_storage.append(image)

    def finalize(self, output_dir: Path, file_name: str = "animation.gif") -> None:
        """
        Save all stored frames as GIF.

        Override this method if you need custom save logic.
        Default behavior: creates animation.gif from all stored frames.

        Args:
            output_dir: Directory where results should be saved (Path object)
        """
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return

        # Create GIF animation
        gif_path = output_dir / file_name
        self._frames_to_gif(gif_path, fps=self.fps)

        print(f"  Saved {len(self.frame_storage)} frames to {gif_path}")

    # ========== Convenience methods for custom rendering ==========

    def _render(self, vertices, faces, mode: str = "rgb", **render_kwargs) -> Any:
        """
        Render a mesh using the shared Visualiser.

        Args:
            vertices: Mesh vertices
            faces: Mesh faces
            mode: Render mode (e.g., "rgb", "normal", "depth")
            **render_kwargs: Additional arguments passed to visualiser.render()

        Returns:
            Rendered output (typically numpy array)
        """
        return self.visualiser.render(
            vertices=vertices,
            faces=faces,
            mode=mode,
            save_frame=False,  # We manage storage ourselves
            **render_kwargs,
        )

    def _save_image(self, image: Any, file_path: Path) -> None:
        """
        Save a single image using the visualiser's save method.

        Args:
            image: Image data (numpy array)
            file_path: Path where image should be saved
        """
        self.visualiser.save_image(image, str(file_path))

    def _frames_to_gif(self, file_path: Path, fps: int | None = None, frame_list: list | None = None) -> None:
        """
        Convert frames to GIF. Uses self.frame_storage by default.

        Args:
            file_path: Output GIF path
            fps: Frames per second (uses self.fps if None)
            frame_list: Specific list of frames to use. If None, uses self.frame_storage
        """
        fps = fps if fps is not None else self.fps
        frames_to_use = frame_list if frame_list is not None else self.frame_storage

        if not frames_to_use:
            return

        # Temporarily swap frames to use visualiser's GIF method
        original_frames = self.visualiser.stored_frames
        self.visualiser.stored_frames = frames_to_use

        try:
            self.visualiser.frames_to_gif(file_path=str(file_path), fps=fps)
        finally:
            # Always restore original frames
            self.visualiser.stored_frames = original_frames

    def clear_frames(self) -> None:
        """Clear all stored frames from memory."""
        self.frame_storage.clear()

    def num_stored_frames(self) -> int:
        """Get number of stored frames."""
        return len(self.frame_storage)
