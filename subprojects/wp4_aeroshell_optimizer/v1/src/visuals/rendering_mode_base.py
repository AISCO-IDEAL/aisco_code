"""
Abstract base class for custom rendering modes.

Each team member should:
1. Create a new file: rendering_mode_<yourname>.py
2. Inherit from RenderingMode
3. Implement the abstract methods: update() and finalize()
4. Register your mode in the config under "rendering_modes"

Example:
    from rendering_mode_base import RenderingMode

    class MyCustomRenderingMode(RenderingMode):
        def __init__(self, visualiser, config: dict):
            super().__init__(visualiser, config)
            # Your initialization here

        def update(self, optimization_state: dict) -> None:
            # Called every optimization step
            # optimization_state contains model, iteration, loss, etc.
            pass

        def finalize(self, output_dir: Path) -> None:
            # Called once at the end
            # Save all your results here
            pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visualiser import Visualiser


class RenderingMode(ABC):
    """
    Abstract base class for custom rendering modes.

    Each mode manages its own frame storage to avoid contention with other modes.
    Modes share access to the Visualiser instance but handle storage independently.
    """

    def __init__(self, visualiser: Visualiser, config: dict):
        """
        Initialize the rendering mode.

        Args:
            visualiser: The Visualiser instance (shared across all modes)
            config: Mode-specific configuration dictionary from YAML
        """
        self.visualiser = visualiser
        self.config = config
        self.enabled = config.get("enabled", False)

        # Each mode manages its own frame storage to avoid conflicts
        self.frame_storage = []

    @abstractmethod
    def update(self, optimization_state: dict[str, Any]) -> None:
        """
        Called once per optimization step.

        Perform rendering operations and store frames/data as needed.
        Each mode is responsible for its own storage to enable parallelization.

        Args:
            optimization_state: Dictionary containing:
                - 'model': The current neural network model
                - 'iteration': Current iteration number (int)
                - 'loss': Current loss value (float)
                - 'vertices': Current mesh vertices (torch.Tensor or np.ndarray, optional)
                - 'faces': Current mesh faces (torch.Tensor or np.ndarray, optional)
                - Any other data your mode needs (custom keys are allowed)

        Example:
            def update(self, optimization_state):
                iteration = optimization_state.get('iteration', 0)
                if iteration % self.render_interval == 0:
                    image = self._render(...)
                    self.frame_storage.append(image)
        """

    @abstractmethod
    def finalize(self, output_dir: Path) -> None:
        """
        Called once when optimization finishes.

        Save all final results, GIFs, images, etc. here using self.frame_storage.

        Args:
            output_dir: Directory where results should be saved (Path object)
                       This is mode-specific; each mode gets its own subdirectory

        Example:
            def finalize(self, output_dir):
                if self.frame_storage:
                    self._frames_to_gif(output_dir / "animation.gif", fps=15)
                    self._save_image(self.frame_storage[0], output_dir / "first.png")
        """

    # ========== Convenience methods for common rendering operations ==========

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
            **render_kwargs
        )

    def _save_image(self, image: Any, file_path: Path) -> None:
        """
        Save a single image using the visualiser's save method.

        Args:
            image: Image data (numpy array)
            file_path: Path where image should be saved
        """
        self.visualiser.save_image(image, str(file_path))

    def _frames_to_gif(self, file_path: Path, fps: int = 15, frame_list: list | None = None) -> None:
        """
        Convert frames to GIF. Uses self.frame_storage by default.

        Args:
            file_path: Output GIF path
            fps: Frames per second
            frame_list: Specific list of frames to use. If None, uses self.frame_storage
        """
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
