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
        - render_mode: "rgb", "normal", etc. (default: "rgb")
        - fps: GIF frame rate (default: 15)
        - render_interval: Render every N iterations (default: 10)
    """
    
    def __init__(self, visualiser, config: dict):
        """Initialize with custom parameter."""
        super().__init__(visualiser, config)
        self.render_interval = config.get("render_interval", 1)
    
    def update(self, optimization_state: dict) -> None:
        """Render every N iterations instead of every call."""
        iteration = optimization_state.get("iteration", 0)
        
        # Only render at specified intervals
        if iteration % self.render_interval != 0:
            return
        
        # Use base class default rendering for the rest
        super().update(optimization_state)
    
    def finalize(self, output_dir: Path) -> None:
        """Save GIF plus first and last frames as PNGs."""
        if not self.frame_storage:
            print(f"  {self.__class__.__name__}: No frames to save")
            return
        
        # Use base class to create GIF
        super().finalize(output_dir)
        
        # Additionally save first and last frames
        self._save_image(self.frame_storage[0], output_dir / "first_frame.png")
        self._save_image(self.frame_storage[-1], output_dir / "final_frame.png")
        
        print(f"  Also saved first/last frames to {output_dir}")
