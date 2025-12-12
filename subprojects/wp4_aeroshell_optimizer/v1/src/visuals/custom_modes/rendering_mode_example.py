"""
Example rendering mode implementation.

This demonstrates the SIMPLEST possible rendering mode:
- Just inherit from RenderingMode
- Set camera parameters in config
- Base class handles everything automatically!

For custom behavior, override update() or finalize().
"""

from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base_class import RenderingMode


class ExampleRenderingMode(RenderingMode):
    """
    Example: Minimal rendering mode using base class defaults.
    
    This mode automatically:
    - Renders from fixed camera position every update() call
    - Stores all frames
    - Creates animation.gif on finalize()
    
    Configuration (set in YAML config under this mode's "config"):
        - camera_position: [x, y, z] (default: [4000, 0, 0])
        - look_at: [x, y, z] (default: [0, 0, 0])
        - render_mode: "rgb", "normal", "depth" (default: "rgb")
        - fps: GIF frame rate (default: 15)
    
    No code needed! Just inherit and configure in YAML.
    """
    pass  # That's it! Base class does everything