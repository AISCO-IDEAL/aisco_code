# Custom Rendering Modes

Create your rendering modes here! Each team member gets their own file.

## Quickstart - Minimal Mode (No Code!)

**1. Create your file:** `rendering_mode_yourname.py`

```python
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base import RenderingMode

class YourMode(RenderingMode):
    pass  # That's it!
```

**2. Configure in YAML:**

```yaml
neural_renderer:
  rendering_modes:
    your_mode:
      enabled: true
      class_path: "custom_modes.rendering_mode_yourname:YourMode"
      config:
        camera_position: [4000, 0, 0]  # Where camera is
        look_at: [0, 0, 0]             # What camera looks at
        render_mode: "rgb"             # rgb, normal, depth, etc.
        fps: 15                        # GIF frame rate
```

**3. Done!** Base class automatically:
- ✅ Renders from fixed camera every `update()` call
- ✅ Stores all frames
- ✅ Creates `animation.gif` on `finalize()`

## Custom Logic (Optional)

Override `update()` or `finalize()` for custom behavior:

```python
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base import RenderingMode
from pathlib import Path

class YourMode(RenderingMode):
    def __init__(self, visualiser, config: dict):
        super().__init__(visualiser, config)
        self.interval = config.get("interval", 10)

    def update(self, optimization_state: dict) -> None:
        # Only render every N iterations
        if optimization_state["iteration"] % self.interval == 0:
            super().update(optimization_state)  # Use base rendering

    def finalize(self, output_dir: Path) -> None:
        # Create GIF + save extra outputs
        super().finalize(output_dir)

        # Save first/last frames as PNGs
        if self.frame_storage:
            self._save_image(self.frame_storage[0], output_dir / "first.png")
            self._save_image(self.frame_storage[-1], output_dir / "last.png")
```

## Examples in This Folder

- **`rendering_mode_example.py`** - Minimal mode (just inherits, no code)
- **`rendering_mode_example_advanced.py`** - Custom mode (interval rendering + extra saves)

## Available Helper Methods

```python
# Render from any camera angle
image = self._render(vertices, faces, mode="rgb",
                     camera_position=[x,y,z],
                     camera_direction=[dx,dy,dz])

# Save single image
self._save_image(image, output_dir / "image.png")

# Create GIF from frames
self._frames_to_gif(output_dir / "anim.gif", fps=15)

# Your frame storage (list)
self.frame_storage.append(image)

# Camera params from config
self.camera_position  # [x, y, z]
self.look_at          # [x, y, z]
self.camera_direction # Computed automatically
self.render_mode      # "rgb", "normal", etc.
self.fps              # GIF frame rate
```

## File Naming

✅ **DO:** `rendering_mode_yourname.py`
❌ **DON'T:** `my_renderer.py`, `custom.py`, etc.

This prevents naming conflicts when multiple people work on the project.

## Git Workflow

1. Create your file: `rendering_mode_yourname.py`
2. Add to config under a unique mode name
3. Test locally (set `enabled: true`)
4. Commit your file + config changes
5. No conflicts! Each person has their own file 🎉

## Output Structure

Results saved to:
```
output_dir/
├── your_mode/
│   └── animation.gif
├── person_a_mode/
│   └── animation.gif
└── person_b_mode/
    └── animation.gif
```

Each mode gets its own subdirectory automatically.
