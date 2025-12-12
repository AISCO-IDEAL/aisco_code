# Multi-Mode Rendering Architecture

A clean, plugin-based system for collaborative rendering implementations without git conflicts.

## Architecture Overview

```
Wrapper (inherits from Visualiser)
    │
    └─── Single Shared Visualiser Instance
              ↓
    ┌─────────┴──────────┬──────────┐
    │                    │          │
    ▼                    ▼          ▼
Mode A              Mode B       Mode C
(own storage)     (own storage) (own storage)
```

## Key Design Principles

1. **Single Shared Renderer** - One GPU visualiser instance shared by all modes
2. **Independent Storage** - Each mode manages `self.frame_storage` separately → enables parallelization
3. **Configuration-Driven** - Enable/disable modes via YAML, no code changes needed
4. **No Git Conflicts** - Each person creates their own `rendering_mode_<name>.py` file
5. **Error Isolation** - One mode failing doesn't affect others

## How It Works

### 1. Create Your Mode

Create `rendering_mode_yourname.py`:

```python
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.rendering_mode_base import RenderingMode

class YourMode(RenderingMode):
    def __init__(self, visualiser, config: dict):
        super().__init__(visualiser, config)
        self.interval = config.get("interval", 10)

    def update(self, optimization_state: dict) -> None:
        """Called every iteration. Render and store frames."""
        iteration = optimization_state.get("iteration")
        if iteration % self.interval == 0:
            image = self._render(
                vertices=optimization_state["vertices"],
                faces=optimization_state["faces"],
                mode="rgb"
            )
            self.frame_storage.append(image)

    def finalize(self, output_dir: Path) -> None:
        """Called at the end. Save all results."""
        if self.frame_storage:
            self._frames_to_gif(output_dir / "animation.gif")
```

### 2. Register in Config

Add to `general_config.yaml` under `neural_renderer.rendering_modes`:

```yaml
your_mode:
  enabled: false  # Set to true when ready
  class_path: "rendering_mode_yourname:YourMode"
  config:
    interval: 10
    # Your custom parameters here
```

### 3. Use in Code

```python
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import Wrapper

# Initialize (loads all enabled modes from config)
wrapper = Wrapper(verbose=True, output_dir=Path("results/"))

# During optimization loop
for iteration in range(num_iterations):
    wrapper.update(optimization_state={
        'model': model,
        'iteration': iteration,
        'loss': loss_val,
        'vertices': vertices,
        'faces': faces,
        # Add any custom data your modes need
    })

# At the end
wrapper.finalize()
# Results saved to: results/<mode_name>/
```

## Convenience Methods in RenderingMode

```python
self._render(vertices, faces, mode="rgb", **render_kwargs)
    # Renders using shared visualiser, saves_frame=False

self._save_image(image, file_path)
    # Saves single image

self._frames_to_gif(file_path, fps=15, frame_list=None)
    # Converts self.frame_storage to GIF
    # Use frame_list=None to use self.frame_storage

self.clear_frames()
    # Clear stored frames from memory

self.num_stored_frames()
    # Get number of stored frames
```

## File Structure

```
visuals/
├── visual_wrapper.py                      # Main orchestrator (shared by all)
├── visualiser.py                          # Core rendering engine (shared)
├── visualiser_utils.py                    # Utilities (shared)
├── rendering_mode_base.py                 # Abstract base class (shared)
├── rendering_mode_example.py              # Example implementation
├── rendering_mode_yourname.py             # YOUR MODE HERE
├── rendering_mode_person_b.py             # PERSON B'S MODE
├── rendering_mode_person_c.py             # PERSON C'S MODE
└── RENDERING_MODES_CONFIG_TEMPLATE.yaml   # Config guide
```

## Enabling Parallelization Later

Currently modes run sequentially. To enable parallelization:

1. Since each mode has independent `self.frame_storage`, they can already run in parallel
2. The shared visualiser needs thread-safety (add locks if needed)
3. No changes to mode code required—architecture already supports it

## Error Handling

Errors are caught at two levels:

1. **During update()** - Mode-specific error won't break optimization
2. **During finalize()** - Mode-specific error won't prevent other modes from saving

All errors printed to console with full traceback for debugging.

## Files in This System

- **`visual_wrapper.py`** - Orchestrates mode initialization, update calls, and finalization
- **`rendering_mode_base.py`** - Abstract base class that all modes inherit from
- **`rendering_mode_example.py`** - Concrete example showing how to implement a mode
- **`RENDERING_MODES_CONFIG_TEMPLATE.yaml`** - Configuration template and developer guide
- **`RENDERING_MODES_README.md`** - This file

## Common Patterns

### Pattern 1: Render Every N Iterations

```python
def update(self, optimization_state):
    if optimization_state["iteration"] % self.interval == 0:
        image = self._render(...)
        self.frame_storage.append(image)
```

### Pattern 2: Conditional Rendering Based on Loss

```python
def update(self, optimization_state):
    if optimization_state["loss"] < self.threshold:
        image = self._render(...)
        self.frame_storage.append(image)
```

### Pattern 3: Multiple Render Modes (RGB, Normal, Depth)

```python
def update(self, optimization_state):
    rgb = self._render(vertices, faces, mode="rgb")
    normal = self._render(vertices, faces, mode="normal")
    self.frame_storage.append((rgb, normal))

def finalize(self, output_dir):
    # Process rgb and normal separately
    ...
```

## Tips

- Use `self.config.get("param", default)` for safe config access
- Check `self.frame_storage` is not empty before processing in `finalize()`
- Use `output_dir` that's passed in—it's unique per mode
- Catch your own exceptions inside mode methods for custom error handling
- Call `self.clear_frames()` if you're concerned about memory usage
