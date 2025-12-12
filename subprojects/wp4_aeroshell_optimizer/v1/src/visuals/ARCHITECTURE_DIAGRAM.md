"""
ARCHITECTURE DIAGRAM

This file visualizes the multi-mode rendering system.

═══════════════════════════════════════════════════════════════════════════════
                           SYSTEM ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                            Wrapper Class                                   │
│                      (inherits from Visualiser)                            │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                                                                      │  │
│  │            Shared Visualiser Instance (GPU Renderer)               │  │
│  │                                                                      │  │
│  │  • image_size, background_color, anti_aliasing                    │  │
│  │  • device (cuda/cpu)                                               │  │
│  │  • render() method called by all modes                            │  │
│  │                                                                      │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                 ▲                                          │
│              ┌──────────────────┼──────────────────┐                       │
│              │                  │                  │                       │
│              ▼                  ▼                  ▼                        │
│         ┌────────┐         ┌────────┐         ┌────────┐                  │
│         │ Mode A │         │ Mode B │         │ Mode C │                  │
│         ├────────┤         ├────────┤         ├────────┤                  │
│         │update()│         │update()│         │update()│                  │
│         │        │         │        │         │        │                  │
│         │frames_a│         │frames_b│         │frames_c│                  │
│         └────────┘         └────────┘         └────────┘                  │
│         finalize()         finalize()         finalize()                   │
│         ↓                  ↓                  ↓                            │
│      mode_a/           mode_b/            mode_c/                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          DATA FLOW (Per Iteration)
═══════════════════════════════════════════════════════════════════════════════

ITERATION N:

    User Code
        │
        ├─ wrapper.update(optimization_state)
        │
        ├─────────────────────────────────────────────────────────────────┐
        │                                                                 │
        ├──► mode_a.update(optimization_state) ──────┐                   │
        │         │                                  ▼                   │
        │         └──► self._render() ──► (render from shared renderer)  │
        │             └─► self.frame_storage_a.append(image)             │
        │                 [error caught, mode continues]                 │
        │                                                                 │
        ├──► mode_b.update(optimization_state) ──────┐                   │
        │         │                                  ▼                   │
        │         └──► self._render() ──► (render from shared renderer)  │
        │             └─► self.frame_storage_b.append(image)             │
        │                 [error caught, mode continues]                 │
        │                                                                 │
        ├──► mode_c.update(optimization_state) ──────┐                   │
        │         │                                  ▼                   │
        │         └──► self._render() ──► (render from shared renderer)  │
        │             └─► self.frame_storage_c.append(image)             │
        │                 [error caught, mode continues]                 │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                          DATA FLOW (Finalization)
═══════════════════════════════════════════════════════════════════════════════

    wrapper.finalize()
        │
        ├─────────────────────────────────────────────────────────────────┐
        │                                                                 │
        ├──► mkdir(output_dir/mode_a/)                                   │
        │     mode_a.finalize(output_dir/mode_a/)                        │
        │         │                                                      │
        │         ├──► self._frames_to_gif(animation.gif)               │
        │         │     [saves results/mode_a/animation.gif]            │
        │         │                                                      │
        │         └──► self._save_image(first.png, last.png)            │
        │             [saves results/mode_a/first.png etc]              │
        │                 [error caught, continues]                     │
        │                                                                 │
        ├──► mkdir(output_dir/mode_b/)                                   │
        │     mode_b.finalize(output_dir/mode_b/)                        │
        │         [same process, different directory]                   │
        │                                                                 │
        └──► mkdir(output_dir/mode_c/)                                   │
             mode_c.finalize(output_dir/mode_c/)                         │
                 [same process, different directory]                    │


═══════════════════════════════════════════════════════════════════════════════
                      STORAGE ISOLATION (Key Feature)
═══════════════════════════════════════════════════════════════════════════════

Mode A:                           Mode B:                        Mode C:
┌──────────────────┐            ┌──────────────────┐           ┌──────────────┐
│ frame_storage_a  │            │ frame_storage_b  │           │frame_storage_│
│                  │            │                  │           │      c       │
│ [image1]         │            │ [image1]         │           │ [meta1]      │
│ [image2]         │            │ [image2_depth]   │           │ [meta2]      │
│ [image3]         │            │ [image3_normal]  │           │ [meta3]      │
│                  │            │                  │           │              │
│ self.frame_      │            │ self.frame_      │           │ self.frame_  │
│ storage_a        │            │ storage_b        │           │ storage_c    │
└──────────────────┘            └──────────────────┘           └──────────────┘

Each mode can:
✓ Store different data types (RGB, normal, depth, metadata, etc.)
✓ Store different frame counts (every 10 iterations vs every 5)
✓ Use memory independently (one mode doesn't affect others)
✓ Run in parallel (no shared state except visualiser instance)


═══════════════════════════════════════════════════════════════════════════════
                        ERROR ISOLATION FLOW
═══════════════════════════════════════════════════════════════════════════════

Scenario: Mode B crashes during update()

    wrapper.update()
        │
        ├──► mode_a.update() ──► OK, stores frames
        │
        ├──► mode_b.update() ──► CRASH!
        │         │                        ├─► [Exception caught]
        │         │                        ├─► [Traceback printed]
        │         │                        └─► [Continue to Mode C]
        │
        └──► mode_c.update() ──► OK, stores frames

Result: Mode A and C save successfully, Mode B's error doesn't break anything


═══════════════════════════════════════════════════════════════════════════════
                    FILE DEPENDENCY GRAPH
═══════════════════════════════════════════════════════════════════════════════

visual_wrapper.py
    │
    ├── imports: visualiser.py (shared renderer)
    ├── imports: rendering_mode_base.py (abstract class)
    │
    ├── dynamically imports: rendering_mode_example.py
    ├── dynamically imports: rendering_mode_yourname.py (user's mode)
    └── dynamically imports: rendering_mode_person_b.py (team member's mode)


rendering_mode_base.py
    │
    └── TYPE_CHECKING: visualiser.py (for type hints only)


rendering_mode_example.py
    │
    └── imports: rendering_mode_base.py


rendering_mode_yourname.py (create this)
    │
    └── imports: rendering_mode_base.py


═══════════════════════════════════════════════════════════════════════════════
                    OUTPUT DIRECTORY STRUCTURE
═══════════════════════════════════════════════════════════════════════════════

results/                          # output_dir specified in Wrapper()
│
├── mode_a/                       # Create automatically
│   ├── animation.gif
│   ├── first_frame.png
│   └── final_frame.png
│
├── mode_b/                       # Create automatically
│   ├── rendering.gif
│   └── metrics.json
│
└── mode_c/                       # Create automatically
    ├── frames/
    │   ├── 0000.png
    │   ├── 0001.png
    │   └── ...
    └── summary.txt


═══════════════════════════════════════════════════════════════════════════════
                        PARALLELIZATION READY
═══════════════════════════════════════════════════════════════════════════════

CURRENT (Sequential):
    ┌─────────────────────────────────────────────────┐
    │ Iteration N                                     │
    ├─────────────────────────────────────────────────┤
    │ Mode A: update() ──► 0.1s                       │
    │ Mode B: update() ──► 0.2s                       │
    │ Mode C: update() ──► 0.15s                      │
    │                                                 │
    │ Total per iteration: 0.45s                      │
    └─────────────────────────────────────────────────┘


FUTURE (Parallel - same code, different Wrapper):
    ┌──────────────────────────────────────────────────┐
    │ Iteration N (with ThreadPool or async)          │
    ├──────────────────────────────────────────────────┤
    │ Mode A: update() ──┐                            │
    │ Mode B: update() ──┼──► (all run in parallel)   │
    │ Mode C: update() ──┘     0.2s (max of above)   │
    │                                                  │
    │ Total per iteration: 0.2s (4.5x faster!)        │
    └──────────────────────────────────────────────────┘

NO CODE CHANGES NEEDED in mode implementations!
(Only wrapper and visualiser need thread-safety updates)
"""
