# Unified Mesh Extraction — Quick Guide

Module: [subprojects/wp5/unified_mesh_extraction/sdf2mesh.py](subprojects/wp5/unified_mesh_extraction/sdf2mesh.py)
Config: [subprojects/wp5/unified_mesh_extraction/mesh_extraction_config.yaml](subprojects/wp5/unified_mesh_extraction/mesh_extraction_config.yaml)

## Usage

Load defaults from YAML and build an extractor:

```python
from subprojects.wp5.unified_mesh_extraction import (
    MeshSDFExtractor,
    load_mesh_extraction_config,
)

cfg = load_mesh_extraction_config()
extractor = MeshSDFExtractor(config=cfg)
vertices, faces = extractor.extract_mesh(sdf_model)  # torch tensors on cfg.device
```

Override fields at call site (e.g. to bind a different bbox, device, or vertex budget):

```python
from dataclasses import replace

cfg = replace(
    load_mesh_extraction_config(),
    device="cuda",
    bbox=[[-3e3, 3e3], [-3e3, 3e3], [-3e3, 3e3]],
    target_num_vertices=30000,       # optional; None = fixed-resolution extraction
    vertex_count_tolerance=0.1,      # ±10% around target
)
extractor = MeshSDFExtractor(config=cfg)
```

Differentiable surrogate for SDF-parameter gradients:

```python
v, f, loss, surrogate, dLdv = extractor.extract_and_compute_loss(sdf_model, loss_fn)
surrogate.backward()  # routes gradients into the SDF parameters
```

### Config fields

| Field | Purpose |
|---|---|
| `device` | `"cuda"` / `"cpu"` for SDF eval and output tensors |
| `isovalue` | Level set extracted by marching cubes |
| `bbox` | `[[xlo,xhi],[ylo,yhi],[zlo,zhi]]` world-space extent |
| `base_resolution` | Uniform grid resolution |
| `subdivision_factor`, `max_refine_levels` | Multiplier: `effective = base * subdiv**refine`, clamped to `[2, 512]` |
| `batch_size` | Chunk size for SDF evaluation |
| `target_num_vertices` | Optional; engages the adaptive resolution loop |
| `vertex_count_tolerance` | Fractional tolerance around the target |

### CFD-AI custom resolution

GAOT2 already wires custom resolution through the shared extractor — see [subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/model_loader.py](subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/model_loader.py). It reads `cfd.mesh_resolution` / `cfd.mesh_resolution_tolerance` from the losses config and injects them via `dataclasses.replace`:

```yaml
# losses_v0_config.yaml
cfd:
  mesh_resolution: 30000        # target vertex count
  mesh_resolution_tolerance: 0.1
```

To change the CFD mesh budget, edit those two values — no code change needed.

---

## How it works

1. **Config → extractor.** `load_mesh_extraction_config()` returns an immutable `MeshExtractionConfig`. `MeshSDFExtractor(config=...)` binds a bbox, effective resolution, isovalue and batch size.
2. **Cached evaluation.** `extract_mesh(sdf_model)` first checks a shared mesh injected by a sibling loss (`set_shared_mesh`), then a process-global cache keyed on `(model id, param version, bbox, resolution, isovalue, target, tol)`. The `_version` counter on the first model parameter auto-invalidates entries after `optimizer.step`.
3. **Sparse coarse-to-fine marching cubes.** For `effective_resolution > 64` the extractor evaluates a cheap coarse volume, flags cells where the sign changes, dilates by one cell, and only evaluates the fine SDF inside those surface-adjacent voxels. Complexity drops from `O(res³)` to roughly `O(surface)`.
4. **Marching cubes.** Uses PyMCubes when available (fast C++), otherwise `skimage.measure.marching_cubes`. Vertices are lifted from grid indices to world coordinates via per-axis spacing and bbox origin.
5. **Target-vertex-count mode.** When `target_num_vertices` is set, `_extract_to_target` runs up to 6 resolution probes, scaling by `sqrt(target / n)` (area ~ res²). Over-budget meshes are decimated via `fast_simplification` to land inside `±vertex_count_tolerance`. The last successful resolution is warm-started on the next call.
6. **Differentiable surrogate.** Marching cubes is not differentiable, but the surface is an implicit level set. For each extracted vertex `v`, the extractor computes the SDF gradient `n = ∇s(v)` and the projection `dL/ds = -⟨dL/dv, n⟩`. It then returns the scalar `Σ dL/ds · s(v)`, whose `.backward()` produces the correct parameter gradient `∂L/∂θ`.
7. **Memory.** SDF evaluation on CUDA runs under fp16 autocast; marching cubes consumes fp32.
