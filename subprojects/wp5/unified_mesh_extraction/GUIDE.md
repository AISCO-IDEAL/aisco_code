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
| Field                                     | Purpose                                                                |
| ----------------------------------------- | ---------------------------------------------------------------------- |
| `device`                                  | `"cuda"` / `"cpu"` for SDF eval and output tensors                     |
| `isovalue`                                | Level set extracted by marching cubes                                  |
| `bbox`                                    | `[[xlo,xhi],[ylo,yhi],[zlo,zhi]]` world-space extent                   |
| `base_resolution`                         | Uniform grid resolution                                                |
| `subdivision_factor`, `max_refine_levels` | Multiplier: `effective = base * subdiv**refine`, clamped to `[2, 512]` |
| `batch_size`                              | Chunk size for SDF evaluation                                          |
| `target_num_vertices`                     | Optional; engages the adaptive resolution loop                         |
| `vertex_count_tolerance`                  | Fractional tolerance around the target                                 |
### Field details
- **`isovalue`** — scalar level the mesh extracts, i.e. the surface `{x : s(x) = isovalue}` where `s` is the SDF. For a signed distance field the geometry boundary lies at `0.0`. Shifting to `+5.0` extracts an outward offset surface (thicker), `-5.0` an inward one. The extractor short-circuits to an empty mesh with a warning if the SDF values on the grid do not bracket the isovalue (no sign change, nothing to extract).
- **`base_resolution`** — samples per axis on the uniform sampling grid before any multiplier. The extractor evaluates the SDF at `base_resolution³` points spanning the bbox; voxel spacing is `(bbox_extent) / (base_resolution - 1)`. Doubling this quadruples triangle count (area ~ res²) and multiplies SDF-eval cost by roughly 8×.
- **`subdivision_factor`, `max_refine_levels`** — combined resolution multiplier: `effective = base_resolution * subdivision_factor**max_refine_levels`, clamped to `[2, 512]`. `max_refine_levels = 0` disables subdivision. Replaces adaptive octree refinement with one equivalent uniform resolution (simpler, and the sparse coarse-to-fine path below already avoids paying for the full dense grid). Example: `base=64, subdiv=2, refine=2 → effective=256`.
- **`batch_size`** — points per SDF forward call. The extractor slices the full `(N, 3)` point tensor into `batch_size` chunks and concatenates results, so this bounds peak GPU memory without affecting output geometry. Larger = faster until activations no longer fit in VRAM.
### CFD-AI custom resolution
GAOT2 already wires custom resolution through the shared extractor — see [subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/model_loader.py](subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/model_loader.py). It reads `cfd.mesh_resolution` / `cfd.mesh_resolution_tolerance` from the losses config and injects them via `dataclasses.replace`:
```yaml
# losses_v0_config.yaml
cfd:
  mesh_resolution: 30000        # target vertex count
  mesh_resolution_tolerance: 0.1
```
To change the CFD mesh budget, edit those two values — no code change needed.
## Unified Mesh Extraction | Step by Step

1. **Config → extractor.** `load_mesh_extraction_config()` returns an immutable `MeshExtractionConfig`. `MeshSDFExtractor(config=...)` binds a bbox, effective resolution, isovalue and batch size.
2. **Cached evaluation.** `extract_mesh(sdf_model)` first looks for a mesh that is already in hand: (a) one explicitly injected by a sibling loss via `set_shared_mesh` (used by `TotalLoss` so that e.g. the solar loss and the developability loss share a single extraction), then (b) a process-global cache keyed on `(model id, param version, bbox, resolution, isovalue, target, tol)`. So yes — "checks a shared mesh" means "checks whether this mesh has already been created this step and returns it verbatim". The `_version` counter on the first model parameter auto-invalidates the cache after `optimizer.step` so the next training iteration re-extracts.
3. **Sparse coarse-to-fine marching cubes.** For `effective_resolution > 64` the extractor evaluates a cheap coarse volume, flags cells where the sign changes, dilates by one cell, and only evaluates the fine SDF inside those surface-adjacent voxels. Complexity drops from `O(res³)` to roughly `O(surface)`.
4. **Marching cubes.** Single implementation: PyMCubes (fast C++ backend). Vertices come back in grid-index space and are lifted to world coordinates via per-axis spacing and the bbox origin. A log line like `[PyMCubes] Extracted mesh: 3,538 vertices, 7,072 faces` confirms this path.
5. **Target-vertex-count mode.** When `target_num_vertices` is set, `_extract_to_target` runs up to 6 resolution probes, scaling by `sqrt(target / n)` (area ~ res²). Over-budget meshes are decimated via `fast_simplification` to land inside `±vertex_count_tolerance`. The last successful resolution is warm-started on the next call.
6. **Differentiable surrogate.** Marching cubes is not differentiable, but the surface is an implicit level set. For each extracted vertex `v`, the extractor computes the SDF gradient `n = ∇s(v)` and the projection `dL/ds = -⟨dL/dv, n⟩`. It then returns the scalar `Σ dL/ds · s(v)`, whose `.backward()` produces the correct parameter gradient `∂L/∂θ`.
7. **Memory.** SDF evaluation on CUDA runs under fp16 autocast; marching cubes consumes fp32.
