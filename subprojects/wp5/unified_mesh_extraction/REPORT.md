# `sdf2mesh.py` — Functional Walkthrough

This report is generated from the step-by-step docstrings attached to every
function in [sdf2mesh.py](subprojects/wp5/unified_mesh_extraction/sdf2mesh.py). It
describes *how* the module turns a Signed Distance Function (SDF) into a
triangle mesh, how meshes are shared across losses, and how gradients are
propagated back into the SDF parameters.

---

## 1. Module-level state

- `_MESH_CACHE: dict[key, (vertices, faces)]` — process-wide cache shared by
  every `MeshSDFExtractor` instance. Losses with matching configs reuse the
  same extracted mesh within a training step.
- `_MESH_CACHE_MAX_ENTRIES = 16` — FIFO eviction bound.
- Auto-invalidation: cache keys include the SDF's first-parameter `_version`
  counter. An in-place op or `optimizer.step()` bumps this counter and
  invalidates stale entries.

## 2. Utility helpers

### `_model_version(model)`
1. Grab the first parameter tensor of `model`.
2. Read its autograd `_version` counter.
3. Fall back to `0` if `model` has no parameters.

Used by the cache to tell "fresh" from "stale" meshes across training steps.

### `_normalise_bbox(bbox)`
1. If `bbox` is a flat 6-tuple of scalars, split into 3 pairs.
2. Otherwise assume it is a 3×2 nested iterable and cast each bound to float.

### `load_mesh_extraction_config(path=None)`
1. Resolve the YAML path (default: sibling `mesh_extraction_config.yaml`).
2. `yaml.safe_load` and reject non-mapping payloads.
3. Build the immutable `MeshExtractionConfig` dataclass, applying defaults
   for optional keys.

### `compute_surface_area(vertices, faces)`
1. Short-circuit to 0 for empty meshes.
2. Gather per-triangle vertex triples → shape `(F, 3, 3)`.
3. Compute two edge vectors and their cross product.
4. Triangle area = `0.5 * ||cross||`; return the sum. Differentiable w.r.t.
   `vertices`.

---

## 3. `MeshSDFExtractor` — public API

### `__init__(config=None)`
1. Use the provided `config` or load YAML defaults.
2. Downgrade to CPU if CUDA was requested but is unavailable.
3. Cache hot fields (`isovalue`, `batch_size`, `target_num_vertices`).
4. Enable fp16 autocast on CUDA for memory-efficient SDF evaluation.
5. Initialise slots for cross-loss shared-mesh injection.
6. Call `set_grid` to compute the effective grid resolution.

### `set_grid(bbox, resolution)`
1. Normalise the bbox.
2. `effective_resolution = base_resolution * subdivision_factor ** max_refine_levels`,
   clamped to `[2, 512]`. This uniform-grid multiplier replaces adaptive
   octree refinement with a simpler, faster equivalent.
3. Invalidate the warm-started resolution for target-count extraction.

### `extract_mesh(sdf_model)` *(core entry point)*
1. **Fast path**: if `TotalLoss` injected a shared mesh via
   `set_shared_mesh`, return it verbatim (zero re-extraction).
2. Move `sdf_model` to `self.device` if it is an `nn.Module`.
3. Build cache key:
   `(id(model), param_version, bbox, effective_resolution, isovalue,
     target_num_vertices, tolerance)`.
4. Return on cache hit.
5. On miss, purge entries for the same model whose `param_version` is stale.
6. Dispatch to `_extract` (fixed-resolution) or `_extract_to_target`
   (target-vertex-count) based on config.
7. FIFO-evict if full, then store the new `(v, f)`.

### `extract_and_compute_loss(sdf_model, loss_fn, enable_backward=True)`
1. Extract mesh via the cached pipeline.
2. Return a penalty loss `(1e4)` on empty meshes to keep training stable.
3. Detach vertices and set `requires_grad_(True)` so `loss_fn` can backprop
   to them.
4. Evaluate `loss = loss_fn(v, f)`.
5. Inference mode: return zero surrogate.
6. Training mode: compute `dL/dv` via `torch.autograd.grad`, then build the
   surrogate through `compute_gradient_flow`. Its `.backward()` yields
   correct gradients w.r.t. SDF parameters even though marching cubes is
   not differentiable.

Returns `(vertices, faces, loss, surrogate_loss, dL_dvertices)`.

### `compute_gradient_flow(vertices, dL_dvertices, sdf_model)` *(differentiability trick)*
Mathematical background: for an implicit surface $\{s(x;\theta)=c\}$,
moving a vertex $v$ along the unit SDF normal $n(v)$ by $\epsilon$ changes
$s(v)$ by $\epsilon$. Hence

$$\frac{\partial L}{\partial \theta}
  = \sum_{v} -\bigl\langle \tfrac{\partial L}{\partial v},\, n(v) \bigr\rangle
             \cdot \frac{\partial s(v)}{\partial \theta}.$$

Implementation:
1. Re-evaluate `s = sdf_model(v)` to obtain `g = d s/d v`.
2. Normalise `g` to unit normals `n`.
3. Project `dL/dv` onto `-n` to get the scalar `dL/ds` per vertex
   (detached: fixed coefficients).
4. Re-evaluate `s2 = sdf_model(v)` on a fresh graph so `θ` participates in
   autograd.
5. Return `sum(dL_ds * s2)`. `.backward()` gives the exact $\partial L/\partial \theta$.

### Cross-loss sharing helpers
- `extraction_key()` — hashable tuple
  `(bbox, effective_resolution, isovalue, target_num_vertices, tolerance)`.
  `TotalLoss` groups extractors with equal keys, extracts once, and replays
  the result into siblings via `set_shared_mesh`.
- `set_shared_mesh(v, f)` / `clear_shared_mesh()` — install/remove the
  injected mesh (instance-level, checked first by `extract_mesh`).
- `get_stats` / `reset_stats` / `log_stats_summary` — API-compatibility
  stubs for callers that used the older timing-heavy extractor.

---

## 4. Internal extraction pipeline

### `_empty()`
Return `(0×3, 0×3)` tensors on the right device (used for early exits).

### `_eval_points(sdf_model, points)` *(memory-bounded forward)*
1. Iterate `points` in chunks of `self.batch_size`.
2. `torch.no_grad()` — forward pass only.
3. CUDA: wrap in fp16 autocast (halves activation memory).
4. Squeeze singleton output dims, cast back to fp32 for marching cubes.
5. Concat chunks into a single 1-D tensor.

### `_eval_grid(sdf_model, res)`
1. Build per-axis `linspace` of length `res` inside the bbox.
2. Form the Cartesian product via `torch.meshgrid(..., indexing='ij')`.
3. Flatten to `(N, 3)` and evaluate via `_eval_points`.
4. Reshape to `(res, res, res)` numpy volume.

### `_marching_cubes(volume, res)`
1. Reject volumes whose range does not bracket the isovalue (nothing to
   extract).
2. Compute per-axis spacing `extent / (res - 1)`.
3. Prefer the PyMCubes C++ backend; fall back to `skimage.measure.marching_cubes`.
4. PyMCubes returns index-space coords → rescale by spacing, shift by bbox
   origin.
5. skimage applies spacing already → only origin shift needed.
6. Move to torch tensors on `self.device`.

### `_extract(sdf_model, res)`
1. `res ≤ 64`: dense full-volume marching cubes (sparse overhead not worth
   it).
2. `res > 64`: dispatch to `_extract_sparse`.

### `_extract_sparse(sdf_model, fine_res, coarse_res)` *(speed critical)*
1. Evaluate a dense `coarse_res³` volume.
2. Reject if it does not bracket the isovalue.
3. For every coarse cell, stack its 8 corner values; flag cells where
   `min < 0 < max` (contain the zero-crossing).
4. Dilate the active mask by one cell (3×3×3 neighbourhood) for safety.
5. Map each fine voxel to its coarse cell via
   `f2c = floor(i * (coarse-1) / (fine-1))`; inactive cells inherit their
   coarse value.
6. Upsample the coarse volume via nearest-neighbour indexing.
7. Decode flat active indices into `(ix, iy, iz)`; compute world-space
   points using bbox-scaled steps.
8. Evaluate the SDF **only** at those surface-adjacent points; write results
   into the upsampled volume in place.
9. Run marching cubes on the patched fine volume.

Complexity: $O(\text{coarse}^3 + \text{surface} \cdot (\text{fine}/\text{coarse})^2)$,
typically an order of magnitude fewer SDF calls than a dense fine grid.

### `_extract_to_target(sdf_model)` *(target vertex count)*
1. `[lo, hi] = target * (1 ± tolerance)`.
2. Warm-start `res` from `_cached_resolution` (success resolution of the
   previous call) or the configured default.
3. Loop up to 6 times:
   - Clamp `res` into `[8, 512]` and extract.
   - Empty mesh → cache `res` and return.
   - `lo ≤ n ≤ hi` → target hit; cache and return.
   - `n > hi` → cache a smaller predicted `res`, then decimate.
   - `n < lo` → scale `res` up by `sqrt(target / n)` (surface area scales
     as res²) and retry.
4. If the loop exits without convergence, decimate if still over, else
   return as-is.

### `_decimate(v, f, target)`
1. No-op if `fast_simplification` is unavailable or already within target.
2. Estimate face budget = `current_faces * target / current_vertices`.
3. Run `fast_simplification.simplify` on `(float64, int64)` arrays.
4. Return torch tensors on `self.device`.

---

## 5. Convenience helpers

### `clear_mesh_cache()`
Drop every entry in `_MESH_CACHE`. Rarely needed (cache auto-invalidates on
param updates via `_version`); useful in tests that rerun extraction with
identical params.

### `extract_sdf_to_temp_ply(sdf_model, config=None, output_name=...)`
1. Create an extractor bound to `config` (or YAML defaults).
2. Extract the mesh via the cached pipeline.
3. Create a unique temp dir under `$TMPDIR` and build the output path.
4. Wrap `(v, f)` in `trimesh.Trimesh(process=False)` and export as PLY.
5. Return the absolute file path.

---

## 6. How the five requirements are satisfied

| Requirement | Mechanism |
| --- | --- |
| 1. Differentiable mesh extraction | `extract_and_compute_loss` + `compute_gradient_flow` (normal-projection surrogate). |
| 2. Fast SDF → mesh | `_extract_sparse` (coarse-to-fine surface-only eval) + PyMCubes C++ backend + fp16 autocast. |
| 3. Target vertex/face count | `_extract_to_target` (adaptive resolution, warm-started) + `_decimate` via `fast_simplification`. |
| 4. Memory efficiency | Chunked forward (`batch_size`), fp16 autocast, sparse grid evaluation, FIFO-bounded cache (≤ 16 entries). |
| 5. Cross-loss mesh reuse | Global `_MESH_CACHE` keyed on `(model id, param version, bbox, resolution, isovalue, target)` + instance-level `set_shared_mesh` injected by `TotalLoss._inject_shared_meshes` after grouping by `extraction_key()`. |
