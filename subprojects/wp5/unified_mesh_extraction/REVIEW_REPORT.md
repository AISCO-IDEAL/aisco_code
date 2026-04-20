# WP5 Unified Mesh Extraction — Review Report

## 1. Files & Consumers

### Tool implementation
- `/AISCO_code/subprojects/wp5/unified_mesh_extraction/sdf2mesh.py` — `MeshSDFExtractor`, `MeshExtractionConfig`, `extract_sdf_to_temp_ply`
- `/AISCO_code/subprojects/wp5/unified_mesh_extraction/__init__.py` — public re-exports
- `/AISCO_code/subprojects/wp5/unified_mesh_extraction/mesh_extraction_config.yaml` — default config

### Consumers (import `MeshSDFExtractor` / `load_mesh_extraction_config`)
| File | Usage |
|------|-------|
| `/AISCO_code/subprojects/wp3_solar_estimator/v3B_2/solar_loss_v3B_2.py` | Instantiates extractor, calls `extract_and_compute_loss()` |
| `/AISCO_code/subprojects/wp3_solar_estimator/v6/solar_loss_v6.py` | Same pattern |
| `/AISCO_code/subprojects/wp3_solar_estimator/v7/solar_loss_v7.py` | Same pattern |
| `/AISCO_code/tests/wp5/src/test_unified_mesh_extraction.py` | Unit tests |
| `/AISCO_code/tests/wp5/src/benchmark_old_vs_new_extractor.py` | Benchmark old vs new |

### Config override pattern (all solar loss versions)
```python
mesh_cfg = load_mesh_extraction_config()                     # loads mesh_extraction_config.yaml
mesh_cfg = _replace(mesh_cfg, device=..., bbox=..., isovalue=...)  # runtime overrides
self.extractor = MeshSDFExtractor(config=mesh_cfg)
```
Only `device`, `bbox`, `isovalue` are overridden at runtime. All other params come from the YAML.

---

## 2. Config Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `method` | `"adaptive_active_cell_marching_cubes"` | Algorithm label (cosmetic; code dispatches on `max_refine_levels`) |
| `device` | `"cuda"` | Torch device for SDF evaluation |
| `isovalue` | `0.0` | Isosurface level in SDF field |
| `bbox` | `[[-5000,5000]]×3` | 3D bounding box for extraction |
| `base_resolution` | `64` | Initial uniform grid resolution (N³ points) |
| `subdivision_factor` | `2` | Children per axis when subdividing a cell (adaptive mode) |
| `max_refine_levels` | `0` | 0 = uniform mode; >0 = octree-like adaptive refinement |
| `leaf_cell_resolution` | `3` | Grid resolution inside each leaf cell (adaptive mode) |
| `sign_epsilon` | `1e-6` | Tolerance for classifying a cell as "active" (sign change) |
| `batch_size` | `65536` | Max points per SDF forward pass |
| `surface_area_tolerance` | `0.05` | Unused in current code (reserved) |
| `merge_vertices` | `true` | Merge duplicate vertices after marching cubes |
| `target_num_vertices` | `null` | If set: auto-adapt resolution + decimate to reach this count |
| `vertex_count_tolerance` | `0.1` | ±10% window around target vertex count |
| `verbose` | `false` | Enable extra print logging |

---

## 3. How the Tool Works

### Extraction pipeline (2 modes)

**Mode A — Uniform + Sparse (`max_refine_levels=0`, default):**
1. If `base_resolution ≤ 64`: evaluate SDF on full N³ grid, run `skimage.measure.marching_cubes`.
2. If `base_resolution > 64`: **coarse-to-fine sparse evaluation**:
   - Evaluate SDF on a coarse 64³ grid.
   - Find "active" coarse cells (sign change across 2×2×2 corners), dilate by 1 cell.
   - Map active coarse cells → fine grid mask. Only evaluate SDF at active fine-grid points.
   - Fill inactive fine-grid points with nearest-neighbor from coarse grid.
   - Run marching cubes on the assembled fine volume.

**Mode B — Adaptive octree (`max_refine_levels > 0`):**
1. Evaluate SDF on base_resolution³ grid, identify active cells (sign change).
2. For each refinement level: subdivide active cells by `subdivision_factor`, re-evaluate corners, keep only active children.
3. In each leaf cell: evaluate SDF on `leaf_cell_resolution³` local grid, run marching cubes.
4. Concatenate and merge all leaf meshes.

### Vertex count targeting (`target_num_vertices != null`)
1. Start at `base_resolution` (or cached resolution from previous call).
2. Extract mesh. If vertex count is within `[target × (1-tol), target × (1+tol)]` → done.
3. If overshoot → decimate with `fast_simplification` (quadric decimation).
4. If undershoot → scale resolution up by `√(target/current)`, retry. Max 6 iterations.
5. Cache final resolution for warm-start on next call.

### Gradient flow (`extract_and_compute_loss`)
1. Extract mesh (no grad).
2. Compute `loss_fn(vertices, faces)` with grad w.r.t. vertices.
3. Project `dL/dvertices` onto SDF normals → scalar `dL/ds` per vertex.
4. Create surrogate: `sum(dL_ds * sdf(vertices))` — backprop through this to reach SDF parameters.

---

## 4. Logging: Current State & Remaining Work

### Implemented (as of 2026-04-20)
- ✅ **Named `logging` logger** (`_LOGGER = logging.getLogger("sdf2mesh")`). All internal messages go through `self.logger.{debug,info,warning}`. The raw `print()` / `[TIMING sdf2mesh]` / `[sdf2mesh WARNING]` forms are gone from `sdf2mesh.py`.
- ✅ **`verbose` flag now just toggles `DEBUG` level** (`self.logger.setLevel(logging.DEBUG)`); no special-case branches.
- ✅ **Stats accumulator** — `_TimingStats` dataclass tracks `calls`, `empty_calls`, `decimate_calls`, `total_extract_s / loss_s / autograd_s / grad_flow_s / total_s`, and `vertex_counts`. Exposed via `extractor.get_stats()` and `extractor.reset_stats()`.
- ✅ **Epoch summary API** — `extractor.log_stats_summary(level=logging.INFO)` prints a single compact line:
  `sdf2mesh summary: calls=18 non_empty=12 empty=6 decimate=4 extract=38.2s loss=8.1s autograd=0.4s grad_flow=0.2s total=46.9s verts[min/median/max]=172/1930/1998`.
- ✅ **WP3 raw-print sources removed** — the two `print(f"[TIMING] …")` calls in `solar_loss_v3B_2.py` now use `logging.getLogger("sdf2mesh").debug(...)`, so per-call timing no longer floods stdout at default log level.

### Remaining / caller responsibilities
- 🔲 **Invoke `log_stats_summary()` + `reset_stats()` at epoch boundaries.** The accumulator is ready, but the training loop (`wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py` / `TotalLossV2`) must call them once per epoch on each extractor it owns (solar, developability, storage-utils preview). Without this, stats grow unboundedly across a full run.
- 🔲 **Propagate a run/epoch/vector tag into log records.** Today the logger has no contextual tag, so messages from different diversity vectors are indistinguishable. Recommended: a lightweight `logging.LoggerAdapter` that injects `{"epoch": ..., "vector": ...}` via `extra=`, set once per vector from the training loop.
- 🔲 **Root-logger configuration.** Nothing in the package configures handlers/formatters, so messages only appear when the application (or a test) calls `logging.basicConfig(...)`. Document this in the module docstring, or add an idempotent `logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")` in `main_v2.py` at startup.
- 🔲 **Downgrade the "Empty mesh returned, using penalty loss." warning** to `debug` for the first few epochs of training (it is expected while the SDF warms up). Simple gating: `level = logging.DEBUG if self._stats.calls < warmup_calls else logging.WARNING`.

---

## 5. Performance: Current State & Remaining Work

### Implemented (as of 2026-04-20)
- ✅ **S1 — Sparse active-cell mask caching.** `_extract_mesh_sparse` caches `(fine_active, fill_volume)` keyed by `(fine_res, coarse_res, bbox)` and reuses it for up to `_sparse_cache_refresh_interval = 8` consecutive calls. Auto-invalidates when the assembled volume no longer brackets `isovalue`.
- ✅ **S3 — Per-point cache removed.** `_evaluate_points` is now a straight batched `torch.no_grad()` loop over `self.batch_size`. No Python-level point dictionary.
- ✅ **S4 — PyMCubes backend when available.** `_marching_cubes` uses `mcubes.marching_cubes` if importable, falls back to `skimage.measure.marching_cubes`. Both paths share the same spacing/origin arithmetic.
- ✅ **S5 — Half-precision SDF eval on CUDA.** `self._use_mixed_precision = self.device.startswith("cuda")`; `_evaluate_points` and `_evaluate_grid` wrap the SDF forward in `torch.autocast(device_type="cuda", dtype=torch.float16)` under `torch.no_grad()`. CPU path stays fp32.
- ✅ **S6 — GPU-side active-point coordinate generation.** `_extract_mesh_sparse` builds `active_pts` directly on `self.device` via `torch.stack` from decoded `(ix, iy, iz)` indices — no 8M-point CPU→GPU transfer.
- ✅ **Warm-start adaptive resolution.** `_cached_resolution` is remembered across calls and clamped to `[min_res=8, max_res=512]` with √(target/current) rescaling. Overshoots now update the cache to a smaller resolution instead of staying locked at the overshoot value.
- ✅ **Coarse/fine split threshold.** `_extract_mesh_uniform` auto-routes to the sparse path when `base_resolution > 64`, so callers get the fast path without opt-in.

### Remaining / open items
- 🔲 **S2 — Cap training resolution.** Default YAML still has `target_num_vertices: null`, so a model whose SDF is far from the isovalue can still briefly push `base_resolution` toward 512 before the adapt loop corrects. Two safe options:
  1. Set `target_num_vertices: 30000` and `vertex_count_tolerance: 0.2` in `mesh_extraction_config.yaml` for training runs. This engages the adaptive loop plus the decimation shortcut from day one.
  2. Add a `max_training_resolution` field (defaulting to 128) consulted by `extract_mesh` before the adaptive loop — visualisation-only callers can then bump it via `_replace(config, max_training_resolution=512)`.
- 🔲 **S7 — Parallel extraction across diversity vectors.** Still sequential. Two concrete paths:
  - **CUDA streams**: launch `N` `_extract_mesh_sparse` calls on separate `torch.cuda.Stream`s; the SDF forward passes are the bottleneck and are stream-safe under `torch.no_grad`.
  - **`torch.compile` + `torch.vmap`**: vectorize the SDF over a `(N_vectors, batch, 3)` tensor so a single forward covers all vectors — requires the solar wrapper to batch input latent codes.
  Estimated upside: ~4× on 6 vectors given current ~5 s per vector.
- 🔲 **S8 — Adaptive `batch_size`.** Still a static `65536` from YAML. Cheap heuristic: double `batch_size` on each successful call until SDF forward time per batch stops decreasing; halve on OOM. Safe to keep behind a flag (`auto_batch_size: true`).
- 🔲 **Re-use `fast_simplification` output across epochs.** After an overshoot+decimate, the next call with a small SDF update will overshoot again and re-decimate. A cheap heuristic: remember `(res → decimate_ratio)` from the previous epoch and start the next epoch at the implied post-decimate resolution — avoids S2's decimation entirely once warm.
- 🔲 **Sparse cache invalidation signal.** `_sparse_cache_refresh_interval = 8` is a hard-coded blind refresh. Better: invalidate when the L∞ distance between successive coarse-grid evaluations exceeds a threshold. Requires storing the previous coarse volume and a ~1 ms comparison.

### Updated priority ranking
1. **S2** — Turn on `target_num_vertices` in the training config (config-only change, zero code).
2. **Epoch-boundary `log_stats_summary()` wiring** — turns the ~300 debug lines per epoch into one info line.
3. **S7 (CUDA streams variant)** — biggest CPU+GPU win once #1 and #2 are in.
4. **Smart sparse cache invalidation** — replaces the blind 8-call refresh.
5. **S8 adaptive batch size** — marginal, save for profiling.

