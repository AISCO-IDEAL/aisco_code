# Changelog

All notable changes to AISCO_code are documented here.

## [0.6.3] — 2026-04-28 (Plan 028)

### Fixed

- **BUG: Stale PLY surface cache** (`pretraining_latent.py`) — `_load_or_build_surface_cache`
  cache key previously included only the PLY filename stem, so an in-place PLY update
  (same path, different bytes) silently returned stale surface pts.  The key now includes
  an 8-char SHA-256 content hash of the PLY file bytes; any byte-level change forces a
  cache miss and a fresh rebuild.  Existing `.pt` cache files written without the content
  hash are automatically bypassed (cache miss) — delete `sdf_cache_dir` to reclaim disk
  space after upgrading.

### Changed

- **`sampling_visualization/core.py` renamed** to `points_SDFmesh_to_ply.py`.  All imports
  updated (`__init__.py`, `pretraining_latent.py`).
- **Domain-pts naming cleanup** (`pretraining_latent.py`) — removed all stale "eikonal"
  labels from variables that hold domain points used for the L1-all loss:
  - Config key `vis_show_eikonal_pts` → `vis_show_domain_pts` (backward-compat alias
    retained in code: falls back to `vis_show_eikonal_pts` if new key absent).
  - Python variable `eikonal_pts_list` → `domain_pts_list`.
  - `_write_epoch_vis_snapshot` parameter `eikonal_pts_per_slot` → `domain_pts_per_slot`.
  - `_write_epoch_vis_snapshot` kwarg `show_eikonal_pts` → `show_domain_pts`.
  - `_make_vis_legend` parameter and legend text updated to reflect L1-all regime.
- **`_make_vis_legend` rewritten** — legend text now accurately describes the L1-all loss
  regime (GREEN = surface half, GREY = domain half, both fed to L1-all loss).  All MSE and
  eikonal loss references removed from the legend.
- **`_compute_input_grad_norms` docstring fixed** — clarified that the probe intentionally
  uses MSE (not the training L1-all loss) and explains why MSE is correct for a
  gradient-sensitivity diagnostic.
- **`main_v2.py` W&B forwarding** — `_maybe_run_latent_pretraining` now accepts
  `use_wandb: bool` and forwards it to `pretraining_latent(...)`.  When the global
  optimizer runs with W&B enabled, pretraining metrics (`l1_all`, `slot_N_loss`,
  `latent_grad_norm`) are now logged to the active W&B run.

## [0.6.2] — 2026-04-27 (Plan 022 Exp-B)

### Changed

- **Loss strategy** (`pretraining_latent.py`) — Replaced MSE-on-surface + eikonal with **L1 on all
  points** (surface + domain). Domain GT SDF (previously discarded) is now retained from
  `_sample_points_for_shape` and included in the loss. `_compute_eikonal_latent` is no longer
  imported or called. W&B logs `l1_all` instead of `eikonal`.
- **W&B metadata** — `loss_type`, `sampling_strategy`, and `run_group` are now read from
  `training_loop` config keys (optional; fall back to sensible defaults) instead of being
  hardcoded.
- **`pretraining_latent_config.yaml`** — `eikonal_weight` retained for YAML schema compatibility;
  not applied by the training loop.

### Added

- `pretraining_latent_config_exp_b.yaml` — 2-slot experiment config (sphere + simple_bullet_blunt)
  used to validate the L1-all strategy. Kept as a reference config.

## [0.6.1] — 2026-04-27 (Plan 021)

### Added

- **Latent gradient telemetry** (`pretraining_latent.py`) — Every `print_interval` epochs the
  training loop now probes `‖∂L/∂z_input‖` and `‖∂L/∂xyz_input‖` using
  `_compute_input_grad_norms`. The ratio `z_grad / xyz_grad` is printed to the terminal
  (appended to the existing epoch line as `| z_grad=... | xyz_grad=... | ratio=...`) and
  logged to W&B as `latent_grad_norm`, `spatial_grad_norm`, `grad_norm_ratio`. A
  `[WARN: ratio<0.01]` suffix is printed when the latent gradient signal is below 1% of
  the spatial gradient — indicating potential latent collapse.

## [0.6.0] — 2026-04-27 (Plans 013 + 018)

### Fixed

- **BUG-001** (`pretraining_latent.py`) — All post-training and per-epoch mesh PLYs were
  byte-identical across slots because `MeshSDFExtractor`’s global cache did not include
  the active latent vector in its key. Fix: `clear_mesh_cache()` is now called immediately
  before each per-slot `_extract_mesh` call (both in the post-training export loop and in
  `_write_epoch_vis_snapshot`).

- **QA-001** (`pretraining_latent_config.yaml`) — `sdf_cache_dir` was hardcoded to a
  cluster-specific absolute path, causing `FileNotFoundError` for non-Euler developers.
  Default is now `null` (cache disabled). Developers opt in by setting an absolute path.

### Changed

- **`pretraining_latent_config.yaml`** — `z_radius` default changed from `1.0` to `2.0`.
  Spatial inputs enter the model as mm / 1000 ≈ [−5, 5]; `z_radius = 2.0` places latent codes
  at a comparable magnitude to the dominant spatial axis, improving latent discriminability
  from initialization.
  **⚠ Checkpoint break**: this changes `_compute_pretrain_config_hash` → different
  checkpoint directory. Any existing checkpoint under `z_radius: 1.0` will not be found
  by fast-resume. Either keep `z_radius: 1.0` explicitly or re-run pretraining.

### Added (Plan 013 — carried into this release)

- **Watertightness guard** (`_load_ply_tensors`) — raises `ValueError` with an actionable
  message when a PLY mesh is not watertight, preventing silent downstream failures.

- **Dual-band near-surface sampling** — `near_surface_tight_sigma` /
  `near_surface_wide_sigma` parameters in `training_loop` config for DeepSDF-style
  two-Gaussian surface sampling.

- **Precomputed SDF cache** (`_load_or_build_surface_cache`) — on-disk SHA256-keyed
  `.pt` cache for surface sample points; eliminates redundant BVH queries across runs.

## [0.5.1] — Plan 010: PLY Fix and Outer-Prism Domain Unification

### Fixed

- **`subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_latent.py`** — `_load_ply_tensors`
  now applies axis permutation [2,0,1] (PLY_Z→AISCO_X, PLY_X→AISCO_Y, PLY_Y→AISCO_Z),
  X-centering, and Z-grounding (Z_min = 100mm ground clearance). Corrects the 10× scale
  error (`ply_scale` changed from 1000.0 to 100.0 in config).

- **`subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v1_1.py`** — `_load_ply_mesh`
  and `_sample_points_for_shape` PLY branch updated with same axis permutation, X-centering,
  Z-grounding, and outer-prism cuboid domain sampling.

- **`subprojects/wp4_aeroshell_optimizer/v2/src/utils/losses_utils.py`** — `get_points`
  signature changed from `bbox_size: float` (cubic) to `bbox_lo: list[float], bbox_hi: list[float]`
  (per-axis cuboid). Now samples the outer-prism domain correctly.

### Added

- **`subprojects/wp2_assembler/version4/utils/reg_imp.py`** — new `get_outer_prism_bounds_mm()`
  function returning the outer-prism X/Y/Z bounds from `regulations_and_parameters.yaml`
  as the single source of truth for all domain sampling.

- **`tests/wp2_assembler/test_reg_imp.py`** — 7 TDD tests for `get_outer_prism_bounds_mm`. All pass.

- **`tests/wp4_aeroshell_optimizer/v2/test_ply_fix_and_domain.py`** — 13 TDD tests covering
  PLY axis permutation / centering / grounding and `get_points` per-axis cuboid sampling. All pass.

### Changed

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/pretraining_latent_config.yaml`** — all 5 PLY
  entries: `ply_scale 1000.0 → 100.0`. Removed `mesh_box_mm`. Changed `vis_mesh_display_mode`
  from `"faces"` to `"vertices"`.

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/losses_v0_config.yaml`** — `bbox_size: 10.0`
  replaced by `bbox_lo/bbox_hi` matching outer-prism dimensions. `connectedness_config.bbox`
  and `developability_config.mesh_bbox` updated to outer-prism mm values.

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/GINN.yaml`** — `diversity_bbox_size`:
  `10.0 → 5800.0` mm (car-length approximation).

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/neural_renderer_v0.yaml`** — `bbox`:
  `[[-5000,5000],...]` → outer-prism `[[-2900,2900],[-1150,1150],[0,1650]]`.

- **`subprojects/wp4_aeroshell_optimizer/v2/src/utils/kaolin_conversions.py`** — default
  `box_dim` updated from `[-5000,5000,...]` to outer-prism mm values.

- **`subprojects/wp4_aeroshell_optimizer/v2/src/losses/parent_loss.py`** — `get_points` call
  updated to use `bbox_lo`/`bbox_hi` keys from config.

---

## [Unreleased — Plan 005: Latent Population Pretraining]

> **Note:** Plan 005 targeted v0.3.0 but pyproject.toml was already at v0.5.0 when
> implementation ran. Version artifact is documented here; pyproject.toml is left unchanged.

### Added

- **`subprojects/wp4_aeroshell_optimizer/v2/src/utils/latent_utils.py`**
  New file. Provides `build_unit_circle_Z(num_shapes, z_dim, radius, device)` which places
  `z₀` at the origin and `z₁ … z_{N-1}` equidistantly on a unit circle in the (z[0], z[1])
  plane. Returns a non-differentiable `(N, z_dim)` tensor.

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/pretraining_latent_config.yaml`**
  New config for latent-population pretraining. Configures 6 slots (3 sphere + 3 cube),
  training hyperparameters, checkpoint path, and optional mesh extraction.

- **`subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_latent.py`**
  New file. Core `pretraining_latent(model, config, device)` function. Features:
  - DeepSDF-style batched training over all latent slots in one forward pass
  - Fast-resume: skips training if valid checkpoint exists
  - PLY validation before training starts
  - Early NaN guard (`SystemExit`) with per-slot diagnostics
  - CosineAnnealingLR scheduler
  - Saves `{model_state_dict, z_pretrained}` checkpoint

- **`subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v5_latent.py`**
  New CLI entry point for latent pretraining. Flags: `--config`, `--epochs`, `--no-mesh`,
  `--device`. Validates model protocol before calling `pretraining_latent`.

- **`tests/wp4_aeroshell_optimizer/v2/test_pretraining_latent.py`**
  27 TDD tests covering all new functionality. All pass (27/27).

### Modified

- **`subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v1_1.py`**
  Three new helpers appended:
  - `_compute_eikonal_latent(model, input_data, z_expanded, device)` — eikonal with explicit z kwarg
  - `_sample_points_for_shape(shape_type, shape_params, batch_size, device)` — surface-concentrated sampling
  - `_extract_mesh(model, pth_path, device, mesh_res, mesh_box)` — shared mesh extractor (refactored from pretraining_v4.py, architecture finding R1)

- **`subprojects/wp4_aeroshell_optimizer/v2/src/GINN/diversity_manager.py`**
  `DiversityTrainer.__init__` now accepts:
  - `initial_Z: Tensor | None = None` — inject pretrained latent population
  - `z_resample_start_epoch: int = 0` — Mode C warm-up guard (no resampling before this epoch)

- **`subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py`**
  - Added `_maybe_run_latent_pretraining(model, general_config, device)` gate function
  - `DiversityTrainer` instantiation now passes `initial_Z` and `z_resample_start_epoch`

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/GINN.yaml`**
  Added `z_resample_start_epoch: 0` with Mode A / Mode B / Mode C comment block.

- **`subprojects/wp4_aeroshell_optimizer/v2/configs/general_config.yaml`**
  Added `latent_pretraining: ""` to `active_configs` (empty string = disabled by default).

### Backward Compatibility

All changes are backward-compatible. `latent_pretraining: ""` disables the new code path
so existing runs are unaffected. `z_resample_start_epoch: 0` preserves existing Mode A / B
diversity behaviour.
