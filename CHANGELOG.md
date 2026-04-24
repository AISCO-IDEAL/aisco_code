# Changelog

All notable changes to AISCO_code are documented here.

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
