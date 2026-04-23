# Changelog

All notable changes to AISCO_code are documented here.

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
