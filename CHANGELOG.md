# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.6.4] - 2026-04-30

### Added

- `MeshAssemblyModeV4` now writes one GIF per camera perspective into a
  dedicated `cam{idx}/` sub-directory inside the mode's output folder, so
  each angle's full optimization trajectory is immediately accessible after a
  run without any post-processing step.  The `frame_storage` attribute is now
  a nested list (`list[list[Any]]`) — one inner list per camera — replacing
  the previous flat list (Plan 026).

### Removed

- `v2/src/visuals/gif_hack.py`: manual de-interleave post-processing script
  superseded by the native per-camera subfolder output in
  `MeshAssemblyModeV4` (Plan 026).

### Changed

- `VisualisationManager.update()` (visual_wrapper): W&B live-stream upload
  now uses an `isinstance` guard to distinguish per-camera nested
  `frame_storage` from flat-list modes, uploading the latest frame per camera
  on key `renders/{mode_name}/cam{idx}` (Plan 026).
- `VisualisationManager.finalize()` (visual_wrapper): W&B GIF upload now
  recurses into `cam{idx}/` sub-directories using `rglob`; per-camera GIFs
  are uploaded to `animations/{mode_name}/cam{idx}` instead of all writing
  to the same `animations/{mode_name}` key (Plan 026).

## [0.5.2] - 2026-04-27

### Fixed

- `total_loss` no longer appears as a competitor slice in
  `loss_competition/normalized_pie`, `loss_competition/normalized_bar`,
  `loss_competition/optimizer_pie`, and `loss_competition/optimizer_bar`
  charts in both `total_loss.py` and `diversity_manager.py`.
  Root cause: `normalized_floats` and `optimizer_floats` competition dicts
  were built without exclusion guards, so the aggregate key was included
  alongside its own addends, producing an artificial ~50% slice.
  Fixed by adding `k != "normalized_losses/total_loss"` and
  `k != "optimizer_losses/total_loss"` guards to both files, plus a
  defensive guard on `weighted_floats` in both files.
- `diversity_manager._safe_pie()` no longer passes `NaN`/`Inf`/non-positive
  values to `matplotlib`, which could silently produce broken charts.
  Hardened with the same `math.isfinite(v) and v > 0` filter used in
  `total_loss._safe_pie()`.
- Added regression tests in
  `tests/wp4_aeroshell_optimizer/v2/test_total_loss.py` and new file
  `tests/wp4_aeroshell_optimizer/v2/test_chart_helpers.py` to prevent
  reintroduction of both issues.
