# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
