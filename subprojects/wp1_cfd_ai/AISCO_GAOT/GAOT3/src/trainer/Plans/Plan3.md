Plan to implement the final Plan3 cleanup.

Goal:
- Finish the GAOT3 trainer cleanup without changing mathematical behavior.
- Keep `python main.py -c <config.yaml>` as the public execution path.
- Preserve output paths, CSV formats, checkpoint behavior, W&B keys, metric database values, and the behavior checked by `tests/test_gaot2_gaot3_equivalence.py`.
- Apply the style and ownership rules from Plan3 to `GAOT3/main.py` and all Python files in `GAOT3/src/trainer`.

Hard acceptance rules:
- Preserve mathematical behavior, output paths, CSV formats, checkpoint behavior, W&B keys, metric database values, and the behavior checked by `tests/test_gaot2_gaot3_equivalence.py`.
- Keep `python main.py -c <config.yaml>` as the public execution path.
- Do not do style-only rewrites before the behavior baseline exists.
- `main_config` is imported once in `main.py` as a dict and modified only in `main.py`.
- Each trainer file should contain at most one trainer class, except `main.py`; utility modules under `src/trainer/utils` may keep utility functions.
- Public methods exist only when called from outside the class.
- Methods used only inside a class are private and start with one underscore.
- Prefer behavior safety over mechanical rule compliance when cleanup rules conflict.

Style and structure targets:
- New or modified trainer files should stay below 250 lines. Existing files already near the limit must not grow unless the behavior-preserving alternative is riskier.
- Every implemented class stores the loaded and modified config as `self.config: dict`.
- Config values are read from `self.config`; do not copy config values into long-lived attributes. Short-lived local aliases inside a method are allowed when they reduce repetition or clarify a computation.
- Every method and variable has a short descriptive name.
- Each Python file starts with one module-level triple-quoted docstring.
- The module docstring briefly describes how the class or module is called and which arguments it expects.
- Normalized and unnormalized values have names that clearly state their normalization state.
- Metric-specific normalization and denormalization happens in `metric_calculator.py`. Visualization, validation diagnostics, drag integration, and prediction CSV parity may denormalize elsewhere when that behavior is user-visible or required for equivalence.
- Inline `#` comments are allowed only for suspected unused code or genuinely non-obvious behavior.
- Trainer files below 100 lines are reviewed for possible integration, but may remain separate when the responsibility is distinct, integration would push a target file over 250 lines, or integration would mix unrelated responsibilities.

Current state to account for:
- The broad architecture from `Plan_expansion.md` is already mostly present.
- Files close to the 250-line limit are `Data_handler.py`, `validation_calculator.py`, `optimiser.py`, and `test_calculator.py`.
- Files below 100 lines are `runtime_handler.py`, `edge_handler.py`, `config_handler.py`, `checkpoint_handler.py`, `loss_plotter.py`, `metric_calculator.py`, `path_handler.py`, `model_handler.py`, and `result_writer.py`.
- `statistics_handler.py` is just over the 100-line threshold and still belongs in the cleanup audit.
- `loss_fn.py` still exposes module-level helper functions (`build_loss_fn`, `RelativeL2Loss`, `L2Loss`), which may break the strict one-class-per-trainer-file direction if they are treated as trainer helpers rather than utility compatibility functions.
- The existing implementation still passes around a trainer-like object. This is acceptable during cleanup if behavior is preserved, but public methods should be reduced to the methods actually called across class boundaries.

Phase 1: Freeze behavior before editing
- Find and record the exact smallest parity test command for GAOT3 vs GAOT2 before editing.
- Save the current file list, line counts, and public method list as the cleanup baseline.
- Record current generated artifacts expected by the equivalence test: prediction CSV headers, row counts, metric database keys, checkpoint names, loss plot outputs, and W&B metric keys if the test touches them. Prefer a repeatable audit command or script over manual notes.
- Do not perform style-only rewrites before the behavior baseline exists.

Phase 2: Classify files by responsibility and size
- Keep separate files when the responsibility is distinct and integration would push the target file over 250 lines.
- Integrate tiny files only where the result stays readable and below 250 lines.
- Preferred integrations:
  - `result_writer.py` into `training_orchestrator.py` if line count remains below 250.
  - `prediction_writer.py` has been integrated into `test_calculator.py` because the combined file remains below 250 lines.
  - `loss_plotter.py` into `optimiser.py` or `training_orchestrator.py` only if the receiving file remains below 250.
- Preferred files to keep separate despite being under 100 lines:
  - `config_handler.py`, because it is used directly by `main.py`.
  - `path_handler.py`, because it owns path mutation and datarow initialization.
  - `checkpoint_handler.py`, because checkpoint behavior is distinct and user-visible.
  - `model_handler.py`, because model construction depends on runtime-detected data shape.
  - `metric_calculator.py`, because Plan3 explicitly assigns metric normalization here.
  - `edge_handler.py`, because integrating it would bloat `Data_handler.py`, which is already at the 250-line ceiling.
  - `runtime_handler.py`, because integrating it would bloat `training_orchestrator.py`.

Phase 3: Enforce public/private method boundaries
- For every trainer class, search call sites for each method.
- Keep a method public only if another class, `main.py`, or a test calls it.
- Rename class-internal helpers to `_helper_name`.
- Update all call sites in the same patch as each rename.
- Expected public surface:
  - `ConfigHandler.parse_command_line`, `ConfigHandler.load_config`, `ConfigHandler.apply_overrides`
  - `TrainingOrchestrator.run`
  - `PathHandler.prepare_paths_and_datarow`
  - `RuntimeHandler.prepare_runtime`, `RuntimeHandler.serializable_config`
  - `DataHandler.prepare_data`, `DataHandler.sample_nodes_neural_field`
  - `StatisticsHandler.calculate_or_load_stats`
  - `EdgeHandler.update_pt_files_with_edges`
  - `ModelHandler.build_model`
  - `LossFn.calculate_loss`, `LossFn.calculate_train_loss`, plus any method called by validation or tests
  - `Optimiser.optimise`
  - `WandBLogger` logging, watch, artifact, and close methods used outside the class
  - `CheckpointHandler` train/test load and save methods used by orchestration or optimization
  - `ValidationCalculator.validate`, `ValidationCalculator.visualize_samples`
  - `TestCalculator.run_test`
  - `MetricCalculator.compute_metrics`
  - writer and plotter methods only if their files stay separate

Phase 4: Remove module-level trainer helpers
- Audit call sites for `build_loss_fn`, `RelativeL2Loss`, and `L2Loss` before moving them.
- If they are trainer-only helpers, move them into `LossFn` or replace their call sites with `LossFn(config).calculate_loss`.
- If older code or tests still import `RelativeL2Loss` or `L2Loss`, keep a small compatibility path and mark it with a suspected-unused inline comment only after confirming it is not used by the current GAOT3 path.
- Keep `main.py` free functions because it is the explicit exception to the one-class-per-file rule.
- Keep utility functions in `src/trainer/utils` only if they are treated as utilities rather than new trainer classes.

Phase 5: Normalize config access
- Check every class has `self.config: dict` in `__init__`.
- Replace long-lived copied config attributes with direct `self.config[...]` access.
- Temporary local variables and short-lived config section aliases are allowed inside a method when they reduce repetition or clarify a computation.
- Avoid storing config sections such as `self.dataset_config`, `self.model_config`, or `self.setup_config` on new or cleaned-up classes.
- If the trainer context still exposes these aliases for behavior compatibility, treat that as legacy compatibility and do not expand it.

Phase 6: Clean comments and docstrings
- Ensure each file starts with exactly one module-level triple-quoted docstring.
- Remove additional block comments and convert essential explanations into short code or names.
- Keep inline `#` comments only for suspected unused code or genuinely non-obvious behavior.
- Update each module docstring to say who calls the class and what argument it expects, for example: `TrainingOrchestrator is called by main.py with config: dict and runs train/test execution.`

Phase 7: Make normalized value names explicit
- Audit `loss_fn.py`, `validation_calculator.py`, `test_calculator.py`, and `metric_calculator.py`.
- Rename ambiguous variables such as `pred`, `target`, `values`, or `outputs` when they carry normalized data.
- Use names like `pred_normalized`, `target_normalized`, `pred_unnormalized`, and `target_unnormalized`.
- Keep metric-specific denormalization inside `metric_calculator.py`.
- Keep validation visualization, drag integration, and prediction CSV denormalization outside `metric_calculator.py` only when required to reproduce existing user-visible behavior. Use explicit normalized or unnormalized names in those paths.

Phase 8: Manage the 100-line and 250-line tension
- Do not blindly integrate every file under 100 lines, because several are separate to keep large classes below 250 lines.
- For every kept-small file, add a short note in the implementation commit or PR explaining why it remains separate.
- If strict interpretation is required, split the largest files first:
  - Move validation image generation out of `validation_calculator.py` only if a new class is accepted.
  - Move epoch CSV or telemetry details out of `optimiser.py` only if a new class is accepted.
  - Keep prediction CSV writing in `test_calculator.py` only while the file remains below 250 lines.
- When the 100-line and 250-line targets conflict, keep the responsibility boundary that is safest for behavior.

Phase 9: Verification after each cleanup batch
- Run a syntax/import check for `GAOT3/main.py` and `GAOT3/src/trainer`.
- Run the GAOT2/GAOT3 equivalence test after any behavior-adjacent change.
- Check line counts for all trainer Python files.
- Check each trainer file has at most one class.
- Check public method names against actual external call sites.
- Check generated prediction CSVs and metric database values against the baseline.

Suggested implementation order:
1. Baseline behavior and collect audit output.
2. Fix method visibility and call-site names without moving behavior.
3. Clean `loss_fn.py` module-level helpers.
4. Rename normalized and unnormalized variables in test, validation, loss, and metric code.
5. Clean docstrings and comments.
6. Decide tiny-file integration exceptions and document them.
7. Run equivalence and line-count checks.
8. Only then consider deeper refactors of the trainer context object.
