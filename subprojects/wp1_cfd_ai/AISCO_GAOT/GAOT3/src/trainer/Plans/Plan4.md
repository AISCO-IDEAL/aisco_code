

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


New folder Structure:
- `trainer/`
  - `training_orchestrator.py`
    - Stays directly in `trainer` because it is the central entry point used by `main.py`.
  - `Plans/`
    - `Plan.md`
    - `Plan3.md`
    - `Plan4.md`
    - `Plan_expansion.md`
  - `handlers/`
    - `config_handler.py`
    - `path_handler.py`
    - `runtime_handler.py`
    - `checkpoint_handler.py`
    - `model_handler.py`
  - `data_pipeline/`
    - `Data_handler.py`
    - `edge_handler.py`
    - `statistics_handler.py`
  - `training_core/`
    - `optimiser.py`
    - `loss_fn.py`
    - `loss_plotter.py`
  - `validation_and_logging/`
    - `validation_calculator.py`
    - `test_calculator.py`
    - `metric_calculator.py`
    - `WandB_logger.py`
  - `utils/`
    - `setup.py`
    - `plot.py`
    - `metric.py`

Folder rules:
- Every folder contains at most five files.
- `training_orchestrator.py` remains in `trainer/`, not in a subfolder.
- `Plans/` contains only planning Markdown files.
- `handlers/` contains only setup, path, runtime, checkpoint, and model construction classes.
- `data_pipeline/` owns data loading, edge generation, and dataset statistics.
- `training_core/` owns training loop support: optimization, loss calculation, and loss plotting.
- `validation_and_logging/` owns validation, test prediction output, metric calculation, and W&B logging.
- `utils/` remains for function-only helper modules and should not receive trainer classes.

Migration order:
1. Create the folders without changing behavior.
2. Move planning files into `Plans/`.
3. Move handler modules and update imports in `main.py` and `training_orchestrator.py`.
4. Move data pipeline modules and update imports in handler, validation, and training files.
5. Move training core modules and update imports in `training_orchestrator.py`, validation, and test code.
6. Move validation and logging modules and update imports in `training_orchestrator.py` and `optimiser.py`.
7. Run an import check before any style cleanup.

Import safety:
- Prefer package imports from `src.trainer.<folder>.<module>` where the existing execution path supports them.
- If `python main.py -c <config.yaml>` currently depends on direct local imports, keep the smallest compatibility shim needed until all moved imports work.
- Do not combine folder movement with method renaming or mathematical cleanup in the same patch.
- After each folder batch, run the GAOT3 import/syntax check and then the GAOT2/GAOT3 equivalence test if imports or call paths changed.
