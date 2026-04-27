Plan for restructuring the GAOT3 training loop.

High-level vision:
- Restructure the current training process without changing its mathematical behavior.
- Verify equivalence with `subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/tests/test_gaot2_gaot3_equivalence.py`.
- Make the new code easier to read, test, and maintain.
- Preserve all externally visible behavior needed by the equivalence test, including output paths, CSV formats, checkpoint behavior, W&B keys, and metric database values.

Behavioral preservation guardrails:
- The first implementation target is parity with the current GAOT3 behavior, not a cleaner behavior.
- Public execution must remain `python main.py -c <config.yaml>`.
- `test_gaot2_gaot3_equivalence.py` is the minimum acceptance test, but not the complete behavior contract.
- Existing paths, CSV headers, CSV row order, checkpoint priority, checkpoint timing, W&B keys, datarow keys, and rank-0-only side effects must be preserved unless this document explicitly says otherwise.

The following files shall be implemented or reimplemented. Additional files may be added only when they keep an existing class below the line limit or avoid mixing unrelated responsibilities.

In the `GAOT3` folder:
- `main.py`: loads the config, applies main-only config mutations, and calls the training orchestrator.

In the `GAOT3/src/trainer` folder:
- `training_orchestrator.py`: owns the high-level run flow and coordinates all trainer components.
- `optimiser.py`: owns optimizer behavior, scheduler behavior, and the epoch loop.
- `WandB_logger.py`: collects and logs values, images, checkpoints, and artifacts to W&B.
- `loss_fn.py`: calculates training loss.
- `validation_calculator.py`: calculates validation values, loss and generates validation images.
- `Data_handler.py`: loads and prepares data, including random downsampling when configured.
- `config_handler.py`: loads the config file, applies command-line override values used by the current run path, converts the config into the dict returned to `main.py`, and does not introduce a new config system.
- `path_handler.py`: resolves output paths, creates required directories, applies the current `setup.run_local` behavior when enabled, and initializes the existing datarow values used for final metrics output.
- `model_handler.py`: builds the current `GAOT3D` model, preserves runtime input/output-size detection, and applies only model setup needed by the current config.
- `checkpoint_handler.py`: loads checkpoints only through currently executed train/test behavior and saves the model checkpoint and copied config outputs currently produced by the run.
- `statistics_handler.py`: calculates or loads normalization statistics used by the current dataset path and keeps only the statistics behavior needed by supported current configs.
- `edge_handler.py`: keeps edge precomputation if it is reachable through any supported current config or required by equivalence.
- `test_calculator.py`: runs the current test/inference path and collects predictions, timings, and values needed by current metrics and CSV export while preserving the current normalized-to-unnormalized prediction flow.
- `metric_calculator.py`: computes only metrics used by the current active metric suite, owns metric-specific normalization or denormalization, and updates the existing datarow keys that are written today.
- `prediction_writer.py`: writes the prediction CSV files currently produced by test execution and preserves the CSV path, header, row, and numeric-value behavior expected by `test_gaot2_gaot3_equivalence.py`.
- `loss_plotter.py`: saves the current loss plot and `.npz` loss history outputs.
- `result_writer.py`: appends the final datarow to the configured metrics database CSV.

Style rules for all new files:
- Each new file shall contain exactly one class, except `main.py`.
- Public methods are allowed only when called from outside the class.
- Methods used only inside the class shall be private and start with a single underscore.
- No new file shall be longer than 250 lines.
- `main_config` shall be imported once in `main.py` as a dict and modified only in `main.py`.
- Every implemented class shall store the loaded and modified config as `self.config: dict`.
- Config values shall not be copied into separate variables or attributes. They shall be accessed directly through `self.config`.
- Every method shall have a short descriptive name that summarizes what the method does.
- All variables and attributes shall have short descriptive names.
- Each document may contain only one comment block: a module-level triple-quoted string at the top of the file.
- The module-level triple-quoted string shall briefly describe how the class is called and which arguments it expects.
- If a value is normalized, both the normalized and unnormalized values shall have names that clearly state their normalization state.
- Metric-specific normalization and denormalization shall happen in `metric_calculator.py`. Test prediction denormalization needed to reproduce existing prediction CSV values may happen in `TestCalculator` or a shared helper, but normalized and unnormalized variable names must make the value state explicit.
- Write a short comment after a # where you suspect the code could be unused.

Section 1: Current functionality movement into new files

The restructuring shall split the current responsibilities of `main.py`, `base.py`, `stat.py`, `optimizers.py`, `logger.py`, and `src/trainer/utils/*` into the files listed above. The new implementation must keep the same mathematical behavior and the same externally visible paths, CSV formats, checkpoint behavior, W&B keys, and metric database values used by `test_gaot2_gaot3_equivalence.py`.

- `main.py` keeps command-line parsing, loading the original config into one dict, applying command-line overrides, applying main-only config mutations, optional multiprocessing launch behavior, and the final call into `TrainingOrchestrator`. It must not become a second trainer class.

- `config_handler.py` owns reusable config-loading helpers only. It can contain the current `FileParser`, W&B override parsing, nested override application, and logging setup if this makes `main.py` shorter without introducing a new config system.

- `path_handler.py` owns the current `prepare_arg` path behavior: output path resolution, directory creation, `setup.run_local` path changes, dataset path changes, and initialization of `config["datarow"]`. It must preserve all current datarow defaults used by final metrics output.

- `training_orchestrator.py` owns the high-level run flow: path preparation, runtime setup, distributed initialization, CUDA rank selection, CPU/GPU device selection, seed setup, dtype selection, autocast context creation, data setup, model setup, logger setup, checkpoint loading, train/test branching, final result writing, logger closing, and distributed cleanup. It replaces the current direct construction of `StaticTrainer3D`.

- `Data_handler.py` owns dataset-related behavior: metadata lookup, order-file resolution, active-head selection, processed-data checks, transform construction, latent-token creation, collate construction, train/validation/test dataset construction, `DistributedSampler` construction, `DataLoader` construction, and neural-field node sampling.

- `statistics_handler.py` owns normalization statistics. It moves the current calculate/load/compute statistics behavior and returns the statistics required by `DataHandler`, `ValidationCalculator`, and `TestCalculator`.

- `edge_handler.py` is optional only if `dataset.update_pt_files_with_edges` is proven unreachable in all supported current configs and not required by equivalence. If any supported config can reach it, it must be implemented with the existing processed `.pt` mutation, MAGNO neighbor strategy parsing, edge/index/count creation, rank-0-only execution, and DDP barrier behavior.

- `model_handler.py` owns MAGNO config helpers, runtime input/output-size consumption, active-head decoder filtering, `GAOT3D(config=...)` construction, `.to(device)`, optional `DistributedDataParallel` wrapping, and model parameter counting.

- `loss_fn.py` owns `RelativeL2Loss`, `L2Loss`, `build_loss_fn`, scalar loss, field loss, masked multi-head loss, field/scalar combined loss, uncertainty-weighted combined loss, and validation loss reuse. Training and validation must use the same loss implementation.

- `optimiser.py` owns optimizer and scheduler construction, uncertainty-weight optimizer parameter groups, gradient clipping, scheduler stepping, the epoch loop, per-batch train steps, validation scheduling, epoch CSV logging, periodic checkpoint save triggers, test-during-training triggers, best-state restore, and all current optimizer telemetry keys. This includes gradient norms, scalar encoder diagnostics, scale-factor statistics, memory values, GPU memory values, train/validation W&B keys, and the `_epoch_metrics.csv` format.

- `WandB_logger.py` owns the current `WandbTracker` behavior and may rename it to `WandBLogger`. It keeps W&B initialization, model watching, scalar metric logging, image logging, checkpoint artifact upload, artifact download, and run closing.

- `checkpoint_handler.py` owns train and test checkpoint loading through `resume_from_artifact`, `resume_from_ckpt`, and `ckpt`; local checkpoint save/load calls; model checkpoint saving; W&B artifact save delegation; model config YAML saving next to the checkpoint; and copying the full run config to `config_save_path`.

- `validation_calculator.py` owns validation and visualization behavior: validation cache handling, full-mesh ground-truth loading, point-cloud and mesh drag integration, prediction updates, scalar/field losses, relative L2 calculation, DDP all-reduce of validation values, image generation, and validation W&B values.

- `test_calculator.py` owns the test dataloop, data-loading timing, inference timing, prediction/target collection, field sample collection, and coordination of metric calculation and prediction writing. It must preserve the current normalized-to-unnormalized value flow used for prediction CSV output and metric inputs.

- `metric_calculator.py` owns only metric functions used by the active test path and updates `config["datarow"]`. It owns metric-specific normalization or denormalization, but it must not change the values written by `prediction_writer.py`. Normalized and unnormalized values must be named clearly. `compute_drivaernet_metric` is needed for the current `drivaernet` metric suite. Other metric helpers should be moved only if they are reachable through current configs or required by equivalence.

- `prediction_writer.py` owns the exact prediction CSV files and headers currently produced by test execution, including per-head prediction CSVs, the general `_predictions.csv`, and field sample metric CSVs when field outputs are the primary result. The general `_predictions.csv` must remain byte-shape compatible with the current output because `test_gaot2_gaot3_equivalence.py` compares headers, row counts, and numeric values.

- `loss_plotter.py` owns loss history output. It must preserve both current outputs: the configured loss `.png` path and the `.npz` file next to it with the same arrays.

- `result_writer.py` owns final metrics database writing. It must preserve the current behavior of creating a dataframe with `datarow.keys()` when the database CSV does not exist and appending one row on rank 0.

- Existing utility functions in `src/trainer/utils/plot.py` can stay as utilities and should be called by `validation_calculator.py` instead of duplicated.

Section 2: New class call relationships

All classes listed above receive `config: dict` and store it as `self.config`. Config values are read from `self.config` directly instead of being copied into separate attributes. Shared runtime objects such as model, loaders, dtype, device, statistics, rank, and tracker are passed through return values or small context dictionaries.

- `main.py` loads and mutates the config dict, constructs `TrainingOrchestrator(config)`, and calls `TrainingOrchestrator.run()`.

- `TrainingOrchestrator.run()` first calls `PathHandler.prepare_paths_and_datarow()` and keeps the returned config with resolved paths and initialized `config["datarow"]`.

- `TrainingOrchestrator` prepares the runtime context before any data, model, optimizer, or W&B setup. The runtime context contains device, dtype, autocast context, rank, world size, and distributed state.

- `TrainingOrchestrator` calls `DataHandler.prepare_data(runtime_context)`. `DataHandler` calls `StatisticsHandler.calculate_or_load_stats(...)` and, only when required, `EdgeHandler.update_pt_files_with_edges(...)`. It returns loaders, latent tokens, collate behavior, active heads, metadata, input/output channel counts, and normalization statistics.

- `TrainingOrchestrator` calls `ModelHandler.build_model(data_context, runtime_context)` after data preparation because the model needs runtime-detected input and output sizes.

- `TrainingOrchestrator` constructs `LossFn(config)` and `Optimiser(config)`. If uncertainty weighting is enabled, the learnable loss parameters are created once and passed to `Optimiser` as an additional parameter group.

- `TrainingOrchestrator` starts `WandBLogger` after model construction and calls `WandBLogger.watch_model(model)` when W&B tracking is active.

- Before training, `TrainingOrchestrator` calls `CheckpointHandler.load_for_training(...)` when the config requests a resume or checkpoint load. Artifact downloads go through `WandBLogger`.

- During training, `TrainingOrchestrator` calls `Optimiser.optimise(training_context)`. The context contains model, loaders, latent tokens, data handler, loss function, validation calculator, checkpoint handler, W&B logger, runtime context, and path/config values needed by the loop.

- Inside each train batch, `Optimiser` asks `DataHandler` for neural-field sampling only when the active strategy needs it, calls the model, and asks `LossFn.calculate_train_loss(...)` for the loss tensor and per-head loss values.

- At validation intervals, `Optimiser` calls `ValidationCalculator.validate(...)`. The returned metrics dict is logged through `WandBLogger` and written to the epoch CSV using the existing key names.

- At visualization intervals, `Optimiser` calls `ValidationCalculator.visualize_samples(...)`. The calculator produces images and drag diagnostics, then logs them through `WandBLogger`.

- At best-model or periodic-save points, `Optimiser` calls `CheckpointHandler.save_checkpoint(...)`. If artifact logging is active, `CheckpointHandler` delegates upload to `WandBLogger`.

- After training, `TrainingOrchestrator` calls `LossPlotter.save_loss_history(...)` and then `CheckpointHandler.save_model_config(...)` plus `CheckpointHandler.copy_run_config(...)` on rank 0.

- If testing is enabled, `TrainingOrchestrator` calls `CheckpointHandler.load_for_testing(...)` and then `TestCalculator.run_test(test_context)`.

Distributed test behavior:
- If distributed training is initialized, only rank 0 may load the test checkpoint, run test inference, write prediction CSVs, append the metrics database, or write final artifacts.
- Non-rank-0 processes must wait at the same barriers as the current implementation before skipping test execution.
- Rank 0 must wait for checkpoint writes before loading a checkpoint for test execution.

- `TestCalculator` calls `MetricCalculator.compute_metrics(...)` with collected predictions, targets, field values, metric suite, metadata, statistics, and `config["datarow"]`. `MetricCalculator` updates the datarow keys that are written today and owns all metric-specific normalized/unnormalized value names.

- `TestCalculator` calls `PredictionWriter.write_predictions(...)` after metric calculation so CSV writing stays separate from metric calculation.

- At the end of the run, `TrainingOrchestrator` calls `ResultWriter.append_datarow(...)` once on rank 0, closes `WandBLogger`, and cleans up distributed training if it was initialized.
