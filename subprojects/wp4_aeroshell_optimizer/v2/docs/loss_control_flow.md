# Loss Calculation Control Flow Overview

This document provides a high-level overview of how the loss calculation and backward-pass control flow are structured in the `wp4_aeroshell_optimizer` subproject, focusing on the latest V2 architecture (`TotalLossV2`, `ALMManager`, `main_v2.py`, and `DiversityTrainer`).

## 1. High-Level Architecture
The codebase handles two generations of models:
1. **Legacy Models (Single-Model)**: Call `model.optimise(...)` inside `main_v2.py`. Inside `.optimise()`, the older `TotalLoss` class computes, weights, and logs everything in one monolithic pass.
2. **Modern Models (Population-Based / GINN)**: Drive their training via `DiversityTrainer`. This uses a modular separation of concerns: `TotalLossV2` (computing & weighting), `ALMManager` (augmented lagrangian multipliers), and `DiversityLossManager`.

## 2. Control Flow in V2 (`DiversityTrainer`)
The main training loop resides in `DiversityTrainer.train()`. During each epoch, the workflow follows these steps:

### A. Pre-Extraction (Mesh Sharing)
Before calculating individual losses, `TotalLossV2` identifies all active loss objects that require a 3D mesh of the model.
- It calls `_inject_shared_meshes(model)` to extract the mesh exactly *once* per extraction configuration and shares it among all losses that need it, avoiding redundant marching-cubes operations.

### B. Loss Computation (`TotalLossV2.compute`)
For a single population member (or the single model), `DiversityTrainer` calls `TotalLossV2.compute(model, ...)`:
- Iterates over all active **objectives**, **constraints**, and **auxiliary** losses.
- Computes their raw, unweighted scalar tensors.
- Returns a `RawLossBundle` containing these separated tensors.
*Note: No state is updated here, and no multipliers are applied.*

### C. Loss Weighting (`TotalLossV2.weight`)
Immediately following computation, the trainer calls `TotalLossV2.weight(bundle, alm_manager, ...)`:
- **Objectives:** If Cruising Speed Estimator (CSE) is active, the raw drag and solar objectives are fed into the iterative CSE solver to yield a cruising-speed metric, which serves as the final objective scalar. Otherwise, manual weights are applied.
- **Constraints:** Passes constraints to the `alm_manager` context to get penalties ($ \lambda \cdot c + 0.5 \cdot \mu \cdot c^2 $). If ALM is disabled, manual weights from the config are applied.
- **Auxiliary:** Always manually weighted.
- **Output:** Returns a `WeightedLossResult` containing `.scalar` (the final combined loss that is attached to the PyTorch computation graph) and a dictionary of detached values for telemetry logging.

### D. Diversity Loss (Optional)
If a population > 1 is active, the `DiversityLossManager.compute()` method evaluates structural variance across the entire population and returns an additional gradient component for the model.

### E. Backward Pass
The `WeightedLossResult.scalar` (plus the diversity scalar, if applicable) has `.backward()` called on it per population member.

### F. ALM Multiplier Update (`ALMManager.update_multipliers`)
At the end of an epoch (or step), once all population members have been evaluated, `ALMManager.update_multipliers(...)` is invoked.
- It calculates the moving averages of the raw constraint values.
- Updates the internal state for $\lambda$, $\mu$, and $\nu$ according to the Augmented Lagrangian Method.
- These updated multipliers are then used in the *next* epoch's `TotalLossV2.weight()` step.

## 3. Class Responsibilities Summary

* **`main_v2.py`**: Entry point. Sets up WandB, loads configs, initializes the `model`, optionally pretrains, then hands off control to either `model.optimise()` or `DiversityTrainer`.
* **`TotalLossV2` (`total_loss_v2.py`)**: Pure functional component for evaluating loss closures (`compute`) and aggregating them (`weight`). Completely state-free regarding ALM or logging.
* **`ALMManager` (`alm.py`)**: Stateful manager for penalty multipliers. Evaluated by `TotalLossV2` to fetch the current epoch's multiplier magnitudes, and mutated by the trainer at the end of the epoch.
* **`TotalLoss` (`total_loss.py`)**: The deprecated legacy class that intertwines computation, ALM updates, logging, and mesh stats into a single `get_total_loss()` method. Expected to be slowly phased out over time.
