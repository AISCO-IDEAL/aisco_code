# Debug System Guide

## Overview

The debug system is controlled by a single class, `DebugManager`, and a single config file, `configs/debug.yaml`.
The class exposes four hook methods that you call inside a training loop.
Zero code changes are needed to turn features on or off — everything is driven by the YAML.

---

## The four hooks

| Method | What it does |
|---|---|
| `dbg.log_run_info(...)` | One-time dump of architecture, device, run config at training start |
| `dbg.log_values(iteration, ...)` | Per-iteration: loss, components, gradients, params, optimizer state, CUDA memory, loss diagnostics |
| `dbg.log_stats(iteration, ...)` | Per-iteration: time per iter, memory delta, convergence rate |
| `dbg.check_warnings(iteration, ...)` | Per-iteration: NaN/Inf, zero grads, boundary violations, slow convergence, solver anomalies |
| `dbg.check_insights(iteration, ...)` | Per-iteration: step-size collapse, loss plateau, gradient explosion/vanishing |

All hooks are no-ops when their controlling toggle is off, so they are safe to leave in production code.

---

## Turning things on and off

### Master switch — single line to enable/disable everything

```yaml
# configs/debug.yaml
GLOBAL_DEBUG: true   # full diagnostics
GLOBAL_DEBUG: false  # zero overhead
GLOBAL_DEBUG: null   # respect granular toggles below
```

### The four main toggles (only active when GLOBAL_DEBUG is null)

```yaml
print_raw_values: true     # enables log_values()
show_stats: true           # enables log_stats()
generate_insights: true    # enables check_insights()
emit_warnings: true        # enables check_warnings()
```

### Sub-toggles for log_values (sections within the values dump)

```yaml
log_loss_components: true       # weighted per-component loss table
log_per_component_grads: true   # gradient norm per loss component
log_param_stats: true           # min/max/mean/std per parameter
log_optimizer_state: true       # Adam internal state
log_cuda_memory: true           # CUDA memory counters
log_alm_state: true             # ALM multipliers
log_run_info: true              # one-time architecture dump
log_loss_diagnostics: true      # per-loss diagnostic fields (see below)
```

### Per-loss, per-field control for loss diagnostics

`log_loss_diagnostics` can be `true` (all losses, all fields), `false` (nothing), or a dict:

```yaml
log_loss_diagnostics:
  bb_v4_2:
    violation_frac: true
    n_points: true
    sdf_pred_min: false    # suppress noisy / low-value field
  eikonal_v0: false        # suppress this loss entirely
  # any loss not listed defaults to: show ALL its fields
```

---

## Tuning thresholds

All thresholds that control when a `[WARNING]` or `[INSIGHT]` fires are in the YAML.
None are hardcoded.

```yaml
# Warning thresholds
warning_nan_threshold: 1.0e+10       # total loss magnitude
warning_grad_zero_atol: 1.0e-12      # gradient norm below this → "zero"
warning_convergence_patience: 50     # stalled iterations before warning
warning_bb_violation_frac: 0.95      # fraction of BB-violating points
warning_eikonal_grad_norm_tol: 0.5   # |grad_norm_mean - 1.0| tolerance
warning_smoothness_laplacian_max: 1.0e+4   # max absolute laplacian
warning_cruising_speed_min_ms: 5.0   # m/s threshold for "very slow" warning

# Insight thresholds
insight_step_shrink_ratio: 0.01          # step_size / initial_step_size
insight_grad_explosion_threshold: 1.0e+6 # grad_norm above this
insight_grad_vanish_threshold: 1.0e-8    # grad_norm below this
insight_plateau_window: 10               # number of recent iters to inspect
insight_plateau_spread_ratio: 1.0e-7     # spread / |loss| below this → plateau
```

---

## How to add logging to a new loss function

### Step 1 — Collect diagnostics inside the loss and return them

```python
# src/losses/my_new_loss.py

def my_new_loss(sdf_pred, target, ...) -> tuple[torch.Tensor, dict]:
    # ... compute loss_value ...

    diagnostics = {
        "n_points":    sdf_pred.shape[0],
        "error_mean":  error.mean().item(),
        "error_max":   error.max().item(),
        "has_nan":     bool(torch.isnan(loss_value).any()),
    }
    return loss_value, diagnostics
```

Key rules:
- All values in the dict must be plain Python scalars (`float`, `int`, `bool`), not tensors — call `.item()`.
- Use human-readable key names with units if relevant, e.g. `"error_mean (m)"`.
- Include `"has_nan"` / `"has_inf"` flags as bools so `check_warnings` can pick them up.

### Step 2 — Store the dict in TotalLoss

```python
# src/losses/total_loss.py

class TotalLoss:
    def __init__(self, ...):
        ...
        self._last_my_new_loss_diagnostics: dict | None = None   # add this field

    def forward(self, ...):
        ...
        my_loss_val, my_diag = my_new_loss(...)
        self._last_my_new_loss_diagnostics = my_diag
        ...

    def get_my_new_loss_diagnostics(self) -> dict | None:        # add this getter
        return self._last_my_new_loss_diagnostics
```

If your loss uses the standard `_last_diagnostics` convention (like `bb_v4_2`, `eikonal_v0`, etc.),
`collect_loss_diagnostics` already picks it up automatically — skip the custom getter.

> **The `_last_diagnostics` convention**: If your loss class has an attribute named
> `_last_diagnostics` that is set after every `forward()` call, `collect_loss_diagnostics()`
> will find it automatically via `getattr(loss_fn, "_last_diagnostics", None)`.
> Use this for new standard losses. Use a custom getter only for losses with unusual call patterns
> (like `cruising_speed`, which is called inline rather than as a loss object).

### Step 3 — Wire the getter into collect_loss_diagnostics (if using a custom getter)

```python
# src/utils/debug_manager.py  →  collect_loss_diagnostics()

def collect_loss_diagnostics(self, total_loss) -> dict[str, dict]:
    all_diags = {}

    # ... existing code for standard losses ...

    # Your new loss (custom getter path)
    my_diag = None
    if hasattr(total_loss, "get_my_new_loss_diagnostics"):
        my_diag = total_loss.get_my_new_loss_diagnostics()
    if my_diag is not None:
        all_diags["my_new_loss"] = my_diag

    return all_diags
```

### Step 4 — Add warnings for anomalous values (optional)

Open `check_warnings()` in `debug_manager.py` and add a block inside the
`for ln, diag in loss_diagnostics.items():` loop:

```python
if ln == "my_new_loss":
    if diag.get("has_nan"):
        warnings_list.append(f"Iter {iteration}: '{ln}' produced NaN output!")
    err = diag.get("error_mean (m)")
    if err is not None and err > 10.0:      # use self.* for configurable thresholds
        warnings_list.append(f"Iter {iteration}: '{ln}' mean error is high: {err:.3e} m")
```

If the threshold should be configurable:
1. Add it to `debug.yaml`: `warning_my_loss_error_threshold: 10.0`
2. Read it in the constructor: `self.my_loss_error_threshold = float(cfg.get("warning_my_loss_error_threshold", 10.0))`
3. Use `self.my_loss_error_threshold` instead of the literal `10.0` in `check_warnings`.

### Step 5 — Add it to debug.yaml (optional, to control fields)

```yaml
log_loss_diagnostics:
  my_new_loss:
    n_points:       true
    error_mean (m): true
    error_max (m):  false   # too noisy, hide by default
    has_nan:        true
```

Any field not listed defaults to shown when the loss entry is a dict.
Omitting the loss name entirely also defaults to showing all its fields.

---

## How to add debug hooks to a new training script

```python
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.debug_manager import DebugManager

dbg = DebugManager()   # loads configs/debug.yaml automatically

# Once, before the loop:
dbg.log_run_info(
    model=model,
    device=device,
    learning_rate=lr,
    training_config=cfg,
)

# Inside the loop:
for iteration in range(n_iters):
    loss, loss_components = compute_loss(...)
    loss.backward()
    optimizer.step()

    grad_norm = compute_grad_norm(model)   # optional helper, see existing loops
    loss_diags = dbg.collect_loss_diagnostics(total_loss_obj)  # optional

    dbg.log_values(
        iteration,
        loss=loss.item(),
        loss_components=loss_components,
        grad_norm=grad_norm,
        loss_diagnostics=loss_diags,
    )
    dbg.log_stats(iteration, loss=loss.item(), grad_norm=grad_norm, lr=get_lr(optimizer))
    dbg.check_warnings(iteration, loss=loss.item(), grad_norm=grad_norm, loss_diagnostics=loss_diags)
    dbg.check_insights(iteration, loss=loss.item(), lr=get_lr(optimizer), grad_norm=grad_norm)
```

All arguments to every hook are optional keyword arguments.
Pass only what you have; omit what you don't.

---

## Quick reference: config file location

```
configs/debug.yaml
```

Loaded automatically by `DebugManager()` relative to the file's install path.
To use a custom config (e.g. in tests):

```python
dbg = DebugManager(config={"GLOBAL_DEBUG": False})
dbg = DebugManager(global_override=True)   # force everything on regardless of file
```
