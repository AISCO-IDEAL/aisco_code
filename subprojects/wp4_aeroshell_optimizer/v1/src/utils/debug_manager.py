"""DebugManager for the WP4 Aeroshell Optimizer.

Centralised debug / diagnostics controller that reads ``configs/debug.yaml``
and exposes lightweight hooks for the optimization loop.

Usage
-----
::

    from subprojects.wp4_aeroshell_optimizer.v1.src.utils.debug_manager import DebugManager

    dbg = DebugManager()                       # loads configs/debug.yaml
    dbg = DebugManager(global_override=True)    # force everything on

The four hook families mirror the YAML toggles:

* **values**   - ``print_raw_values``
* **stats**    - ``show_stats``
* **insights** - ``generate_insights``
* **warnings** - ``emit_warnings``
"""

from __future__ import annotations

import math
import time
import tracemalloc
from pathlib import Path
from typing import Any

import torch
import yaml

# ── helpers ──────────────────────────────────────────────────────────────────

_SEPARATOR = "─" * 72
_SEP_THIN = "┄" * 72


def _load_debug_yaml() -> dict:
    """Load the debug YAML from the canonical configs/ location."""
    config_path = Path(__file__).resolve().parents[2] / "configs" / "debug.yaml"
    with config_path.open() as fh:
        return yaml.safe_load(fh)


def _fmt(v: Any) -> str:
    """Format a value for debug printing: tensors show shape+dtype, floats use sci notation."""
    if isinstance(v, torch.Tensor):
        return f"Tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
    if isinstance(v, float):
        return f"{v:.6e}"
    return repr(v)


# ── DebugManager ─────────────────────────────────────────────────────────────


class DebugManager:
    """Reads ``debug.yaml`` and exposes per-category enabled flags + hook
    methods that are effectively no-ops when disabled."""

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        config: dict | None = None,
        global_override: bool | None = None,
    ) -> None:
        cfg = config if config is not None else _load_debug_yaml()

        # Resolve the master switch -----------------------------------------
        master: Any = global_override if global_override is not None else cfg.get("GLOBAL_DEBUG")

        # print current time and datestamp for easier correlation with logs
        print(f"\n{_SEPARATOR}")
        print(f"[DebugManager] Initializing debug manager with GLOBAL_DEBUG={master}")
        print(f"[DebugManager] Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        if master is True:
            self.print_raw_values = True
            self.show_stats = True
            self.generate_insights = True
            self.emit_warnings = True
        elif master is False:
            self.print_raw_values = False
            self.show_stats = False
            self.generate_insights = False
            self.emit_warnings = False
        else:
            # Custom / None → respect granular toggles
            self.print_raw_values = bool(cfg.get("print_raw_values", False))
            self.show_stats = bool(cfg.get("show_stats", False))
            self.generate_insights = bool(cfg.get("generate_insights", False))
            self.emit_warnings = bool(cfg.get("emit_warnings", False))

        # Sub-toggles for values (only relevant when print_raw_values is on) --
        self.log_loss_components: bool = bool(cfg.get("log_loss_components", True))
        self.log_per_component_grads: bool = bool(cfg.get("log_per_component_grads", True))
        self.log_param_stats: bool = bool(cfg.get("log_param_stats", True))
        self.log_optimizer_state: bool = bool(cfg.get("log_optimizer_state", True))
        self.log_cuda_memory: bool = bool(cfg.get("log_cuda_memory", True))
        self.log_alm_state: bool = bool(cfg.get("log_alm_state", True))
        self.log_run_info: bool = bool(cfg.get("log_run_info", True))
        # log_loss_diagnostics can be:
        #   True  → show all fields for all losses
        #   False → show nothing
        #   dict  → per-loss, optionally per-field (see debug.yaml)
        raw_diag = cfg.get("log_loss_diagnostics", True)
        self.log_loss_diagnostics: bool | dict = raw_diag if isinstance(raw_diag, dict) else bool(raw_diag)

        # Tuning knobs ------------------------------------------------------
        self.stats_log_interval: int = int(cfg.get("stats_log_interval", 1))
        # Warning thresholds
        self.nan_threshold: float = float(cfg.get("warning_nan_threshold", 1.0e10))
        self.grad_zero_atol: float = float(cfg.get("warning_grad_zero_atol", 1.0e-12))
        self.convergence_patience: int = int(cfg.get("warning_convergence_patience", 50))
        self.bb_violation_frac: float = float(cfg.get("warning_bb_violation_frac", 0.95))
        self.eikonal_grad_norm_tol: float = float(cfg.get("warning_eikonal_grad_norm_tol", 0.5))
        self.smoothness_laplacian_max: float = float(cfg.get("warning_smoothness_laplacian_max", 1.0e4))
        self.cruising_speed_min_ms: float = float(cfg.get("warning_cruising_speed_min_ms", 5.0))
        # Insight thresholds
        self.insight_step_shrink_ratio: float = float(cfg.get("insight_step_shrink_ratio", 0.01))
        self.insight_grad_explosion_threshold: float = float(cfg.get("insight_grad_explosion_threshold", 1.0e6))
        self.insight_grad_vanish_threshold: float = float(cfg.get("insight_grad_vanish_threshold", 1.0e-8))
        self.insight_plateau_window: int = int(cfg.get("insight_plateau_window", 10))
        self.insight_plateau_spread_ratio: float = float(cfg.get("insight_plateau_spread_ratio", 1.0e-7))

        # Internal bookkeeping for stats / insights -------------------------
        self._iter_start_time: float | None = None
        self._loss_history: list[float] = []
        self._step_size_history: list[float] = []
        self._stall_counter: int = 0
        self._prev_loss: float | None = None
        self._mem_tracking_active: bool = False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def any_enabled(self) -> bool:
        return self.print_raw_values or self.show_stats or self.generate_insights or self.emit_warnings

    def summary(self) -> str:
        """Return a human-readable summary of what is turned on."""
        flags = {
            "print_raw_values": self.print_raw_values,
            "show_stats": self.show_stats,
            "generate_insights": self.generate_insights,
            "emit_warnings": self.emit_warnings,
        }
        sub_toggles = {
            "log_loss_components": self.log_loss_components,
            "log_per_component_grads": self.log_per_component_grads,
            "log_param_stats": self.log_param_stats,
            "log_optimizer_state": self.log_optimizer_state,
            "log_cuda_memory": self.log_cuda_memory,
            "log_alm_state": self.log_alm_state,
            "log_run_info": self.log_run_info,
        }
        lines = [f"\n{_SEPARATOR}", "[DebugManager] Active toggles:", _SEPARATOR]
        for name, on in flags.items():
            lines.append(f"  {name:<25s} = {'ON' if on else 'OFF'}")
        lines.append(f"  {'':─<40s}")
        lines.append("  Sub-toggles (under print_raw_values):")
        for name, on in sub_toggles.items():
            lines.append(f"    {name:<25s} = {'ON' if on else 'OFF'}")
        # log_loss_diagnostics shown separately as it can be nested
        if isinstance(self.log_loss_diagnostics, dict):
            lines.append(f"    {'log_loss_diagnostics':<25s} = CUSTOM ({len(self.log_loss_diagnostics)} losses configured)")
            for loss_name, loss_cfg in self.log_loss_diagnostics.items():
                if isinstance(loss_cfg, dict):
                    n_on = sum(1 for v in loss_cfg.values() if v)
                    n_tot = len(loss_cfg)
                    lines.append(f"      {loss_name:<23s}: {n_on}/{n_tot} fields ON")
                else:
                    lines.append(f"      {loss_name:<23s}: {'ON' if loss_cfg else 'OFF'}")
        else:
            lines.append(f"    {'log_loss_diagnostics':<25s} = {'ON' if self.log_loss_diagnostics else 'OFF'}")
        lines.append(_SEPARATOR)
        return "\n".join(lines)

    # ==================================================================
    # 0. RUN CONFIG  — one-time info dump at start of training
    # ==================================================================
    def log_run_config(  # noqa: PLR0912
        self,
        *,
        model: Any = None,
        training_config: dict | None = None,
        general_config: dict | None = None,  # noqa: ARG002
        device: str | None = None,
        num_epochs: int | None = None,
        learning_rate: float | None = None,
        optimizer_type: str | None = None,
        loss_config: dict | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Print a one-time summary of the run configuration at training start."""
        if not self.print_raw_values or not self.log_run_info:
            return

        print(f"\n{'═' * 72}")
        print("[DEBUG RUN CONFIG] Training session starting")
        print(f"{'═' * 72}")

        if device is not None:
            print(f"  Device           : {device}")
            if "cuda" in str(device) and torch.cuda.is_available():
                print(f"  GPU              : {torch.cuda.get_device_name()}")
                print(f"  GPU Memory       : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        if model is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total_params - trainable
            print(f"  Model class      : {type(model).__name__}")
            print(f"  Total params     : {total_params:,}")
            print(f"  Trainable params : {trainable:,}")
            print(f"  Frozen params    : {frozen:,}")
            if hasattr(model, "get_model_info_dict"):
                info = model.get_model_info_dict()
                for k, v in info.items():
                    print(f"  {k:<18s}: {v}")

        if learning_rate is not None:
            print(f"  Learning rate    : {learning_rate:.6e}")
        if optimizer_type is not None:
            print(f"  Optimizer        : {optimizer_type}")
        if num_epochs is not None:
            print(f"  Num epochs       : {num_epochs}")

        if training_config is not None:
            tp = training_config.get("training_params", training_config)
            print(f"  {_SEP_THIN}")
            print("  Training config:")
            for k, v in tp.items():
                print(f"    {k:<30s}: {v}")

        if loss_config is not None:
            print(f"  {_SEP_THIN}")
            print("  Loss config:")
            for k, v in loss_config.items():
                print(f"    {k:<30s}: {v}")

        if extra:
            print(f"  {_SEP_THIN}")
            for k, v in extra.items():
                print(f"  {k:<18s}: {_fmt(v)}")

        print(f"{'═' * 72}\n")

    # ==================================================================
    # 1. VALUES  — print critical state variables / inputs / outputs
    # ==================================================================
    def log_values(  # noqa: PLR0912, PLR0915
        self,
        iteration: int,
        *,
        inputs: Any = None,
        outputs: Any = None,
        loss: float | None = None,
        loss_components: dict[str, float] | None = None,
        per_component_grad_norms: dict[str, float] | None = None,
        grad_norm: float | None = None,
        lr: float | None = None,
        param_stats: dict[str, dict[str, float]] | None = None,
        optimizer_state: dict[str, Any] | None = None,
        cuda_memory: dict[str, float] | None = None,
        alm_state: dict[str, Any] | None = None,
        loss_weights: dict[str, float] | None = None,
        loss_diagnostics: dict[str, dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Print a comprehensive snapshot of the current iteration state."""
        if not self.print_raw_values:
            return
        print(f"\n{_SEPARATOR}")
        print(f"[DEBUG VALUES] Iteration {iteration}")
        print(_SEPARATOR)

        if inputs is not None:
            print(f"  Inputs       : {_fmt(inputs)}")
        if outputs is not None:
            print(f"  Outputs      : {_fmt(outputs)}")

        # ── Total loss ───────────────────────────────────────────
        if loss is not None:
            print(f"  Total Loss   : {loss:.6e}")

        # ── Per-component losses ─────────────────────────────────
        if loss_components and self.log_loss_components:
            print(f"  {_SEP_THIN}")
            print("  Loss components:")
            for name, val in loss_components.items():
                print(f"    {name:<30s}: {val:.6e}")

        # ── Loss weights from config ─────────────────────────────
        if loss_weights and self.log_loss_components:
            print("  Loss weights:")
            for name, w in loss_weights.items():
                print(f"    {name:<30s}: {w:.6e}")

        # ── Gradient info ────────────────────────────────────────
        if grad_norm is not None:
            print(f"  {_SEP_THIN}")
            print(f"  Grad norm    : {grad_norm:.6e}")

        if per_component_grad_norms and self.log_per_component_grads:
            print("  Per-component gradient norms:")
            for name, gn in per_component_grad_norms.items():
                print(f"    {name:<30s}: {gn:.6e}")

        # ── Learning rate ────────────────────────────────────────
        if lr is not None:
            print(f"  LR           : {lr:.6e}")

        # ── Parameter stats ──────────────────────────────────────
        if param_stats and self.log_param_stats:
            print(f"  {_SEP_THIN}")
            print("  Parameter stats:")
            for layer_name, stats in param_stats.items():
                parts = [f"{k}={v:.4e}" for k, v in stats.items()]
                print(f"    {layer_name:<35s}: {' | '.join(parts)}")

        # ── Optimizer state ──────────────────────────────────────
        if optimizer_state and self.log_optimizer_state:
            print(f"  {_SEP_THIN}")
            print("  Optimizer state:")
            for k, v in optimizer_state.items():
                print(f"    {k:<30s}: {_fmt(v)}")

        # ── CUDA memory ─────────────────────────────────────────
        if cuda_memory and self.log_cuda_memory:
            print(f"  {_SEP_THIN}")
            print("  CUDA memory:")
            for k, v in cuda_memory.items():
                print(f"    {k:<30s}: {v:.1f} MB")

        # ── ALM state ────────────────────────────────────────────
        if alm_state and self.log_alm_state:
            print(f"  {_SEP_THIN}")
            print("  ALM multiplier state:")
            for k, v in alm_state.items():
                print(f"    {k:<30s}: {_fmt(v)}")

        # ── Per-loss diagnostics ─────────────────────────────────
        if loss_diagnostics and self.log_loss_diagnostics is not False:
            printed_any = False
            for loss_name, diag in loss_diagnostics.items():
                loss_cfg = _get_loss_diag_cfg(self.log_loss_diagnostics, loss_name)
                filtered = _filter_diag_fields(diag, loss_cfg)
                if not filtered:
                    continue
                if not printed_any:
                    print(f"  {_SEP_THIN}")
                    print("  Per-loss diagnostics:")
                    printed_any = True
                print(f"    ┌─ {loss_name}")
                for k, v in filtered.items():
                    if isinstance(v, float):
                        print(f"    │  {k:<30s}: {v:.6e}")
                    else:
                        print(f"    │  {k:<30s}: {v}")
                print(f"    └{'─' * 36}")

        # ── Extras (catch-all) ───────────────────────────────────
        if extra:
            print(f"  {_SEP_THIN}")
            for k, v in extra.items():
                print(f"  {k:<14s}: {_fmt(v)}")

        print(_SEPARATOR)

    # ==================================================================
    # 2. STATS  — performance metrics
    # ==================================================================
    def begin_iteration(self, _iteration: int) -> None:
        """Call at the **start** of each optimisation iteration."""
        if not self.show_stats:
            return
        self._iter_start_time = time.perf_counter()
        if not self._mem_tracking_active:
            tracemalloc.start()
            self._mem_tracking_active = True

    def end_iteration(
        self,
        iteration: int,
        *,
        loss: float | None = None,
        grad_norm: float | None = None,
        lr: float | None = None,
    ) -> dict[str, float]:
        """Call at the **end** of each iteration. Returns the collected stats dict."""
        stats: dict[str, float] = {}
        if not self.show_stats:
            return stats

        # Execution time
        if self._iter_start_time is not None:
            elapsed = time.perf_counter() - self._iter_start_time
            stats["iter_time_s"] = elapsed
            self._iter_start_time = None

        # Memory (current, peak) via tracemalloc
        if self._mem_tracking_active:
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            stats["mem_current_MB"] = current_mem / (1024 * 1024)
            stats["mem_peak_MB"] = peak_mem / (1024 * 1024)

        # Convergence rate / objective delta
        if loss is not None:
            stats["loss"] = loss
            if self._prev_loss is not None:
                delta = loss - self._prev_loss
                stats["loss_delta"] = delta
                if abs(self._prev_loss) > 1e-30:
                    stats["convergence_rate"] = abs(delta / self._prev_loss)
            self._prev_loss = loss
            self._loss_history.append(loss)

        if grad_norm is not None:
            stats["grad_norm"] = grad_norm
        if lr is not None:
            stats["learning_rate"] = lr

        # Print at configured interval
        if iteration % self.stats_log_interval == 0:
            _print_stats(iteration, stats)

        return stats

    # ==================================================================
    # 3. WARNINGS  — flag anomalous behaviour
    # ==================================================================
    def check_warnings(  # noqa: PLR0912, PLR0915
        self,
        iteration: int,
        *,
        loss: float | None = None,
        grad_norm: float | None = None,
        params: Any | None = None,
        param_bounds: tuple[float, float] | None = None,
        loss_components: dict[str, float] | None = None,
        loss_diagnostics: dict[str, dict[str, Any]] | None = None,
    ) -> list[str]:
        """Run anomaly checks and return (+ print) a list of warnings."""
        warnings_list: list[str] = []
        if not self.emit_warnings:
            return warnings_list

        # NaN / Inf in loss
        if loss is not None:
            if math.isnan(loss):
                warnings_list.append(f"Iter {iteration}: loss is NaN!")
            elif math.isinf(loss):
                warnings_list.append(f"Iter {iteration}: loss is Inf!")
            elif abs(loss) > self.nan_threshold:
                warnings_list.append(
                    f"Iter {iteration}: loss magnitude ({loss:.3e}) exceeds threshold ({self.nan_threshold:.1e})"
                )

        # Per-component NaN/Inf check
        if loss_components is not None:
            for name, val in loss_components.items():
                if math.isnan(val):
                    warnings_list.append(f"Iter {iteration}: loss component '{name}' is NaN!")
                elif math.isinf(val):
                    warnings_list.append(f"Iter {iteration}: loss component '{name}' is Inf!")

        # Zero / vanishing gradient
        if grad_norm is not None and grad_norm < self.grad_zero_atol:
            warnings_list.append(f"Iter {iteration}: gradient norm ({grad_norm:.3e}) is effectively zero")

        # Per-loss diagnostic warnings
        if loss_diagnostics is not None:
            for ln, diag in loss_diagnostics.items():
                # Empty mesh / pointgrid
                if diag.get("mesh_empty"):
                    warnings_list.append(f"Iter {iteration}: '{ln}' mesh extraction returned EMPTY mesh!")
                if diag.get("n_points", 1) == 0:
                    warnings_list.append(f"Iter {iteration}: '{ln}' has 0 sample points (division-by-zero in .mean())!")
                if diag.get("estimator_returned_none"):
                    warnings_list.append(f"Iter {iteration}: '{ln}' crude estimator returned None!")

                # BB-specific: all points violating
                vf = diag.get("violation_frac")
                if vf is not None and vf > self.bb_violation_frac:
                    warnings_list.append(f"Iter {iteration}: '{ln}' {vf * 100:.1f}% of points violating bounding box!")

                # Eikonal-specific: gradient norms far from 1
                gnm = diag.get("grad_norm_mean")
                if gnm is not None and abs(gnm - 1.0) > self.eikonal_grad_norm_tol:
                    warnings_list.append(f"Iter {iteration}: '{ln}' eikonal grad norm mean={gnm:.3f} (should be ~1.0)")

                # Smoothness: extreme curvature
                lap_max = diag.get("laplacian_abs_max")
                if lap_max is not None and lap_max > self.smoothness_laplacian_max:
                    warnings_list.append(f"Iter {iteration}: '{ln}' extreme curvature: max |laplacian|={lap_max:.3e}")

                # GAOT-specific: NaN/Inf in force/drag/pressure
                warnings_list.extend(
                    f"Iter {iteration}: '{ln}' produced NaN output!"
                    for flag_key in ("force_is_nan", "drag_is_nan", "cd_proxy_is_nan")
                    if diag.get(flag_key)
                )
                warnings_list.extend(
                    f"Iter {iteration}: '{ln}' produced Inf output!"
                    for flag_key in ("force_is_inf", "drag_is_inf", "cd_proxy_is_inf")
                    if diag.get(flag_key)
                )

                pf_nan = diag.get("pressure_field_nan_count", 0)
                if pf_nan > 0:
                    warnings_list.append(f"Iter {iteration}: '{ln}' pressure field has {pf_nan} NaN values!")
                pf_inf = diag.get("pressure_field_inf_count", 0)
                if pf_inf > 0:
                    warnings_list.append(f"Iter {iteration}: '{ln}' pressure field has {pf_inf} Inf values!")

                # Surface area: degenerate triangles
                degen = diag.get("degenerate_triangles", 0)
                if degen > 0:
                    warnings_list.append(f"Iter {iteration}: '{ln}' has {degen} degenerate triangles (area < 1e-10)!")

                # Cruising speed solver
                if ln == "cruising_speed":
                    if diag.get("speed_is_nan"):
                        warnings_list.append(f"Iter {iteration}: cruising speed solver returned NaN!")
                    if diag.get("speed_is_inf"):
                        warnings_list.append(f"Iter {iteration}: cruising speed solver returned Inf!")
                    spd = diag.get("speed (m/s)")
                    if spd is not None and not diag.get("speed_is_nan") and spd < self.cruising_speed_min_ms:
                        warnings_list.append(f"Iter {iteration}: cruising speed very low: {spd:.2f} m/s ({spd * 3.6:.1f} km/h)")
                    drag = diag.get("drag_force (N)")
                    if drag is not None and drag <= 0:
                        warnings_list.append(
                            f"Iter {iteration}: cruising speed solver received non-positive drag force: {drag:.3e} N"
                        )
                    solar = diag.get("solar_power (W)")
                    if solar is not None and solar <= 0:
                        warnings_list.append(
                            f"Iter {iteration}: cruising speed solver received non-positive solar power: {solar:.3e} W"
                        )

        # Boundary-limit check for parameters
        if params is not None and param_bounds is not None:
            lo, hi = param_bounds
            _check_boundary(params, lo, hi, iteration, warnings_list)

        # Slow convergence (stall detection)
        if loss is not None and self._prev_loss is not None:
            rel = abs(loss - self._prev_loss) / max(abs(self._prev_loss), 1e-30)
            if rel < 1e-8:
                self._stall_counter += 1
            else:
                self._stall_counter = 0
            if self._stall_counter >= self.convergence_patience:
                warnings_list.append(f"Iter {iteration}: convergence stalled for {self._stall_counter} consecutive iterations")

        for w in warnings_list:
            print(f"  [WARNING] {w}")
        return warnings_list

    # ==================================================================
    # 4. INSIGHTS  — analytical context
    # ==================================================================
    def check_insights(
        self,
        iteration: int,
        *,
        loss: float | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
    ) -> list[str]:
        """Generate and print analytical insights for the current state."""
        insights: list[str] = []
        if not self.generate_insights:
            return insights

        # Step-size shrinking analysis
        if lr is not None:
            self._step_size_history.append(lr)
            if len(self._step_size_history) >= 2:
                ratio = self._step_size_history[-1] / max(self._step_size_history[0], 1e-30)
                if ratio < self.insight_step_shrink_ratio:
                    insights.append(
                        f"Iter {iteration}: Step size has shrunk to {ratio:.2e}x of initial — "
                        "potentially stuck in a local minimum."
                    )

        # Loss plateau detection
        if loss is not None and len(self._loss_history) >= self.insight_plateau_window:
            recent = self._loss_history[-self.insight_plateau_window :]
            spread = max(recent) - min(recent)
            if spread < self.insight_plateau_spread_ratio * max(abs(recent[0]), 1e-30):
                insights.append(
                    f"Iter {iteration}: Loss plateau detected over last 10 iterations "
                    f"(spread={spread:.3e}). Consider increasing learning rate or perturbing."
                )

        # Gradient explosion hint
        if grad_norm is not None and grad_norm > self.insight_grad_explosion_threshold:
            insights.append(
                f"Iter {iteration}: Gradient norm is very large ({grad_norm:.3e}). "
                "Consider gradient clipping or reducing learning rate."
            )

        # Gradient vanishing hint
        if grad_norm is not None and 0 < grad_norm < self.insight_grad_vanish_threshold:
            insights.append(
                f"Iter {iteration}: Gradient norm is very small ({grad_norm:.3e}). "
                "Model may have reached a saddle point or flat region."
            )

        for ins in insights:
            print(f"  [INSIGHT] {ins}")
        return insights

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Stop background tracking (tracemalloc)."""
        if self._mem_tracking_active:
            tracemalloc.stop()
            self._mem_tracking_active = False


# ── Public helpers (usable from training loops) ──────────────────────────────


def collect_param_stats(model: Any) -> dict[str, dict[str, float]]:
    """Extract min/max/mean/std for every named parameter of a ``nn.Module``."""
    stats: dict[str, dict[str, float]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        data = param.detach().float()
        stats[name] = {
            "min": data.min().item(),
            "max": data.max().item(),
            "mean": data.mean().item(),
            "std": data.std().item() if data.numel() > 1 else 0.0,
        }
    return stats


def collect_optimizer_state(optimizer: Any) -> dict[str, Any]:
    """Summarise Adam/AdamW state (step count, exp_avg norms, exp_avg_sq norms)."""
    result: dict[str, Any] = {
        "lr": optimizer.param_groups[0]["lr"],
        "betas": optimizer.param_groups[0].get("betas", "N/A"),
        "weight_decay": optimizer.param_groups[0].get("weight_decay", 0.0),
    }
    # Aggregate momentum stats across all param groups
    exp_avg_norms: list[float] = []
    exp_avg_sq_norms: list[float] = []
    steps: list[int] = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p)
            if state is None:
                continue
            if "step" in state:
                steps.append(int(state["step"]))
            if "exp_avg" in state:
                exp_avg_norms.append(state["exp_avg"].norm().item())
            if "exp_avg_sq" in state:
                exp_avg_sq_norms.append(state["exp_avg_sq"].norm().item())

    if steps:
        result["optimizer_step"] = max(steps)
    if exp_avg_norms:
        result["exp_avg_norm_mean"] = sum(exp_avg_norms) / len(exp_avg_norms)
        result["exp_avg_norm_max"] = max(exp_avg_norms)
    if exp_avg_sq_norms:
        result["exp_avg_sq_norm_mean"] = sum(exp_avg_sq_norms) / len(exp_avg_sq_norms)
        result["exp_avg_sq_norm_max"] = max(exp_avg_sq_norms)

    return result


def collect_cuda_memory() -> dict[str, float]:
    """Return current CUDA memory stats in MB.  Returns empty dict on CPU."""
    if not torch.cuda.is_available():
        return {}
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**2,
        "reserved": torch.cuda.memory_reserved() / 1024**2,
        "max_allocated": torch.cuda.max_memory_allocated() / 1024**2,
        "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2,
    }


def collect_alm_state(loss_function: Any) -> dict[str, Any]:
    """Extract ALM multiplier state from a TotalLoss instance."""
    state: dict[str, Any] = {}
    if not getattr(loss_function, "_use_alm", False):
        return state
    state["use_alm"] = True
    if hasattr(loss_function, "lamdas"):
        state["lamdas"] = loss_function.lamdas.detach().cpu().tolist()
    if hasattr(loss_function, "mus"):
        state["mus"] = loss_function.mus.detach().cpu().tolist()
    if hasattr(loss_function, "nus"):
        state["nus"] = loss_function.nus.detach().cpu().tolist()
    if hasattr(loss_function, "alpha"):
        state["alpha"] = loss_function.alpha
    if hasattr(loss_function, "gamma"):
        state["gamma"] = loss_function.gamma
    return state


def collect_loss_components(loss_function: Any) -> dict[str, float]:
    """Extract per-component weighted loss values from TotalLoss._last_weighted_losses."""
    components: dict[str, float] = {}
    wl = getattr(loss_function, "_last_weighted_losses", {})
    for name, tensor in wl.items():
        components[name] = tensor.detach().item()
    return components


def collect_loss_weights(loss_function: Any) -> dict[str, float]:
    """Extract manual loss weights from TotalLoss config."""
    return dict(getattr(loss_function, "_manual_weights", {}))


def collect_loss_diagnostics(loss_function: Any) -> dict[str, dict[str, Any]]:
    """Extract per-component diagnostics from each active loss instance.

    Each loss stores ``_last_diagnostics`` (a dict) after its ``_loss()`` runs.
    This function iterates all active losses in a TotalLoss and collects them.

    Returns a dict of ``{loss_name: diagnostics_dict}`` for every loss that
    has diagnostics available.
    """
    active = getattr(loss_function, "_active_losses", {})
    result: dict[str, dict[str, Any]] = {}
    for name, instance in active.items():
        diag = getattr(instance, "_last_diagnostics", None)
        if diag:
            result[name] = dict(diag)
    # Cruising speed lives on TotalLoss directly (not in _active_losses)
    cs_diag = None
    getter = getattr(loss_function, "get_cruising_speed_diagnostics", None)
    if callable(getter):
        cs_diag = getter()
    if cs_diag:
        result["cruising_speed"] = dict(cs_diag)
    return result


# ── Private helpers ──────────────────────────────────────────────────────────


def _get_loss_diag_cfg(log_cfg: bool | dict, loss_name: str) -> bool | dict:
    """Return the effective config for one loss given the top-level log_loss_diagnostics value.

    ``log_cfg`` is either:
    * ``True``  → show everything
    * ``False`` → show nothing
    * ``dict``  → per-loss entries; if loss not listed, default is show-all (True)

    Returns ``True``, ``False``, or a ``{field: bool}`` dict.
    """
    if isinstance(log_cfg, bool):
        return log_cfg
    # dict mode: look up the loss; default to True (show all) if not listed
    return log_cfg.get(loss_name, True)


def _filter_diag_fields(diag: dict, loss_cfg: bool | dict) -> dict:
    """Return a filtered copy of ``diag`` based on ``loss_cfg``.

    * ``True``            → return all fields
    * ``False``           → return empty dict (caller skips the block)
    * ``{field: bool}``   → keep only fields whose value is True;
                            fields not listed in the config default to True
    """
    if loss_cfg is True:
        return diag
    if loss_cfg is False:
        return {}
    # dict: keep fields that are True in the config; unknown fields default True
    return {k: v for k, v in diag.items() if loss_cfg.get(k, True)}


def _print_stats(iteration: int, stats: dict[str, float]) -> None:
    """Pretty-print a stats dictionary."""
    parts = [f"[DEBUG STATS] Iter {iteration}:"]
    for k, v in stats.items():
        if "time" in k:
            parts.append(f"{k}={v:.4f}s")
        elif "MB" in k:
            parts.append(f"{k}={v:.1f}MB")
        else:
            parts.append(f"{k}={v:.6e}")
    print("  " + "  |  ".join(parts))


def _check_boundary(
    params: Any,
    lo: float,
    hi: float,
    iteration: int,
    warnings_list: list[str],
) -> None:
    """Append warnings for parameters at boundary limits."""
    try:
        params_flat: torch.Tensor | list[Any]
        if isinstance(params, torch.Tensor):
            params_flat = params.detach().cpu().view(-1)
        elif isinstance(params, list):
            params_flat = params
        else:
            return

        for idx, val in enumerate(params_flat):
            v = float(val)
            if v <= lo:
                warnings_list.append(f"Iter {iteration}: param[{idx}] = {v:.4e} hit lower bound {lo}")
            elif v >= hi:
                warnings_list.append(f"Iter {iteration}: param[{idx}] = {v:.4e} hit upper bound {hi}")
    except Exception:
        pass
