"""Comprehensive test suite for the WP4 DebugManager and its integration
into the real training loops (BasicMLP, pretraining, WIRE, VictorModel).

Run with::

    python -m pytest tests/wp4_aeroshell_optimizer/test_debug_manager.py -v

Or standalone::

    python tests/wp4_aeroshell_optimizer/test_debug_manager.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Ensure the project root is importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subprojects.wp4_aeroshell_optimizer.v1.src.utils.debug_manager import (  # noqa: E402
    DebugManager,
    _filter_diag_fields,
    _get_loss_diag_cfg,
    _load_debug_yaml,
    collect_alm_state,
    collect_cuda_memory,
    collect_loss_components,
    collect_loss_diagnostics,
    collect_loss_weights,
    collect_optimizer_state,
    collect_param_stats,
)

# ---------------------------------------------------------------------------
# Fixtures & constants
# ---------------------------------------------------------------------------

FULL_ON_CFG: dict = {
    "GLOBAL_DEBUG": None,
    "print_raw_values": True,
    "show_stats": True,
    "generate_insights": True,
    "emit_warnings": True,
    "stats_log_interval": 1,
    "warning_nan_threshold": 1e10,
    "warning_grad_zero_atol": 1e-12,
    "warning_convergence_patience": 50,
    "insight_step_shrink_ratio": 0.01,
    "log_loss_components": True,
    "log_per_component_grads": True,
    "log_param_stats": True,
    "log_optimizer_state": True,
    "log_cuda_memory": True,
    "log_alm_state": True,
    "log_run_info": True,
    "log_loss_diagnostics": True,
}

FULL_OFF_CFG: dict = {
    "GLOBAL_DEBUG": None,
    "print_raw_values": False,
    "show_stats": False,
    "generate_insights": False,
    "emit_warnings": False,
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Config file tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConfigFile:
    """Validate that configs/debug.yaml exists and is well-formed."""

    def test_yaml_loads_successfully(self) -> None:
        cfg = _load_debug_yaml()
        assert isinstance(cfg, dict)

    def test_has_global_debug_key(self) -> None:
        cfg = _load_debug_yaml()
        assert "GLOBAL_DEBUG" in cfg

    def test_has_all_granular_toggles(self) -> None:
        cfg = _load_debug_yaml()
        for key in ("print_raw_values", "show_stats", "generate_insights", "emit_warnings"):
            assert key in cfg, f"Missing granular toggle: {key}"

    def test_has_tuning_knobs(self) -> None:
        cfg = _load_debug_yaml()
        for key in (
            "stats_log_interval",
            "warning_nan_threshold",
            "warning_grad_zero_atol",
            "warning_convergence_patience",
            "insight_step_shrink_ratio",
        ):
            assert key in cfg, f"Missing tuning knob: {key}"

    def test_has_sub_toggles(self) -> None:
        cfg = _load_debug_yaml()
        for key in (
            "log_loss_components",
            "log_per_component_grads",
            "log_param_stats",
            "log_optimizer_state",
            "log_cuda_memory",
            "log_alm_state",
            "log_run_info",
            "log_loss_diagnostics",
        ):
            assert key in cfg, f"Missing sub-toggle: {key}"

    def test_log_loss_diagnostics_is_dict_in_yaml(self) -> None:
        cfg = _load_debug_yaml()
        # The YAML now uses nested per-field config, not a plain bool
        assert isinstance(cfg["log_loss_diagnostics"], dict)

    def test_global_debug_has_valid_value(self) -> None:
        cfg = _load_debug_yaml()
        assert cfg["GLOBAL_DEBUG"] in (True, False, None)


# ═══════════════════════════════════════════════════════════════════════════
# 2. GLOBAL_DEBUG override logic
# ═══════════════════════════════════════════════════════════════════════════


class TestGlobalOverride:
    def test_global_true_forces_all_on(self) -> None:
        dbg = DebugManager(config={"GLOBAL_DEBUG": True})
        assert dbg.print_raw_values is True
        assert dbg.show_stats is True
        assert dbg.generate_insights is True
        assert dbg.emit_warnings is True

    def test_global_false_forces_all_off(self) -> None:
        dbg = DebugManager(config={"GLOBAL_DEBUG": False})
        assert dbg.print_raw_values is False
        assert dbg.show_stats is False
        assert dbg.generate_insights is False
        assert dbg.emit_warnings is False

    def test_global_none_respects_granular(self) -> None:
        cfg = {
            "GLOBAL_DEBUG": None,
            "print_raw_values": True,
            "show_stats": False,
            "generate_insights": True,
            "emit_warnings": False,
        }
        dbg = DebugManager(config=cfg)
        assert dbg.print_raw_values is True
        assert dbg.show_stats is False
        assert dbg.generate_insights is True
        assert dbg.emit_warnings is False

    def test_constructor_global_override_param_true(self) -> None:
        """global_override=True in constructor trumps config."""
        cfg = {"GLOBAL_DEBUG": False, "print_raw_values": False}
        dbg = DebugManager(config=cfg, global_override=True)
        assert dbg.print_raw_values is True
        assert dbg.show_stats is True

    def test_constructor_global_override_param_false(self) -> None:
        """global_override=False in constructor trumps config."""
        cfg = {"GLOBAL_DEBUG": True}
        dbg = DebugManager(config=cfg, global_override=False)
        assert dbg.print_raw_values is False
        assert dbg.emit_warnings is False

    def test_any_enabled_true_when_at_least_one_on(self) -> None:
        cfg = {"GLOBAL_DEBUG": None, "print_raw_values": True}
        dbg = DebugManager(config=cfg)
        assert dbg.any_enabled is True

    def test_any_enabled_false_when_none_on(self) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        assert dbg.any_enabled is False

    def test_global_true_overrides_false_granular(self) -> None:
        """Even if granular toggles are False, GLOBAL_DEBUG=True wins."""
        cfg = {
            "GLOBAL_DEBUG": True,
            "print_raw_values": False,
            "show_stats": False,
            "generate_insights": False,
            "emit_warnings": False,
        }
        dbg = DebugManager(config=cfg)
        assert dbg.print_raw_values is True
        assert dbg.show_stats is True

    def test_global_false_overrides_true_granular(self) -> None:
        """Even if granular toggles are True, GLOBAL_DEBUG=False wins."""
        cfg = {
            "GLOBAL_DEBUG": False,
            "print_raw_values": True,
            "show_stats": True,
            "generate_insights": True,
            "emit_warnings": True,
        }
        dbg = DebugManager(config=cfg)
        assert dbg.print_raw_values is False
        assert dbg.show_stats is False


# ═══════════════════════════════════════════════════════════════════════════
# 3. Values hook
# ═══════════════════════════════════════════════════════════════════════════


class TestLogValues:
    def test_prints_when_enabled(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_values(0, inputs=[1, 2, 3], outputs=[4, 5, 6], loss=0.123)
        captured = capsys.readouterr().out
        assert "[DEBUG VALUES]" in captured
        assert "Iteration 0" in captured
        assert "1.230000e-01" in captured

    def test_silent_when_disabled(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        dbg.log_values(0, inputs=[1], outputs=[2], loss=0.5)
        assert capsys.readouterr().out == ""

    def test_extra_fields_printed(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_values(5, extra={"my_metric": 42.0})
        captured = capsys.readouterr().out
        assert "my_metric" in captured

    def test_tensor_input_formatted(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        t = torch.randn(4, 3)
        dbg.log_values(0, inputs=t)
        captured = capsys.readouterr().out
        assert "Tensor" in captured
        assert "shape=" in captured

    def test_no_loss_still_prints_header(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_values(10)  # no inputs/outputs/loss
        captured = capsys.readouterr().out
        assert "Iteration 10" in captured


# ═══════════════════════════════════════════════════════════════════════════
# 4. Stats hook
# ═══════════════════════════════════════════════════════════════════════════


class TestStats:
    def test_end_iteration_returns_stats(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        stats = dbg.end_iteration(0, loss=1.0)
        assert "iter_time_s" in stats
        assert "loss" in stats
        assert stats["loss"] == 1.0
        dbg.stop()

    def test_convergence_rate_computed(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        dbg.end_iteration(0, loss=10.0)
        dbg.begin_iteration(1)
        stats = dbg.end_iteration(1, loss=9.0)
        assert "loss_delta" in stats
        assert "convergence_rate" in stats
        assert stats["loss_delta"] == pytest.approx(-1.0)
        assert stats["convergence_rate"] == pytest.approx(0.1)
        dbg.stop()

    def test_noop_when_disabled(self) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        dbg.begin_iteration(0)
        stats = dbg.end_iteration(0, loss=1.0)
        assert stats == {}

    def test_stats_log_interval(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "stats_log_interval": 5}
        dbg = DebugManager(config=cfg)
        for i in range(10):
            dbg.begin_iteration(i)
            dbg.end_iteration(i, loss=float(i))
        captured = capsys.readouterr().out
        assert "[DEBUG STATS] Iter 0:" in captured
        assert "[DEBUG STATS] Iter 5:" in captured
        assert "[DEBUG STATS] Iter 1:" not in captured
        dbg.stop()

    def test_memory_tracked(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        _ = [0] * 10000
        stats = dbg.end_iteration(0, loss=1.0)
        assert "mem_current_MB" in stats
        assert "mem_peak_MB" in stats
        assert stats["mem_current_MB"] >= 0
        dbg.stop()

    def test_grad_norm_stored(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        stats = dbg.end_iteration(0, grad_norm=1.234)
        assert stats["grad_norm"] == pytest.approx(1.234)
        dbg.stop()

    def test_learning_rate_stored(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        stats = dbg.end_iteration(0, lr=0.001)
        assert stats["learning_rate"] == pytest.approx(0.001)
        dbg.stop()


# ═══════════════════════════════════════════════════════════════════════════
# 5. Warnings hook
# ═══════════════════════════════════════════════════════════════════════════


class TestWarnings:
    def test_nan_loss_warning(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss=float("nan"))
        assert any("NaN" in w for w in ws)
        assert "[WARNING]" in capsys.readouterr().out

    def test_inf_loss_warning(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss=float("inf"))
        assert any("Inf" in w for w in ws)

    def test_large_loss_warning(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss=1e15)
        assert any("exceeds threshold" in w for w in ws)

    def test_zero_gradient_warning(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, grad_norm=0.0)
        assert any("zero" in w.lower() for w in ws)

    def test_no_warning_for_normal_values(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss=0.5, grad_norm=1.0)
        assert ws == []

    def test_silent_when_disabled(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        ws = dbg.check_warnings(0, loss=float("nan"))
        assert ws == []
        assert capsys.readouterr().out == ""

    def test_boundary_warning_with_list(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, params=[0.0, 5.0, 10.0], param_bounds=(0.0, 10.0))
        assert any("lower bound" in w for w in ws)
        assert any("upper bound" in w for w in ws)

    def test_boundary_warning_with_tensor(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        params = torch.tensor([0.0, 5.0, 10.0])
        ws = dbg.check_warnings(0, params=params, param_bounds=(0.0, 10.0))
        assert any("lower bound" in w for w in ws)
        assert any("upper bound" in w for w in ws)

    def test_convergence_stall_warning(self) -> None:
        cfg = {**FULL_ON_CFG, "warning_convergence_patience": 3}
        dbg = DebugManager(config=cfg)
        for i in range(5):
            dbg._prev_loss = 1.0
            dbg.check_warnings(i, loss=1.0)
        ws = dbg.check_warnings(5, loss=1.0)
        assert any("stalled" in w for w in ws)

    def test_custom_nan_threshold(self) -> None:
        cfg = {**FULL_ON_CFG, "warning_nan_threshold": 100.0}
        dbg = DebugManager(config=cfg)
        ws1 = dbg.check_warnings(0, loss=50.0)
        assert ws1 == []
        ws2 = dbg.check_warnings(1, loss=200.0)
        assert any("exceeds threshold" in w for w in ws2)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Insights hook
# ═══════════════════════════════════════════════════════════════════════════


class TestInsights:
    def test_step_shrink_insight(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.check_insights(0, lr=1.0)
        ins = dbg.check_insights(1, lr=0.001)
        assert any("shrunk" in i.lower() or "stuck" in i.lower() for i in ins)
        assert "[INSIGHT]" in capsys.readouterr().out

    def test_loss_plateau_insight(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg._loss_history = [1.0] * 15
        ins = dbg.check_insights(20, loss=1.0)
        assert any("plateau" in i.lower() for i in ins)

    def test_gradient_explosion_insight(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ins = dbg.check_insights(0, grad_norm=1e8)
        assert any("large" in i.lower() or "clipping" in i.lower() for i in ins)

    def test_gradient_vanishing_insight(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ins = dbg.check_insights(0, grad_norm=1e-10)
        assert any("small" in i.lower() or "saddle" in i.lower() for i in ins)

    def test_silent_when_disabled(self) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        ins = dbg.check_insights(0, lr=1e-20, loss=1.0, grad_norm=1e12)
        assert ins == []

    def test_no_insight_with_healthy_values(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ins = dbg.check_insights(0, loss=0.5, lr=0.001, grad_norm=1.0)
        assert ins == []


# ═══════════════════════════════════════════════════════════════════════════
# 7. Summary & cleanup
# ═══════════════════════════════════════════════════════════════════════════


class TestMisc:
    def test_summary_contains_toggle_names(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        s = dbg.summary()
        assert "print_raw_values" in s
        assert "show_stats" in s
        assert "ON" in s

    def test_summary_shows_off_when_disabled(self) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        s = dbg.summary()
        assert "OFF" in s

    def test_stop_is_idempotent(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        dbg.stop()
        dbg.stop()  # should not raise

    def test_defaults_for_missing_keys(self) -> None:
        dbg = DebugManager(config={})
        assert dbg.print_raw_values is False
        assert dbg.show_stats is False

    def test_tuning_knobs_loaded(self) -> None:
        cfg = {
            **FULL_ON_CFG,
            "stats_log_interval": 10,
            "warning_nan_threshold": 42.0,
            "warning_grad_zero_atol": 1e-6,
            "warning_convergence_patience": 99,
            "insight_step_shrink_ratio": 0.05,
        }
        dbg = DebugManager(config=cfg)
        assert dbg.stats_log_interval == 10
        assert dbg.nan_threshold == pytest.approx(42.0)
        assert dbg.grad_zero_atol == pytest.approx(1e-6)
        assert dbg.convergence_patience == 99
        assert dbg.insight_step_shrink_ratio == pytest.approx(0.05)


# ═══════════════════════════════════════════════════════════════════════════
# 8a. Sub-toggles
# ═══════════════════════════════════════════════════════════════════════════


class TestSubToggles:
    """Verify that individual sub-toggles suppress their specific sections."""

    def test_sub_toggles_default_to_true(self) -> None:
        dbg = DebugManager(config={"GLOBAL_DEBUG": True})
        assert dbg.log_loss_components is True
        assert dbg.log_per_component_grads is True
        assert dbg.log_param_stats is True
        assert dbg.log_optimizer_state is True
        assert dbg.log_cuda_memory is True
        assert dbg.log_alm_state is True
        assert dbg.log_run_info is True

    def test_sub_toggles_off_suppresses_loss_components(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_loss_components": False}
        dbg = DebugManager(config=cfg)
        dbg.log_values(0, loss=1.0, loss_components={"cd": 0.5, "cl": 0.3})
        captured = capsys.readouterr().out
        assert "cd" not in captured

    def test_sub_toggles_on_shows_loss_components(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_values(0, loss=1.0, loss_components={"cd": 0.5, "cl": 0.3})
        captured = capsys.readouterr().out
        assert "cd" in captured
        assert "cl" in captured

    def test_sub_toggle_off_suppresses_param_stats(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_param_stats": False}
        dbg = DebugManager(config=cfg)
        dbg.log_values(0, param_stats={"layer.weight": {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.5}})
        captured = capsys.readouterr().out
        assert "layer.weight" not in captured

    def test_sub_toggle_off_suppresses_optimizer_state(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_optimizer_state": False}
        dbg = DebugManager(config=cfg)
        dbg.log_values(0, optimizer_state={"lr": 0.001, "betas": (0.9, 0.999)})
        captured = capsys.readouterr().out
        assert "betas" not in captured

    def test_sub_toggle_off_suppresses_grad_norms(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_per_component_grads": False}
        dbg = DebugManager(config=cfg)
        dbg.log_values(0, per_component_grad_norms={"cd_loss": 0.5})
        captured = capsys.readouterr().out
        assert "cd_loss" not in captured

    def test_summary_shows_sub_toggles(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        s = dbg.summary()
        assert "log_loss_components" in s
        assert "log_param_stats" in s


# ═══════════════════════════════════════════════════════════════════════════
# 8b. log_run_config
# ═══════════════════════════════════════════════════════════════════════════


class TestLogRunConfig:
    """Verify the one-time startup config dump."""

    def test_prints_when_enabled(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        model = torch.nn.Linear(3, 1)
        dbg.log_run_config(model=model, device="cpu", learning_rate=1e-3, num_epochs=100)
        captured = capsys.readouterr().out
        assert "[DEBUG RUN CONFIG]" in captured
        assert "cpu" in captured
        assert "1.000000e-03" in captured

    def test_silent_when_values_disabled(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        dbg.log_run_config(model=torch.nn.Linear(3, 1), device="cpu")
        assert capsys.readouterr().out == ""

    def test_silent_when_sub_toggle_off(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_run_info": False}
        dbg = DebugManager(config=cfg)
        dbg.log_run_config(model=torch.nn.Linear(3, 1), device="cpu")
        assert capsys.readouterr().out == ""

    def test_prints_model_param_count(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        model = torch.nn.Linear(10, 5, bias=True)
        dbg.log_run_config(model=model)
        captured = capsys.readouterr().out
        assert "Total params" in captured
        assert "55" in captured  # 10*5 + 5 = 55

    def test_prints_extra_fields(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_run_config(extra={"shape": "sphere", "batch_size": 64})
        captured = capsys.readouterr().out
        assert "sphere" in captured
        assert "64" in captured

    def test_cuda_uses_total_memory_attribute(self, capsys: pytest.CaptureFixture) -> None:
        """Ensure get_device_properties uses .total_memory (not .total_mem)."""
        dbg = DebugManager(config=FULL_ON_CFG)
        fake_props = MagicMock()
        fake_props.total_memory = 8 * 1024**3  # 8 GB
        # total_mem must NOT be accessed; only total_memory
        del fake_props.total_mem
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_name", return_value="FakeGPU"),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            dbg.log_run_config(device="cuda:0")
        captured = capsys.readouterr().out
        assert "8.0 GB" in captured


# ═══════════════════════════════════════════════════════════════════════════
# 8c. Collect helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestCollectHelpers:
    """Test the public collect_* helper functions."""

    def test_collect_param_stats(self) -> None:
        model = torch.nn.Linear(4, 2)
        stats = collect_param_stats(model)
        assert "weight" in next(iter(stats.keys()))
        for v in stats.values():
            assert "min" in v and "max" in v and "mean" in v and "std" in v

    def test_collect_optimizer_state(self) -> None:
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        # Do a dummy step to populate optimizer state
        loss = model(torch.randn(1, 4)).sum()
        loss.backward()
        opt.step()
        state = collect_optimizer_state(opt)
        assert state["lr"] == pytest.approx(0.01)
        assert "betas" in state
        assert "optimizer_step" in state

    def test_collect_optimizer_state_before_step(self) -> None:
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        state = collect_optimizer_state(opt)
        assert state["lr"] == pytest.approx(0.01)
        # No step yet, so no step count
        assert "optimizer_step" not in state

    def test_collect_cuda_memory_on_cpu(self) -> None:
        mem = collect_cuda_memory()
        if not torch.cuda.is_available():
            assert mem == {}

    def test_collect_cuda_memory_uses_total_memory_attribute(self) -> None:
        """Ensure collect_cuda_memory uses .total_memory (not .total_mem)."""
        fake_props = MagicMock()
        fake_props.total_memory = 16 * 1024**2  # 16 MB in bytes
        del fake_props.total_mem
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
            patch("torch.cuda.memory_allocated", return_value=4 * 1024**2),
            patch("torch.cuda.memory_reserved", return_value=8 * 1024**2),
            patch("torch.cuda.max_memory_allocated", return_value=10 * 1024**2),
        ):
            mem = collect_cuda_memory()
        assert mem["free"] == pytest.approx(12.0)  # 16 - 4 MB
        assert mem["allocated"] == pytest.approx(4.0)
        assert mem["reserved"] == pytest.approx(8.0)
        assert mem["max_allocated"] == pytest.approx(10.0)

    def test_collect_alm_state_no_alm(self) -> None:
        mock_loss = MagicMock()
        mock_loss._use_alm = False
        state = collect_alm_state(mock_loss)
        assert state == {}

    def test_collect_alm_state_with_alm(self) -> None:
        mock_loss = MagicMock()
        mock_loss._use_alm = True
        mock_loss.lamdas = torch.tensor([1.0, 2.0])
        mock_loss.mus = torch.tensor([0.5])
        state = collect_alm_state(mock_loss)
        assert state["use_alm"] is True
        assert state["lamdas"] == [1.0, 2.0]

    def test_collect_loss_components(self) -> None:
        mock_loss = MagicMock()
        mock_loss._last_weighted_losses = {
            "cd": torch.tensor(0.5),
            "cl": torch.tensor(0.3),
        }
        components = collect_loss_components(mock_loss)
        assert components["cd"] == pytest.approx(0.5)
        assert components["cl"] == pytest.approx(0.3)

    def test_collect_loss_weights(self) -> None:
        mock_loss = MagicMock()
        mock_loss._manual_weights = {"cd": 1.0, "cl": 0.5}
        weights = collect_loss_weights(mock_loss)
        assert weights == {"cd": 1.0, "cl": 0.5}

    def test_collect_loss_weights_no_attr(self) -> None:
        mock_loss = MagicMock(spec=[])
        weights = collect_loss_weights(mock_loss)
        assert weights == {}


# ═══════════════════════════════════════════════════════════════════════════
# 8d. Enriched values & warnings
# ═══════════════════════════════════════════════════════════════════════════


class TestEnrichedHooks:
    """Test enriched log_values and check_warnings with new parameters."""

    def test_log_values_with_all_new_fields(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_values(
            0,
            loss=1.0,
            loss_components={"cd": 0.5, "cl": 0.3},
            per_component_grad_norms={"cd": 0.1, "cl": 0.05},
            grad_norm=0.15,
            lr=0.001,
            param_stats={"fc.weight": {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.5}},
            optimizer_state={"lr": 0.001, "betas": (0.9, 0.999)},
            cuda_memory={},
            alm_state={},
            loss_weights={"cd": 1.0, "cl": 0.5},
        )
        captured = capsys.readouterr().out
        assert "cd" in captured
        assert "cl" in captured
        assert "fc.weight" in captured
        assert "betas" in captured

    def test_check_warnings_per_component_nan(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_components={"cd": float("nan"), "cl": 0.3})
        assert any("cd" in w and "NaN" in w for w in ws)

    def test_check_warnings_per_component_inf(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_components={"boundary": float("inf")})
        assert any("boundary" in w and "Inf" in w for w in ws)

    def test_check_warnings_per_component_ok(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_components={"cd": 0.5, "cl": 0.3})
        assert ws == []


# ═══════════════════════════════════════════════════════════════════════════
# 8d2. Loss diagnostics collection & warnings
# ═══════════════════════════════════════════════════════════════════════════


class TestLossDiagnostics:
    """Test collect_loss_diagnostics and loss-specific warnings."""

    def test_collect_loss_diagnostics_empty(self) -> None:
        mock_loss = MagicMock()
        mock_loss._active_losses = {}
        mock_loss.get_cruising_speed_diagnostics.return_value = None
        result = collect_loss_diagnostics(mock_loss)
        assert result == {}

    def test_collect_loss_diagnostics_with_data(self) -> None:
        mock_eik = MagicMock()
        mock_eik._last_diagnostics = {"grad_norm_mean": 0.98, "n_points": 1000}
        mock_bb = MagicMock()
        mock_bb._last_diagnostics = {"violation_frac": 0.05, "n_points": 500}

        mock_loss = MagicMock()
        mock_loss._active_losses = {"eikonal": mock_eik, "bb": mock_bb}
        result = collect_loss_diagnostics(mock_loss)
        assert "eikonal" in result
        assert "bb" in result
        assert result["eikonal"]["grad_norm_mean"] == 0.98
        assert result["bb"]["violation_frac"] == 0.05

    def test_collect_ignores_losses_without_diagnostics(self) -> None:
        mock_null = MagicMock(spec=[])
        mock_loss = MagicMock()
        mock_loss._active_losses = {"null_loss": mock_null}
        mock_loss.get_cruising_speed_diagnostics.return_value = None
        result = collect_loss_diagnostics(mock_loss)
        assert result == {}

    def test_log_values_prints_diagnostics(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.log_values(
            0,
            loss=1.0,
            loss_diagnostics={
                "bb_v4": {"violation_frac": 0.12, "upper_violation_max": 3.5, "n_points": 500},
                "eikonal": {"grad_norm_mean": 0.95, "deviation_from_1_mean": 0.08},
            },
        )
        captured = capsys.readouterr().out
        assert "bb_v4" in captured
        assert "eikonal" in captured
        assert "violation_frac" in captured
        assert "grad_norm_mean" in captured

    def test_log_values_hidden_when_toggle_off(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_loss_diagnostics": False}
        dbg = DebugManager(config=cfg)
        dbg.log_values(0, loss=1.0, loss_diagnostics={"bb": {"n_points": 100}})
        captured = capsys.readouterr().out
        assert "bb" not in captured or "Per-loss diagnostics" not in captured

    def test_per_field_filtering_hides_disabled_field(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {
            **FULL_ON_CFG,
            "log_loss_diagnostics": {
                "my_loss": {"n_points": True, "sdf_mean": False},
            },
        }
        dbg = DebugManager(config=cfg)
        dbg.log_values(
            0,
            loss=1.0,
            loss_diagnostics={
                "my_loss": {"n_points": 100, "sdf_mean": -0.5},
            },
        )
        captured = capsys.readouterr().out
        assert "n_points" in captured
        assert "sdf_mean" not in captured

    def test_per_loss_false_hides_entire_loss(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {
            **FULL_ON_CFG,
            "log_loss_diagnostics": {
                "hidden_loss": False,
                "visible_loss": True,
            },
        }
        dbg = DebugManager(config=cfg)
        dbg.log_values(
            0,
            loss=1.0,
            loss_diagnostics={
                "hidden_loss": {"n_points": 42},
                "visible_loss": {"n_points": 99},
            },
        )
        captured = capsys.readouterr().out
        assert "hidden_loss" not in captured
        assert "visible_loss" in captured

    def test_unlisted_loss_defaults_to_show_all(self, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_loss_diagnostics": {"other_loss": True}}
        dbg = DebugManager(config=cfg)
        dbg.log_values(
            0,
            loss=1.0,
            loss_diagnostics={
                "unlisted_loss": {"some_field": 1.23},
            },
        )
        captured = capsys.readouterr().out
        assert "unlisted_loss" in captured
        assert "some_field" in captured

    # -- Helper unit tests --

    def test_get_loss_diag_cfg_bool_true(self) -> None:
        assert _get_loss_diag_cfg(True, "anything") is True

    def test_get_loss_diag_cfg_bool_false(self) -> None:
        assert _get_loss_diag_cfg(False, "anything") is False

    def test_get_loss_diag_cfg_dict_hit(self) -> None:
        cfg = {"bb_v4": {"n_points": True}}
        assert _get_loss_diag_cfg(cfg, "bb_v4") == {"n_points": True}

    def test_get_loss_diag_cfg_dict_miss_defaults_true(self) -> None:
        assert _get_loss_diag_cfg({"other": True}, "unlisted") is True

    def test_filter_diag_fields_true_returns_all(self) -> None:
        d = {"a": 1, "b": 2}
        assert _filter_diag_fields(d, True) == d

    def test_filter_diag_fields_false_returns_empty(self) -> None:
        assert _filter_diag_fields({"a": 1}, False) == {}

    def test_filter_diag_fields_dict_filters(self) -> None:
        d = {"a": 1, "b": 2, "c": 3}
        result = _filter_diag_fields(d, {"a": True, "b": False})
        assert "a" in result
        assert "b" not in result
        # c not in filter config → defaults True
        assert "c" in result

    def test_warning_empty_mesh(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"gaot": {"mesh_empty": True}})
        assert any("EMPTY mesh" in w for w in ws)

    def test_warning_zero_points(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"bb_v4": {"n_points": 0}})
        assert any("0 sample points" in w for w in ws)

    def test_warning_high_violation_frac(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"bb": {"violation_frac": 0.98}})
        assert any("violating bounding box" in w for w in ws)

    def test_warning_eikonal_grad_norm_off(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"eikonal": {"grad_norm_mean": 2.0}})
        assert any("eikonal grad norm" in w for w in ws)

    def test_warning_extreme_curvature(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"smoothness": {"laplacian_abs_max": 1e6}})
        assert any("extreme curvature" in w for w in ws)

    def test_warning_nan_in_gaot(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"gaot": {"force_is_nan": True}})
        assert any("NaN" in w for w in ws)

    def test_warning_inf_in_drag(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"gaot2": {"drag_is_inf": True}})
        assert any("Inf" in w for w in ws)

    def test_warning_pressure_field_nan(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"gaot2": {"pressure_field_nan_count": 5}})
        assert any("pressure field" in w and "5 NaN" in w for w in ws)

    def test_warning_degenerate_triangles(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"surface_area": {"degenerate_triangles": 10}})
        assert any("degenerate" in w for w in ws)

    def test_warning_none_estimator(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss_diagnostics={"crude": {"estimator_returned_none": True}})
        assert any("returned None" in w for w in ws)

    def test_no_warning_for_healthy_diagnostics(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(
            0,
            loss_diagnostics={
                "bb": {"violation_frac": 0.05, "n_points": 500},
                "eikonal": {"grad_norm_mean": 1.01},
                "smoothness": {"laplacian_abs_max": 0.5},
            },
        )
        assert ws == []

    # -- Cruising speed diagnostics ---

    def test_collect_loss_diagnostics_includes_cruising_speed(self) -> None:
        mock_loss = MagicMock()
        mock_loss._active_losses = {}
        mock_loss.get_cruising_speed_diagnostics.return_value = {"speed (m/s)": 22.5, "speed_is_nan": False}
        result = collect_loss_diagnostics(mock_loss)
        assert "cruising_speed" in result
        assert result["cruising_speed"]["speed (m/s)"] == 22.5

    def test_collect_loss_diagnostics_omits_cruising_speed_when_none(self) -> None:
        mock_loss = MagicMock()
        mock_loss._active_losses = {}
        mock_loss.get_cruising_speed_diagnostics.return_value = None
        result = collect_loss_diagnostics(mock_loss)
        assert "cruising_speed" not in result

    def test_collect_loss_diagnostics_no_getter(self) -> None:
        """TotalLoss instances without the getter (e.g. mocks) don't crash."""
        mock_loss = MagicMock(spec=[])  # no attributes at all
        result = collect_loss_diagnostics(mock_loss)
        assert result == {}

    def test_warning_cruising_speed_nan(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(
            0,
            loss_diagnostics={
                "cruising_speed": {"speed (m/s)": float("nan"), "speed_is_nan": True, "speed_is_inf": False},
            },
        )
        assert any("cruising speed" in w and "NaN" in w for w in ws)

    def test_warning_cruising_speed_inf(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(
            0,
            loss_diagnostics={
                "cruising_speed": {"speed (m/s)": float("inf"), "speed_is_nan": False, "speed_is_inf": True},
            },
        )
        assert any("cruising speed" in w and "Inf" in w for w in ws)

    def test_warning_cruising_speed_very_low(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(
            0,
            loss_diagnostics={
                "cruising_speed": {"speed (m/s)": 1.2, "speed_is_nan": False, "speed_is_inf": False},
            },
        )
        assert any("cruising speed very low" in w for w in ws)

    def test_warning_cruising_speed_non_positive_drag(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(
            0,
            loss_diagnostics={
                "cruising_speed": {"speed (m/s)": 20.0, "speed_is_nan": False, "speed_is_inf": False, "drag_force (N)": -5.0},
            },
        )
        assert any("drag force" in w for w in ws)

    def test_warning_cruising_speed_healthy(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(
            0,
            loss_diagnostics={
                "cruising_speed": {
                    "speed (m/s)": 22.0,
                    "speed_is_nan": False,
                    "speed_is_inf": False,
                    "drag_force (N)": 10.5,
                    "solar_power (W)": 500.0,
                },
            },
        )
        assert ws == []


# ═══════════════════════════════════════════════════════════════════════════
# 8e. Full lifecycle test: all hooks in sequence
# ═══════════════════════════════════════════════════════════════════════════


class TestFullLifecycle:
    """Simulate an entire mini training loop calling every hook in order."""

    def test_all_hooks_in_sequence(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        losses = [10.0, 8.0, 6.5, 5.5, 5.0]
        for i, loss_val in enumerate(losses):
            dbg.begin_iteration(i)
            dbg.log_values(i, loss=loss_val, extra={"epoch": i})
            dbg.check_warnings(i, loss=loss_val, grad_norm=1.0)
            dbg.check_insights(i, loss=loss_val, lr=0.001, grad_norm=1.0)
            stats = dbg.end_iteration(i, loss=loss_val, grad_norm=1.0, lr=0.001)
            assert "iter_time_s" in stats
        dbg.stop()
        out = capsys.readouterr().out
        assert "[DEBUG VALUES]" in out
        assert "[DEBUG STATS]" in out

    def test_lifecycle_all_disabled_produces_no_output(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_OFF_CFG)
        for i in range(5):
            dbg.begin_iteration(i)
            dbg.log_values(i, loss=float(i))
            dbg.check_warnings(i, loss=float(i), grad_norm=0.0)
            dbg.check_insights(i, loss=float(i), lr=0.001, grad_norm=0.0)
            dbg.end_iteration(i, loss=float(i))
        dbg.stop()
        assert capsys.readouterr().out == ""


# ═══════════════════════════════════════════════════════════════════════════
# 9. Integration: real BasicMLP model with DebugManager
# ═══════════════════════════════════════════════════════════════════════════


class TestBasicMLPIntegration:
    """Test that DebugManager integrates correctly with the real BasicMLP model."""

    def _make_model(self) -> torch.nn.Module:
        from subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1 import BasicMLP  # noqa: PLC0415

        return BasicMLP(config={}, overwrite_base_init=False, in_dim=3, hidden=16, layers=2)

    def _make_configs(self, num_epochs: int = 3) -> tuple[dict, dict]:
        general_config = {
            "general_info": {"version": "v1", "verbose": False},
            "storage_paths": {"storage_dir": "/tmp/test_wp4_debug"},
        }
        training_config = {
            "training_params": {
                "learning_rate": 1e-3,
                "num_epochs": num_epochs,
                "mesh_save_interval": 99999,
            }
        }
        return general_config, training_config

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.TotalLoss")
    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.wandb")
    def test_optimise_with_debug_on(self, mock_wandb, MockTotalLoss) -> None:
        """BasicMLP.optimise should run to completion with debug fully enabled."""
        mock_wandb.run = None
        mock_wandb.config = {}

        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(1.0, requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        mock_loss_fn._last_weighted_losses = {"test_loss": mock_loss_tensor}
        mock_loss_fn._gradient_memory_diagnostics = False
        MockTotalLoss.return_value = mock_loss_fn

        model = self._make_model()
        general_config, training_config = self._make_configs(num_epochs=3)
        dbg = DebugManager(config={"GLOBAL_DEBUG": True})

        model.optimise(
            general_config=general_config,
            training_config=training_config,
            debug_manager=dbg,
        )
        assert mock_loss_fn.get_total_loss.call_count == 3

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.TotalLoss")
    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.wandb")
    def test_optimise_with_debug_off(self, mock_wandb, MockTotalLoss, capsys) -> None:
        """BasicMLP.optimise with debug off should produce no debug output."""
        mock_wandb.run = None
        mock_wandb.config = {}

        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(2.0, requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        mock_loss_fn._last_weighted_losses = {"test_loss": mock_loss_tensor}
        mock_loss_fn._gradient_memory_diagnostics = False
        MockTotalLoss.return_value = mock_loss_fn

        model = self._make_model()
        general_config, training_config = self._make_configs(num_epochs=3)
        dbg = DebugManager(config={"GLOBAL_DEBUG": False})

        model.optimise(
            general_config=general_config,
            training_config=training_config,
            debug_manager=dbg,
        )
        captured = capsys.readouterr().out
        assert "[DEBUG VALUES]" not in captured
        assert "[DEBUG STATS]" not in captured
        assert "[WARNING]" not in captured
        assert "[INSIGHT]" not in captured

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.TotalLoss")
    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.wandb")
    def test_optimise_without_debug_manager(self, mock_wandb, MockTotalLoss) -> None:
        """BasicMLP.optimise should work fine when no debug_manager is passed (backward compat)."""
        mock_wandb.run = None
        mock_wandb.config = {}

        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(1.5, requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        mock_loss_fn._last_weighted_losses = {"test_loss": mock_loss_tensor}
        mock_loss_fn._gradient_memory_diagnostics = False
        MockTotalLoss.return_value = mock_loss_fn

        model = self._make_model()
        general_config, training_config = self._make_configs(num_epochs=2)

        model.optimise(
            general_config=general_config,
            training_config=training_config,
        )
        assert mock_loss_fn.get_total_loss.call_count == 2

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.TotalLoss")
    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1.wandb")
    def test_optimise_with_custom_toggles(self, mock_wandb, MockTotalLoss, capsys) -> None:
        """Only warnings and stats enabled, not values or insights."""
        mock_wandb.run = None
        mock_wandb.config = {}

        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(float("nan"), requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        mock_loss_fn._last_weighted_losses = {"nan_loss": mock_loss_tensor}
        mock_loss_fn._gradient_memory_diagnostics = False
        MockTotalLoss.return_value = mock_loss_fn

        model = self._make_model()
        general_config, training_config = self._make_configs(num_epochs=2)
        dbg = DebugManager(
            config={
                "GLOBAL_DEBUG": None,
                "print_raw_values": False,
                "show_stats": True,
                "generate_insights": False,
                "emit_warnings": True,
            }
        )

        model.optimise(
            general_config=general_config,
            training_config=training_config,
            debug_manager=dbg,
        )
        captured = capsys.readouterr().out
        assert "[DEBUG STATS]" in captured
        assert "[WARNING]" in captured
        assert "[DEBUG VALUES]" not in captured
        assert "[INSIGHT]" not in captured


# ═══════════════════════════════════════════════════════════════════════════
# 10. Integration: pretraining loop with DebugManager
# ═══════════════════════════════════════════════════════════════════════════


class TestPretrainingIntegration:
    """Test that the pretraining loop correctly uses the DebugManager."""

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1.load_config")
    def test_pretraining_with_debug_on(self, mock_load_config, capsys) -> None:
        mock_load_config.return_value = {
            "model": "BasicMLP",
            "model_filename": "basic_MLP_v1",
            "pretraining_shape": "sphere",
            "training_loop": {
                "learning_rate": 1e-3,
                "epochs": 3,
                "random_batch_size": 32,
                "print_interval": 1,
            },
        }

        from subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1 import BasicMLP  # noqa: PLC0415

        model = BasicMLP(config={}, overwrite_base_init=False, in_dim=3, hidden=16, layers=2)
        dbg = DebugManager(config={"GLOBAL_DEBUG": True})

        from subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1 import pretraining  # noqa: PLC0415

        result = pretraining(model=model, debug_manager=dbg)
        assert result is not None
        captured = capsys.readouterr().out
        assert "[DEBUG VALUES]" in captured
        assert "[DEBUG STATS]" in captured

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1.load_config")
    def test_pretraining_with_debug_off(self, mock_load_config, capsys) -> None:
        mock_load_config.return_value = {
            "model": "BasicMLP",
            "model_filename": "basic_MLP_v1",
            "pretraining_shape": "sphere",
            "training_loop": {
                "learning_rate": 1e-3,
                "epochs": 3,
                "random_batch_size": 32,
                "print_interval": 1,
            },
        }

        from subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1 import BasicMLP  # noqa: PLC0415

        model = BasicMLP(config={}, overwrite_base_init=False, in_dim=3, hidden=16, layers=2)
        dbg = DebugManager(config={"GLOBAL_DEBUG": False})

        from subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1 import pretraining  # noqa: PLC0415

        result = pretraining(model=model, debug_manager=dbg)
        assert result is not None
        captured = capsys.readouterr().out
        assert "[DEBUG VALUES]" not in captured
        assert "[DEBUG STATS]" not in captured
        assert "[WARNING]" not in captured

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1.load_config")
    def test_pretraining_without_debug_manager(self, mock_load_config) -> None:
        """Pretraining should work with no debug_manager (backward compat)."""
        mock_load_config.return_value = {
            "model": "BasicMLP",
            "model_filename": "basic_MLP_v1",
            "pretraining_shape": "sphere",
            "training_loop": {
                "learning_rate": 1e-3,
                "epochs": 2,
                "random_batch_size": 32,
                "print_interval": 1,
            },
        }

        from subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1 import BasicMLP  # noqa: PLC0415

        model = BasicMLP(config={}, overwrite_base_init=False, in_dim=3, hidden=16, layers=2)

        from subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1 import pretraining  # noqa: PLC0415

        result = pretraining(model=model)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════
# 11. Integration: VictorModel with DebugManager
# ═══════════════════════════════════════════════════════════════════════════


class TestVictorModelIntegration:
    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.testvictor.TotalLoss")
    def test_optimise_with_debug_on(self, MockTotalLoss, capsys) -> None:
        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(3.0, requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        MockTotalLoss.return_value = mock_loss_fn

        from subprojects.wp4_aeroshell_optimizer.v1.src.models.testvictor import VictorModel  # noqa: PLC0415

        model = VictorModel(in_dim=3, hidden=16, layers=2)
        dbg = DebugManager(config={"GLOBAL_DEBUG": True})

        training_config = {
            "learning_rate": 1e-3,
            "training_params": {"num_epochs": 3},
        }

        model.optimise(
            general_config={},
            training_config=training_config,
            debug_manager=dbg,
        )
        captured = capsys.readouterr().out
        assert "[DEBUG VALUES]" in captured
        assert "[DEBUG STATS]" in captured

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.testvictor.TotalLoss")
    def test_optimise_with_debug_off_no_output(self, MockTotalLoss, capsys) -> None:
        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(3.0, requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        MockTotalLoss.return_value = mock_loss_fn

        from subprojects.wp4_aeroshell_optimizer.v1.src.models.testvictor import VictorModel  # noqa: PLC0415

        model = VictorModel(in_dim=3, hidden=16, layers=2)
        dbg = DebugManager(config={"GLOBAL_DEBUG": False})

        training_config = {
            "learning_rate": 1e-3,
            "training_params": {"num_epochs": 3},
        }

        model.optimise(
            general_config={},
            training_config=training_config,
            debug_manager=dbg,
        )
        captured = capsys.readouterr().out
        assert "[DEBUG VALUES]" not in captured
        assert "[DEBUG STATS]" not in captured

    @patch("subprojects.wp4_aeroshell_optimizer.v1.src.models.testvictor.TotalLoss")
    def test_optimise_backward_compat_no_debug(self, MockTotalLoss) -> None:
        """VictorModel.optimise with no debug_manager should not crash."""
        mock_loss_fn = MagicMock()
        mock_loss_tensor = torch.tensor(3.0, requires_grad=True)
        mock_loss_fn.get_total_loss.return_value = mock_loss_tensor
        MockTotalLoss.return_value = mock_loss_fn

        from subprojects.wp4_aeroshell_optimizer.v1.src.models.testvictor import VictorModel  # noqa: PLC0415

        model = VictorModel(in_dim=3, hidden=16, layers=2)

        training_config = {
            "learning_rate": 1e-3,
            "training_params": {"num_epochs": 2},
        }

        model.optimise(
            general_config={},
            training_config=training_config,
        )
        assert mock_loss_fn.get_total_loss.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════
# 12. Edge cases and robustness
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_negative_loss(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss=-5.0)
        assert ws == []

    def test_very_large_grad_norm_warning(self) -> None:
        """Large grad_norm should NOT trigger a warning (that's insights territory)."""
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, grad_norm=1e15)
        assert ws == []

    def test_very_large_grad_norm_insight(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ins = dbg.check_insights(0, grad_norm=1e15)
        assert any("large" in i.lower() for i in ins)

    def test_multiple_warnings_in_one_call(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        ws = dbg.check_warnings(0, loss=float("nan"), grad_norm=0.0)
        assert len(ws) >= 2

    def test_begin_without_end_is_safe(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        dbg.begin_iteration(1)
        stats = dbg.end_iteration(1, loss=1.0)
        assert "iter_time_s" in stats
        dbg.stop()

    def test_end_without_begin_is_safe(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        stats = dbg.end_iteration(0, loss=1.0)
        assert "loss" in stats
        dbg.stop()

    def test_load_from_yaml_file_directly(self) -> None:
        """DebugManager with no config loads from the YAML file."""
        dbg = DebugManager()
        assert isinstance(dbg.print_raw_values, bool)
        assert isinstance(dbg.show_stats, bool)
        dbg.stop()


# ═══════════════════════════════════════════════════════════════════════════
# 9. Solar loss diagnostics tests
# ═══════════════════════════════════════════════════════════════════════════

# -- Shared solar diagnostic payloads --

_SOLAR_SCALED_DIAG = {
    "loss_value_raw": 0.042,
    "loss_value_scaled": 4.2,
    "power_conversion_factor": 100.0,
    "method": "flexicubes",
    "counter": 5,
    "area_budget_mm2": 15000.0,
    "resolution": 128,
}

_SOLAR_SIMPLE_DIAG = {
    "loss_value": 0.037,
    "method": "diso_mc",
    "counter": 3,
    "area_budget_mm2": 12000.0,
    "resolution": 64,
}

# All loss names that use the "scaled" payload (v3B, v7)
_SCALED_SOLAR_NAMES = ("solar_v3B", "solar_v7")
# All loss names that use the "simple" payload (v4, v5, v6)
_SIMPLE_SOLAR_NAMES = ("solar_v4", "solar_v5", "solar_v6")
# All solar loss names
_ALL_SOLAR_NAMES = _SCALED_SOLAR_NAMES + _SIMPLE_SOLAR_NAMES


class TestSolarLossDiagnostics:
    """Test that solar loss diagnostics integrate correctly with DebugManager."""

    # -- collect_loss_diagnostics picks up solar _last_diagnostics --

    @pytest.mark.parametrize("name", _ALL_SOLAR_NAMES)
    def test_collect_solar_diagnostics(self, name: str) -> None:
        diag = _SOLAR_SCALED_DIAG if name in _SCALED_SOLAR_NAMES else _SOLAR_SIMPLE_DIAG
        mock_inst = MagicMock()
        mock_inst._last_diagnostics = diag

        mock_loss = MagicMock()
        mock_loss._active_losses = {name: mock_inst}
        mock_loss.get_cruising_speed_diagnostics.return_value = None

        result = collect_loss_diagnostics(mock_loss)
        assert name in result
        assert result[name] == diag

    # -- log_values prints solar diagnostics --

    @pytest.mark.parametrize("name", _ALL_SOLAR_NAMES)
    def test_log_values_prints_solar(self, name: str, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        diag = _SOLAR_SCALED_DIAG if name in _SCALED_SOLAR_NAMES else _SOLAR_SIMPLE_DIAG
        dbg.log_values(0, loss=1.0, loss_diagnostics={name: diag})
        captured = capsys.readouterr().out
        assert name in captured
        assert "method" in captured

    # -- per-field YAML filtering works for solar losses --

    @pytest.mark.parametrize("name", _ALL_SOLAR_NAMES)
    def test_solar_per_field_filtering(self, name: str, capsys: pytest.CaptureFixture) -> None:
        cfg = {
            **FULL_ON_CFG,
            "log_loss_diagnostics": {name: {"method": True, "counter": False}},
        }
        dbg = DebugManager(config=cfg)
        diag = _SOLAR_SCALED_DIAG if name in _SCALED_SOLAR_NAMES else _SOLAR_SIMPLE_DIAG
        dbg.log_values(0, loss=1.0, loss_diagnostics={name: diag})
        captured = capsys.readouterr().out
        assert "method" in captured
        assert "counter" not in captured

    # -- hiding entire solar loss via per-loss false --

    @pytest.mark.parametrize("name", _ALL_SOLAR_NAMES)
    def test_solar_hide_entire_loss(self, name: str, capsys: pytest.CaptureFixture) -> None:
        cfg = {**FULL_ON_CFG, "log_loss_diagnostics": {name: False}}
        dbg = DebugManager(config=cfg)
        diag = _SOLAR_SCALED_DIAG if name in _SCALED_SOLAR_NAMES else _SOLAR_SIMPLE_DIAG
        dbg.log_values(0, loss=1.0, loss_diagnostics={name: diag})
        captured = capsys.readouterr().out
        assert name not in captured

    # -- all solar losses together in one diagnostics dict --

    def test_all_solar_losses_together(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        diag_all = {}
        for name in _SCALED_SOLAR_NAMES:
            diag_all[name] = dict(_SOLAR_SCALED_DIAG)
        for name in _SIMPLE_SOLAR_NAMES:
            diag_all[name] = dict(_SOLAR_SIMPLE_DIAG)
        dbg.log_values(0, loss=1.0, loss_diagnostics=diag_all)
        captured = capsys.readouterr().out
        for name in _ALL_SOLAR_NAMES:
            assert name in captured

    # -- solar + non-solar combined --

    def test_solar_combined_with_other_losses(self, capsys: pytest.CaptureFixture) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        diag = {
            "bb_v4_2": {"violation_frac": 0.05, "n_points": 500},
            "eikonal_v0": {"grad_norm_mean": 0.99, "n_points": 1000},
            "solar_v3B": dict(_SOLAR_SCALED_DIAG),
            "solar_v5": dict(_SOLAR_SIMPLE_DIAG),
        }
        dbg.log_values(0, loss=1.0, loss_diagnostics=diag)
        captured = capsys.readouterr().out
        assert "bb_v4_2" in captured
        assert "eikonal_v0" in captured
        assert "solar_v3B" in captured
        assert "solar_v5" in captured

    def test_solar_combined_with_cruising_speed(self) -> None:
        mock_solar = MagicMock()
        mock_solar._last_diagnostics = dict(_SOLAR_SIMPLE_DIAG)
        mock_loss = MagicMock()
        mock_loss._active_losses = {"solar_v6": mock_solar}
        mock_loss.get_cruising_speed_diagnostics.return_value = {"speed (m/s)": 22.0}

        result = collect_loss_diagnostics(mock_loss)
        assert "solar_v6" in result
        assert "cruising_speed" in result

    # -- collect_loss_diagnostics with ALL loss types present --

    def test_collect_all_loss_types_together(self) -> None:
        mocks = {}
        for name in ("bb_v4_2", "eikonal_v0", "smoothness_v0", "gaot_loss"):
            m = MagicMock()
            m._last_diagnostics = {"n_points": 100}
            mocks[name] = m
        for name in _SCALED_SOLAR_NAMES:
            m = MagicMock()
            m._last_diagnostics = dict(_SOLAR_SCALED_DIAG)
            mocks[name] = m
        for name in _SIMPLE_SOLAR_NAMES:
            m = MagicMock()
            m._last_diagnostics = dict(_SOLAR_SIMPLE_DIAG)
            mocks[name] = m

        mock_loss = MagicMock()
        mock_loss._active_losses = mocks
        mock_loss.get_cruising_speed_diagnostics.return_value = {"speed (m/s)": 20.0}

        result = collect_loss_diagnostics(mock_loss)
        expected_keys = {"bb_v4_2", "eikonal_v0", "smoothness_v0", "gaot_loss", "cruising_speed"} | set(_ALL_SOLAR_NAMES)
        assert set(result.keys()) == expected_keys

    # -- warnings: solar losses with no diagnostics should not crash --

    def test_warnings_with_solar_diagnostics_safe(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        for name in _ALL_SOLAR_NAMES:
            diag = _SOLAR_SCALED_DIAG if name in _SCALED_SOLAR_NAMES else _SOLAR_SIMPLE_DIAG
            ws = dbg.check_warnings(0, loss_diagnostics={name: diag})
            # Solar diagnostics should not trigger any warnings for healthy values
            assert isinstance(ws, list)

    # -- mixed healthy payload with all loss types produces no warnings --

    def test_all_healthy_no_warnings(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        diag = {
            "bb_v4_2": {"violation_frac": 0.05, "n_points": 500},
            "eikonal_v0": {"grad_norm_mean": 1.01},
            "smoothness_v0": {"laplacian_abs_max": 0.5},
            "solar_v3B": dict(_SOLAR_SCALED_DIAG),
            "solar_v4": dict(_SOLAR_SIMPLE_DIAG),
            "solar_v5": dict(_SOLAR_SIMPLE_DIAG),
            "solar_v6": dict(_SOLAR_SIMPLE_DIAG),
            "solar_v7": dict(_SOLAR_SCALED_DIAG),
        }
        ws = dbg.check_warnings(0, loss_diagnostics=diag)
        assert ws == []

    # -- insights: solar diagnostics passed through without crash --

    def test_insights_with_solar_diagnostics(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        for _name in _ALL_SOLAR_NAMES:
            ins = dbg.check_insights(0, loss=0.5)
            assert isinstance(ins, list)

    # -- stats: solar diagnostics don't break stats logging --

    def test_stats_with_solar_diagnostics(self) -> None:
        dbg = DebugManager(config=FULL_ON_CFG)
        dbg.begin_iteration(0)
        stats = dbg.end_iteration(0, loss=1.0)
        assert "loss" in stats
        dbg.stop()

    # -- debug.yaml listing for solar losses --

    def test_yaml_has_solar_entries(self) -> None:
        cfg = _load_debug_yaml()
        log_diag = cfg.get("log_loss_diagnostics", {})
        if isinstance(log_diag, dict):
            for name in _ALL_SOLAR_NAMES:
                assert name in log_diag, f"{name} missing from debug.yaml log_loss_diagnostics"


# ---------------------------------------------------------------------------
# Stand-alone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
