"""Comprehensive tests for the refactored TotalLoss pipeline.

Tests every public and private method of TotalLoss using mocks to avoid
heavy I/O (config files, WandB, GPU, mesh extraction, rendering).

Run:  pytest tests/wp4_aeroshell_optimizer/test_total_loss.py -v
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
MODULE = "subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss"

# ---------------------------------------------------------------------------
# Helpers & Stubs
# ---------------------------------------------------------------------------


def _make_general_config(**overrides: Any) -> dict[str, Any]:
    """Return a minimal general_config dict."""
    cfg: dict[str, Any] = {
        "general_info": {"version": "v1"},
        "active_configs": {"losses": "losses_v0_config.yaml"},
        "wandb_general": {
            "use_wandb": False,
            "use_renderer": False,
            "use_neural_renderer": False,
            "rendering_frequency": 1,
        },
    }
    cfg.update(overrides)
    return cfg


def _make_losses_config(
    *,
    alm: str = "True",
    cse: str = "True",
    alm_aggressive: str = "False",
    alm_thresholds: list[float] | None = None,
    initial_lambda: float | list[float] = 0.01,
    initial_mu: float | list[float] = 1.0,
    initial_nu: float | list[float] = 1.0,
    alpha: float = 0.99,
    gamma: float = 0.01,
    gamma_aggressive: float = 1.1,
    cse_prefactors: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return a minimal losses_config dict with active losses covering all categories."""
    if alm_thresholds is None:
        alm_thresholds = [0.1, 0.1, 0.1]
    if cse_prefactors is None:
        cse_prefactors = {"drag": 1.0, "solar": 1.0}
    return {
        "general_info": {
            "ALM": alm,
            "CSE": cse,
            "ALM_aggressive": alm_aggressive,
            "alm_thresholds": alm_thresholds,
            "initial_lambda": initial_lambda,
            "initial_mu": initial_mu,
            "initial_nu": initial_nu,
            "alpha": alpha,
            "gamma": gamma,
            "gamma_aggressive": gamma_aggressive,
            "cse_prefactors": cse_prefactors,
        },
        "losses": {
            "aero": {
                "gaot_loss": [1, 1.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/aerodynamic_loss/gaot_loss.py"],
            },
            "bb": {
                "bb_v3": [1, 10, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/bb_loss/bb_v3.py"],
            },
            "eikonal": {
                "eikonal_v0": [1, 10.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/eikonal_loss/eikonal_v0.py"],
            },
            "smoothness": {
                "smoothness_v0": [1, 0.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/smoothness_loss/smoothness_v0.py"],
            },
            "surface_area": {
                "surface_loss_v0": [
                    1,
                    1.0,
                    "subprojects/wp4_aeroshell_optimizer/v2/src/losses/surface_area_loss/surface_loss_v0.py",
                ],
            },
            "solar": {
                "solar_v3B": [1, 50.0, "subprojects/wp4_aeroshell_optimizer/v2/src/losses/solar_loss/solar_v3B.py"],
            },
            "total": {
                "total_loss": [1, 1.0, "path"],
            },
        },
        "get_points_params": {"num_points": 10, "bbox_size": 1.0},
    }


def _make_logging_config() -> dict[str, Any]:
    return {"memory_diagnostics": False, "mesh_statistics": False, "gradient_memory_diagnostics": False}


def _make_renderer_config() -> dict[str, Any]:
    return {
        "mesh_extraction": {
            "algorithm": "mc",
            "resolution": 32,
            "bbox": [[-1, 1], [-1, 1], [-1, 1]],
            "isovalue": 0.0,
            "batch_size": 64,
            "device": "cpu",
            "verbose": False,
        },
    }


class StubLossInstance:
    """Mimics ParentLoss - returns a fixed tensor from get_loss()."""

    def __init__(self, value: float, requires_grad: bool = True, device: str = "cpu", **_kw: Any) -> None:
        self._value = value
        self._requires_grad = requires_grad
        self._device = device

    def get_loss(self, model: Any, epoch_name: int) -> torch.Tensor:  # noqa: ARG002
        t = torch.tensor(self._value, device=self._device, dtype=torch.float32)
        if self._requires_grad:
            t = t.clone().requires_grad_(True)
        return t

    def log_gradients(self, epoch_name: int) -> None:
        pass


class SimpleModel(nn.Module):
    """Tiny model that satisfies BaseModel's interface for tests."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Config dispatcher used by all tests
# ---------------------------------------------------------------------------


def _config_dispatcher(
    general_cfg: dict | None = None,
    losses_cfg: dict | None = None,
    logging_cfg: dict | None = None,
    renderer_cfg: dict | None = None,
):
    """Return a side_effect callable for load_config."""
    gc = general_cfg or _make_general_config()
    lc = losses_cfg or _make_losses_config()
    logc = logging_cfg or _make_logging_config()
    rc = renderer_cfg or _make_renderer_config()

    def _dispatch(v: str = "v1", cfg_name: str = "general_config.yaml") -> dict:  # noqa: ARG001
        if cfg_name == "general_config.yaml":
            return gc
        if "losses" in cfg_name:
            return lc
        if cfg_name == "logging.yaml":
            return logc
        if "renderer" in cfg_name:
            return rc
        return gc  # fallback

    return _dispatch


# ---------------------------------------------------------------------------
# Fixture: build a TotalLoss with all heavy deps mocked
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_total_loss():
    """Factory that returns (TotalLoss instance, dict of mocks).

    Usage:
        tl, mocks = make_total_loss()  # default configs
        tl, mocks = make_total_loss(losses_cfg=custom)
    """

    def _factory(
        general_cfg: dict | None = None,
        losses_cfg: dict | None = None,
        logging_cfg: dict | None = None,
        renderer_cfg: dict | None = None,
        loss_values: dict[str, float] | None = None,
    ):
        lv: dict[str, float] = loss_values or {
            "gaot_loss": 2.0,
            "bb_v3": 0.5,
            "eikonal_v0": 0.3,
            "smoothness_v0": 0.1,
            "surface_loss_v0": 1.0,
            "solar_v3B": 100.0,
        }

        stub_instances: dict[str, StubLossInstance] = {
            name: StubLossInstance(value=val, device="cpu") for name, val in lv.items()
        }

        def _fake_load_class(path: str, family: str):  # noqa: ARG001
            """Return a callable that, when invoked, returns the pre-built stub."""
            stem = Path(path).stem
            for name, stub in stub_instances.items():
                if name == stem or name.startswith(stem):
                    return lambda device="cpu", loss_name="", losses_config=None, _s=stub: _s  # noqa: ARG005
            return lambda device="cpu", _loss_name="", _losses_config=None: StubLossInstance(0.0, device=device)

        mocks: dict[str, Any] = {}

        patches = [
            patch(f"{MODULE}.load_config", side_effect=_config_dispatcher(general_cfg, losses_cfg, logging_cfg, renderer_cfg)),
            patch(f"{MODULE}.load_class_from_path", side_effect=_fake_load_class),
            patch(f"{MODULE}.MeshExtractor", return_value=MagicMock()),
            patch(f"{MODULE}.MeshStatistics", return_value=MagicMock()),
            patch(f"{MODULE}.wandb", MagicMock()),
        ]

        for p in patches:
            m = p.start()
            key = p.attribute if hasattr(p, "attribute") and p.attribute else str(p)
            mocks[key] = m

        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss  # noqa: PLC0415

        tl = TotalLoss(dev="cpu", renderer_manager=None)

        def _cleanup():
            for p in patches:
                p.stop()

        return tl, mocks, _cleanup

    return _factory


# ===========================================================================
# Test: __init__ and _load_configs                                              # noqa: ERA001
# ===========================================================================


class TestInit:
    """Tests for __init__ and its sub-helpers."""

    def test_device_normalisation_from_string(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            assert isinstance(tl._device, torch.device)
            assert tl._device == torch.device("cpu")
        finally:
            cleanup()

    def test_alm_state_initialised(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            assert tl.lamdas.shape == (3, 1)
            assert tl.mus.shape == (3, 1)
            assert tl.nus.shape == (3, 1)
            assert tl.alphas.shape == (3, 1)
            assert torch.allclose(tl.lamdas, torch.full((3, 1), 0.01))
            assert torch.allclose(tl.mus, torch.full((3, 1), 1.0))
            assert torch.allclose(tl.nus, torch.full((3, 1), 1.0))
        finally:
            cleanup()

    def test_constraint_order_includes_connectedness(self, make_total_loss):
        cfg = _make_losses_config()
        cfg["general_info"]["alm_thresholds"] = [0.1, 0.1, 0.1, 0.1]
        cfg["general_info"]["initial_lambda"] = [0.01, 0.01, 0.01, 0.01]
        cfg["general_info"]["initial_mu"] = [1.0, 1.0, 1.0, 1.0]
        cfg["general_info"]["initial_nu"] = [1.0, 1.0, 1.0, 1.0]
        cfg["losses"]["connectedness"] = {
            "connectedness_v1": [
                1,
                1.0,
                "subprojects/wp4_aeroshell_optimizer/v2/src/losses/connectedness_loss/connectedness_v1.py",
            ]
        }

        tl, _m, cleanup = make_total_loss(
            losses_cfg=cfg,
            loss_values={
                "gaot_loss": 2.0,
                "bb_v3": 0.5,
                "eikonal_v0": 0.3,
                "smoothness_v0": 0.1,
                "connectedness_v1": 0.2,
                "surface_loss_v0": 1.0,
                "solar_v3B": 100.0,
            },
        )
        try:
            assert list(tl._constraint_losses.keys()) == ["bb", "eikonal", "smoothness", "connectedness"]
            assert tl.lamdas.shape == (4, 1)
            assert tl.mus.shape == (4, 1)
            assert tl.nus.shape == (4, 1)
        finally:
            cleanup()

    def test_manual_mode_no_alm_state(self, make_total_loss):
        """ALM='False' → manual mode: no lamdas/mus/nus attributes."""
        tl, _m, cleanup = make_total_loss(losses_cfg=_make_losses_config(alm="False"))
        try:
            assert tl._use_alm is False
            assert not hasattr(tl, "lamdas")
            assert not hasattr(tl, "mus")
        finally:
            cleanup()

    def test_wandb_flags_set_from_config(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            assert tl._use_wandb is False
            assert tl._use_renderer is False
        finally:
            cleanup()

    def test_logging_flags_defaults(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            assert tl._memory_diagnostics is False
            assert tl._mesh_statistics_enabled is False
            assert tl._gradient_memory_diagnostics is False
        finally:
            cleanup()

    def test_per_constraint_alm_init_from_lists(self, make_total_loss):
        """initial_lambda/mu/nu as lists → distinct per-constraint values."""
        cfg = _make_losses_config(
            initial_lambda=[0.1, 0.2, 0.3],
            initial_mu=[2.0, 3.0, 4.0],
            initial_nu=[0.5, 0.6, 0.7],
        )
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            assert tl.lamdas.shape == (3, 1)
            assert tl.lamdas[0].item() == pytest.approx(0.1)
            assert tl.lamdas[1].item() == pytest.approx(0.2)
            assert tl.lamdas[2].item() == pytest.approx(0.3)
            assert tl.mus[0].item() == pytest.approx(2.0)
            assert tl.mus[1].item() == pytest.approx(3.0)
            assert tl.nus[2].item() == pytest.approx(0.7)
        finally:
            cleanup()

    def test_per_constraint_scalar_broadcasts(self, make_total_loss):
        """Scalar initial_lambda → same value for all constraints."""
        cfg = _make_losses_config(initial_lambda=0.05, initial_mu=2.0, initial_nu=0.5)
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            assert torch.allclose(tl.lamdas, torch.full((3, 1), 0.05))
            assert torch.allclose(tl.mus, torch.full((3, 1), 2.0))
            assert torch.allclose(tl.nus, torch.full((3, 1), 0.5))
        finally:
            cleanup()

    def test_use_cse_flag(self, make_total_loss):
        tl_on, _, c1 = make_total_loss(losses_cfg=_make_losses_config(cse="True"))
        tl_off, _, c2 = make_total_loss(losses_cfg=_make_losses_config(cse="False"))
        try:
            assert tl_on._use_cse is True
            assert tl_off._use_cse is False
        finally:
            c1()
            c2()


# ===========================================================================
# Test: _init_active_losses / Category classification
# ===========================================================================


class TestCategoryClassification:
    """Verify losses are classified into objective / constraint / auxiliary."""

    def test_objective_losses_contain_aero_and_solar(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            assert "aero" in tl._objective_losses
            assert "solar" in tl._objective_losses
            assert tl._objective_losses["aero"][0] == "gaot_loss"
            assert tl._objective_losses["solar"][0] == "solar_v3B"
        finally:
            cleanup()

    def test_constraint_losses_ordered_bb_eikonal_smoothness(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            keys = list(tl._constraint_losses.keys())
            assert keys == ["bb", "eikonal", "smoothness"]
            assert tl._constraint_losses["bb"][0] == "bb_v3"
            assert tl._constraint_losses["eikonal"][0] == "eikonal_v0"
            assert tl._constraint_losses["smoothness"][0] == "smoothness_v0"
        finally:
            cleanup()

    def test_auxiliary_losses_contain_surface_area(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            assert "surface_loss_v0" in tl._auxiliary_losses
        finally:
            cleanup()

    def test_total_category_is_skipped(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            all_names = set(tl._active_losses.keys())
            assert "total_loss" not in all_names
        finally:
            cleanup()

    def test_active_losses_contains_all_six(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            expected = {"gaot_loss", "bb_v3", "eikonal_v0", "smoothness_v0", "surface_loss_v0", "solar_v3B"}
            assert set(tl._active_losses.keys()) == expected
        finally:
            cleanup()

    def test_disabled_loss_not_loaded(self, make_total_loss):
        """A loss with compute=0 should not appear in _active_losses."""
        cfg = _make_losses_config()
        cfg["losses"]["aero"]["gaot_loss"] = [
            0,
            1.0,
            "subprojects/wp4_aeroshell_optimizer/v2/src/losses/aerodynamic_loss/gaot_loss.py",
        ]
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            assert "gaot_loss" not in tl._active_losses
            assert "aero" not in tl._objective_losses
        finally:
            cleanup()


# ===========================================================================
# Test: _compute_raw_losses                                                     # noqa: ERA001
# ===========================================================================


class TestComputeRawLosses:
    """Unit tests for the raw loss computation stage."""

    def test_returns_dict_with_all_active_losses(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            assert set(raw.keys()) == set(tl._active_losses.keys())
        finally:
            cleanup()

    def test_values_are_tensors(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            for v in raw.values():
                assert isinstance(v, torch.Tensor)
        finally:
            cleanup()

    def test_values_match_stub_outputs(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            assert raw["gaot_loss"].item() == pytest.approx(2.0)
            assert raw["bb_v3"].item() == pytest.approx(0.5)
            assert raw["solar_v3B"].item() == pytest.approx(100.0)
        finally:
            cleanup()


# ===========================================================================
# Test: _apply_weighting                                                    # noqa: ERA001
# ===========================================================================


class TestALMWeighting:
    """Bit-for-bit equivalence tests for the ALM weighting path."""

    @staticmethod
    def _hand_compute_alm(
        o_val: float,
        c_vals: list[float],
        lamdas: list[float],
        mus: list[float],
    ) -> float:
        """Hand-compute expected total: o + sum[lambda_i * c_i + 0.5 * mu_i * c_i^2]."""
        total = o_val
        for i, c in enumerate(c_vals):
            total += lamdas[i] * c + 0.5 * mus[i] * c**2
        return total

    def test_alm_total_loss_value_with_cse_off(self, make_total_loss):
        """ALM=True, CSE=False: objectives get manual weights, constraints get ALM.

        gaot_loss=2.0 * w=1.0, solar_v3B=100.0 * w=50.0, surface_loss_v0=1.0 * w=1.0.
        Constraints: ALM(bb=0.5, eikonal=0.3, smoothness=0.1).
        """
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            total = tl._apply_weighting(raw, torch.device("cpu"))

            lam_after = tl.lamdas.view(-1).tolist()
            mu_after = tl.mus.view(-1).tolist()

            # Manual objective contribution: gaot*1 + solar*50
            obj_expected = 2.0 * 1.0 + 100.0 * 50.0
            # Auxiliary contribution: surface_loss_v0 * 1.0
            aux_expected = 1.0 * 1.0
            # ALM constraint contribution
            c_vals = [0.5, 0.3, 0.1]
            alm_expected = sum(lam_after[i] * c_vals[i] + 0.5 * mu_after[i] * c_vals[i] ** 2 for i in range(3))

            expected = obj_expected + aux_expected + alm_expected
            assert total.item() == pytest.approx(expected, rel=1e-5)
        finally:
            cleanup()

    def test_last_weighted_losses_populated_alm_cse(self, make_total_loss):
        """ALM=True, CSE=True: cse_objective + 3 alm constraints + auxiliary."""
        cfg = _make_losses_config(alm="True", cse="True")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                tl._apply_weighting(raw, torch.device("cpu"))

            assert "cse_objective" in tl._last_weighted_losses
            assert "alm_bb_v3" in tl._last_weighted_losses
            assert "alm_eikonal_v0" in tl._last_weighted_losses
            assert "alm_smoothness_v0" in tl._last_weighted_losses
            assert "surface_loss_v0" in tl._last_weighted_losses
            assert len(tl._last_weighted_losses) == 5
        finally:
            cleanup()

    def test_last_weighted_losses_populated_alm_manual_obj(self, make_total_loss):
        """ALM=True, CSE=False: manual objective keys + alm constraints + auxiliary."""
        cfg = _make_losses_config(alm="True", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            assert "gaot_loss" in tl._last_weighted_losses
            assert "solar_v3B" in tl._last_weighted_losses
            assert "alm_bb_v3" in tl._last_weighted_losses
            assert "surface_loss_v0" in tl._last_weighted_losses
        finally:
            cleanup()

    def test_auxiliary_losses_included_with_manual_weight(self, make_total_loss):
        """Auxiliary losses should always be present with their manual weight."""
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            assert "surface_loss_v0" in tl._last_weighted_losses
            # weight from config is 1.0, raw value is 1.0 → 1.0
            assert tl._last_weighted_losses["surface_loss_v0"].item() == pytest.approx(1.0)
        finally:
            cleanup()

    def test_cruising_speed_stored_when_cse_on(self, make_total_loss):
        """When CSE=True, _last_cruising_speed should be a tensor."""
        cfg = _make_losses_config(cse="True")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                tl._apply_weighting(raw, torch.device("cpu"))
            assert tl._last_cruising_speed is not None
            assert tl._last_cruising_speed.item() == pytest.approx(25.0)
        finally:
            cleanup()

    def test_cruising_speed_none_when_cse_off(self, make_total_loss):
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))
            assert tl._last_cruising_speed is None
        finally:
            cleanup()


# ===========================================================================
# Test: Graph independence of _last_weighted_losses
# ===========================================================================


class TestGraphIndependence:
    """Each entry in _last_weighted_losses must have an independent graph."""

    def test_no_shared_stack_backward(self, make_total_loss):
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            grad_fns = [v.grad_fn for v in tl._last_weighted_losses.values() if v.grad_fn is not None]

            assert len(grad_fns) == len(set(grad_fns)), "Shared grad_fn detected - StackBackward leak!"
        finally:
            cleanup()

    def test_individual_backward_does_not_raise(self, make_total_loss):
        """Calling .backward(retain_graph=False) on each component must not raise."""
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            for weighted_loss in tl._last_weighted_losses.values():
                if not weighted_loss.requires_grad:
                    continue
                weighted_loss.backward(retain_graph=False)
        finally:
            cleanup()


# ===========================================================================
# Test: ALM multiplier update (mathematical correctness)
# ===========================================================================


class TestALMMultiplierUpdate:
    """Verify lambda, mu, nu updates match the analytical formulas."""

    def test_rms_nu_update(self, make_total_loss):
        """nu <- alpha*nu + (1-alpha)*c^2.  With alpha=0.99, nu0=1.0, c=0.5 -> nu=0.9925."""
        cfg = _make_losses_config(alpha=0.99, alm_aggressive="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            expected_nu_bb = 0.99 * 1.0 + 0.01 * (0.5**2)  # = 0.9925
            assert tl.nus[0].item() == pytest.approx(expected_nu_bb, rel=1e-6)
        finally:
            cleanup()

    def test_rms_mu_update(self, make_total_loss):
        """mu = max(gamma/(sqrt(nu)+eps), gamma/(tau+eps)). No threshold -> raw RMS formula."""
        cfg = _make_losses_config(gamma=0.01, alpha=0.99, alm_aggressive="False", alm_thresholds=[])
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            eps = 1e-16
            expected_nu_bb = 0.99 * 1.0 + 0.01 * (0.5**2)
            expected_mu_bb = 0.01 / (math.sqrt(expected_nu_bb) + eps)
            assert tl.mus[0].item() == pytest.approx(expected_mu_bb, rel=1e-5)
        finally:
            cleanup()

    def test_rms_lambda_update(self, make_total_loss):
        """lambda <- lambda + mu*c.  Initial lambda=0.01. No thresholds -> raw RMS mu."""
        cfg = _make_losses_config(gamma=0.01, alpha=0.99, initial_lambda=0.01, alm_aggressive="False", alm_thresholds=[])
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            eps = 1e-16
            nu_after = 0.99 * 1.0 + 0.01 * (0.5**2)
            mu_after = 0.01 / (math.sqrt(nu_after) + eps)
            expected_lam = 0.01 + mu_after * 0.5
            assert tl.lamdas[0].item() == pytest.approx(expected_lam, rel=1e-5)
        finally:
            cleanup()

    def test_rms_mu_floored_by_threshold(self, make_total_loss):
        """When threshold is set, mu >= gamma/(threshold + eps)."""
        cfg = _make_losses_config(
            gamma=0.01,
            alpha=0.99,
            alm_aggressive="False",
            alm_thresholds=[0.1, 0.1, 0.1],
        )
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            eps = 1e-16
            min_mu = 0.01 / (0.1 + eps)
            assert tl.mus[0].item() >= min_mu - 1e-6
        finally:
            cleanup()

    def test_capu_mu_scaling(self, make_total_loss):
        """CAPU: if |c| > 0.25*nu -> mu *= gamma_aggressive."""
        cfg = _make_losses_config(
            alm_aggressive="True",
            gamma_aggressive=1.1,
            initial_mu=1.0,
            initial_nu=0.1,  # small nu so |c| > 0.25*nu triggers
        )
        tl, _m, cleanup = make_total_loss(
            losses_cfg=cfg,
            loss_values={
                "gaot_loss": 2.0,
                "bb_v3": 5.0,
                "eikonal_v0": 5.0,
                "smoothness_v0": 5.0,
                "surface_loss_v0": 1.0,
                "solar_v3B": 100.0,
            },
        )
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)

            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                tl._apply_weighting(raw, torch.device("cpu"))

            assert tl.mus[0].item() == pytest.approx(1.0 * 1.1, rel=1e-5)
        finally:
            cleanup()

    def test_capu_nu_update(self, make_total_loss):
        """CAPU: nu <- alpha*nu + (1-alpha)*|c|."""
        cfg = _make_losses_config(
            alm_aggressive="True",
            alpha=0.99,
            initial_nu=1.0,
        )
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                tl._apply_weighting(raw, torch.device("cpu"))

            # bb_v3 = 0.5 -> nu = 0.99 * 1.0 + 0.01 * |0.5| = 0.995
            expected_nu = 0.99 * 1.0 + 0.01 * abs(0.5)
            assert tl.nus[0].item() == pytest.approx(expected_nu, rel=1e-6)
        finally:
            cleanup()


# ===========================================================================
# Test: _apply_weighting                                                  # noqa: ERA001
# ===========================================================================


class TestManualWeighting:
    """Tests for the manual (config-weight) weighting path."""

    def test_manual_weights_stored_from_config(self, make_total_loss):
        """_manual_weights should contain the second element of each [compute, weight, path] tuple."""
        tl, _m, cleanup = make_total_loss(losses_cfg=_make_losses_config(alm="False", cse="False"))
        try:
            assert tl._manual_weights["gaot_loss"] == pytest.approx(1.0)
            assert tl._manual_weights["bb_v3"] == pytest.approx(10.0)
            assert tl._manual_weights["eikonal_v0"] == pytest.approx(10.0)
            assert tl._manual_weights["smoothness_v0"] == pytest.approx(0.0)
            assert tl._manual_weights["solar_v3B"] == pytest.approx(50.0)
        finally:
            cleanup()

    def test_manual_total_is_weighted_sum(self, make_total_loss):
        """Total = sum(loss_i * weight_i) for all active losses."""
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            total = tl._apply_weighting(raw, torch.device("cpu"))

            expected = sum(raw[n].item() * tl._manual_weights[n] for n in raw)
            assert total.item() == pytest.approx(expected, rel=1e-5)
        finally:
            cleanup()

    def test_manual_last_weighted_losses_contain_all_active(self, make_total_loss):
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            assert set(tl._last_weighted_losses.keys()) == set(tl._active_losses.keys())
        finally:
            cleanup()

    def test_manual_no_cruising_speed(self, make_total_loss):
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))
            assert tl._last_cruising_speed is None
        finally:
            cleanup()

    def test_manual_graph_independence(self, make_total_loss):
        """Each entry in _last_weighted_losses must have an independent graph."""
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            grad_fns = [v.grad_fn for v in tl._last_weighted_losses.values() if v.grad_fn is not None]
            assert len(grad_fns) == len(set(grad_fns)), "Shared grad_fn detected!"
        finally:
            cleanup()

    def test_manual_individual_backward(self, make_total_loss):
        """Per-component backward must not raise."""
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            tl._apply_weighting(raw, torch.device("cpu"))

            for weighted_loss in tl._last_weighted_losses.values():
                if not weighted_loss.requires_grad:
                    continue
                weighted_loss.backward(retain_graph=False)
        finally:
            cleanup()

    def test_manual_get_total_loss_end_to_end(self, make_total_loss):
        """get_total_loss works end-to-end in manual mode."""
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            total = tl.get_total_loss(model, epoch_name=0)

            assert isinstance(total, torch.Tensor)
            assert torch.isfinite(total).all()
            recomputed = sum(v.item() for v in tl._last_weighted_losses.values())
            assert total.item() == pytest.approx(recomputed, rel=1e-5)
        finally:
            cleanup()

    def test_manual_telemetry_no_alm_params(self, make_total_loss):
        """In manual mode, wandb.log should NOT contain alm/ keys."""
        gc = _make_general_config()
        gc["wandb_general"]["use_wandb"] = True
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(general_cfg=gc, losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            total = tl._apply_weighting(raw, torch.device("cpu"))
            loss_log = {n: t.item() for n, t in raw.items()}
            loss_log["total_loss"] = total.item()

            from subprojects.wp4_aeroshell_optimizer.v2.src.losses import total_loss as tl_mod  # noqa: PLC0415

            tl_mod.wandb.log = MagicMock()
            tl._handle_telemetry(loss_log, 0, torch.device("cpu"), model, total)

            assert tl_mod.wandb.log.call_count == 1
            logged_dict = tl_mod.wandb.log.call_args[0][0]
            alm_keys = [k for k in logged_dict if k.startswith("alm/")]
            assert alm_keys == [], f"Unexpected ALM keys in manual mode: {alm_keys}"
        finally:
            cleanup()


# ===========================================================================
# Test: CSE prefactors
# ===========================================================================


class TestCSEPrefactors:
    """Test that CSE prefactors scale objectives before the cruising-speed solver."""

    def test_prefactors_passed_to_cruising_speed(self, make_total_loss):
        """Verify drag/solar passed to cruising_speed include prefactor scaling."""
        cfg = _make_losses_config(cse="True", cse_prefactors={"drag": 2.0, "solar": 0.5})
        gc = _make_general_config()
        tl, _m, cleanup = make_total_loss(general_cfg=gc, losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)

            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(10.0, requires_grad=True)) as mock_cse:
                tl._apply_weighting(raw, torch.device("cpu"))

            mock_cse.assert_called_once()
            call_kwargs = mock_cse.call_args[1]
            # drag raw=2.0 * pf=2.0 = 4.0;  solar raw=100.0 * pf=0.5 = 50.0
            assert call_kwargs["drag_force_at_25ms"].item() == pytest.approx(4.0)
            assert call_kwargs["solar_power"].item() == pytest.approx(50.0)
        finally:
            cleanup()

    def test_default_prefactors_are_one(self, make_total_loss):
        """Without cse_prefactors in config, defaults to 1.0."""
        gi = _make_losses_config(cse="True")
        del gi["general_info"]["cse_prefactors"]  # simulates missing key
        tl, _m, cleanup = make_total_loss(losses_cfg=gi)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)

            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(10.0, requires_grad=True)) as mock_cse:
                tl._apply_weighting(raw, torch.device("cpu"))

            call_kwargs = mock_cse.call_args[1]
            # raw values directly (pf=1.0)
            assert call_kwargs["drag_force_at_25ms"].item() == pytest.approx(2.0)
            assert call_kwargs["solar_power"].item() == pytest.approx(100.0)
        finally:
            cleanup()


# ===========================================================================
# Test: all four weighting modes end-to-end
# ===========================================================================


class TestFourModes:
    """Verify all four ALM x CSE combinations produce valid results."""

    def test_alm_cse(self, make_total_loss):
        """ALM=True, CSE=True: cse_objective + alm constraints + auxiliary."""
        cfg = _make_losses_config(alm="True", cse="True")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                total = tl.get_total_loss(model, epoch_name=0)
            assert torch.isfinite(total)
            assert "cse_objective" in tl._last_weighted_losses
            assert any(k.startswith("alm_") for k in tl._last_weighted_losses)
            assert tl._last_cruising_speed is not None
        finally:
            cleanup()

    def test_alm_manual_obj(self, make_total_loss):
        """ALM=True, CSE=False: manual objectives + alm constraints."""
        cfg = _make_losses_config(alm="True", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            total = tl.get_total_loss(model, epoch_name=0)
            assert torch.isfinite(total)
            # Objectives should appear by name, not as cse_objective
            assert "gaot_loss" in tl._last_weighted_losses
            assert "solar_v3B" in tl._last_weighted_losses
            assert any(k.startswith("alm_") for k in tl._last_weighted_losses)
            assert tl._last_cruising_speed is None
        finally:
            cleanup()

    def test_manual_cstr_cse(self, make_total_loss):
        """ALM=False, CSE=True: CSE objective + manual constraints."""
        cfg = _make_losses_config(alm="False", cse="True")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                total = tl.get_total_loss(model, epoch_name=0)
            assert torch.isfinite(total)
            assert "cse_objective" in tl._last_weighted_losses
            # Constraints should appear by name, not alm_ prefixed
            assert "bb_v3" in tl._last_weighted_losses
            assert not any(k.startswith("alm_") for k in tl._last_weighted_losses)
            assert tl._last_cruising_speed is not None
        finally:
            cleanup()

    def test_fully_manual(self, make_total_loss):
        """ALM=False, CSE=False: everything manual-weighted."""
        cfg = _make_losses_config(alm="False", cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            total = tl.get_total_loss(model, epoch_name=0)
            assert torch.isfinite(total)
            assert set(tl._last_weighted_losses.keys()) == set(tl._active_losses.keys())
            assert tl._last_cruising_speed is None
        finally:
            cleanup()

    def test_backward_all_four_modes(self, make_total_loss):
        """Per-component backward works in all four modes without raising."""
        modes = [
            {"alm": "True", "cse": "True"},
            {"alm": "True", "cse": "False"},
            {"alm": "False", "cse": "True"},
            {"alm": "False", "cse": "False"},
        ]
        for mode in modes:
            cfg = _make_losses_config(**mode)
            tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
            try:
                model = SimpleModel()
                with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                    tl.get_total_loss(model, epoch_name=0)
                for wl in tl._last_weighted_losses.values():
                    if wl.requires_grad:
                        wl.backward(retain_graph=False)
            finally:
                cleanup()


# ===========================================================================
# Test: _handle_telemetry                                                        # noqa: ERA001
# ===========================================================================


class TestHandleTelemetry:
    """Tests for the telemetry/logging dispatch method."""

    def test_gradient_logging_called_even_without_wandb(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                total = tl._apply_weighting(raw, torch.device("cpu"))
            loss_log = {n: t.item() for n, t in raw.items()}
            loss_log["total_loss"] = total.item()

            for inst in tl._active_losses.values():
                inst.log_gradients = MagicMock()

            tl._handle_telemetry(loss_log, 0, torch.device("cpu"), model, total)

            for inst in tl._active_losses.values():
                inst.log_gradients.assert_called_once_with(0)
        finally:
            cleanup()

    def test_wandb_log_called_with_loss_dict_when_enabled(self, make_total_loss):
        gc = _make_general_config()
        gc["wandb_general"]["use_wandb"] = True
        tl, _m, cleanup = make_total_loss(general_cfg=gc)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=5)
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                total = tl._apply_weighting(raw, torch.device("cpu"))
            loss_log = {n: t.item() for n, t in raw.items()}
            loss_log["total_loss"] = total.item()

            from subprojects.wp4_aeroshell_optimizer.v2.src.losses import total_loss as tl_mod  # noqa: PLC0415

            tl_mod.wandb.log = MagicMock()

            tl._handle_telemetry(loss_log, 5, torch.device("cpu"), model, total)

            # After batching fix: single wandb.log call per step
            assert tl_mod.wandb.log.call_count == 1

            logged_dict = tl_mod.wandb.log.call_args[0][0]
            all_logged_keys = set(logged_dict.keys())

            assert "cruising_speed_ms" in all_logged_keys
            assert any(k.startswith("weighted_losses/") for k in all_logged_keys)
            assert any(k.startswith("alm_params/lambda_") for k in all_logged_keys)
            assert any(k.startswith("alm_params/mu_") for k in all_logged_keys)
            assert any(k.startswith("alm_params/nu_") for k in all_logged_keys)
            # Verify step parameter
            assert tl_mod.wandb.log.call_args[1].get("step") == 5
        finally:
            cleanup()

    def test_no_cruising_speed_logged_when_cse_off(self, make_total_loss):
        gc = _make_general_config()
        gc["wandb_general"]["use_wandb"] = True
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(general_cfg=gc, losses_cfg=cfg)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            total = tl._apply_weighting(raw, torch.device("cpu"))
            loss_log = {n: t.item() for n, t in raw.items()}
            loss_log["total_loss"] = total.item()

            from subprojects.wp4_aeroshell_optimizer.v2.src.losses import total_loss as tl_mod  # noqa: PLC0415

            tl_mod.wandb.log = MagicMock()

            tl._handle_telemetry(loss_log, 0, torch.device("cpu"), model, total)

            # Single batched call
            assert tl_mod.wandb.log.call_count == 1
            logged_dict = tl_mod.wandb.log.call_args[0][0]
            all_logged_keys = set(logged_dict.keys())

            assert "cruising_speed_ms" not in all_logged_keys
        finally:
            cleanup()

    def test_skip_rendering_message_when_renderer_off(self, make_total_loss, capsys):
        gc = _make_general_config()
        gc["wandb_general"]["use_wandb"] = True
        tl, _m, cleanup = make_total_loss(general_cfg=gc)
        try:
            model = SimpleModel()
            raw = tl._compute_raw_losses(model, epoch_name=0)
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                total = tl._apply_weighting(raw, torch.device("cpu"))
            loss_log = {"total_loss": total.item()}

            from subprojects.wp4_aeroshell_optimizer.v2.src.losses import total_loss as tl_mod  # noqa: PLC0415

            tl_mod.wandb.log = MagicMock()

            tl._handle_telemetry(loss_log, 0, torch.device("cpu"), model, total)

            captured = capsys.readouterr()
            assert "Skipping rendering" in captured.out
        finally:
            cleanup()


# ===========================================================================
# Test: _log_memory_diagnostics                                                  # noqa: ERA001
# ===========================================================================


class TestLogMemoryDiagnostics:
    """Tests for the memory diagnostics helper."""

    def test_noop_when_disabled(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            tl._memory_diagnostics = False
            tl._log_memory_diagnostics(0, torch.device("cpu"))
        finally:
            cleanup()

    def test_noop_on_cpu_even_if_enabled(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            tl._memory_diagnostics = True
            tl._log_memory_diagnostics(0, torch.device("cpu"))
        finally:
            cleanup()


# ===========================================================================
# Test: _run_cad_vis                                                             # noqa: ERA001
# ===========================================================================


class TestRunCadVis:
    """Tests for the CAD visualisation helper."""

    def test_creates_stl_and_renders(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            verts = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=torch.float32)
            faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

            from subprojects.wp4_aeroshell_optimizer.v2.src.losses import total_loss as tl_mod  # noqa: PLC0415

            mock_cad = MagicMock()
            tl_mod.CADVisualizer = MagicMock(return_value=mock_cad)
            tl_mod.wandb = MagicMock()
            tl_mod.trimesh = MagicMock()

            tl._run_cad_vis(verts, faces, epoch_name=42)

            tl_mod.trimesh.Trimesh.assert_called_once()
            tl_mod.CADVisualizer.assert_called_once()
            mock_cad.render_until_blue_detected.assert_called_once()
        finally:
            cleanup()


# ===========================================================================
# Test: get_total_loss (end-to-end orchestrator)
# ===========================================================================


class TestGetTotalLoss:
    """End-to-end tests for the orchestrator."""

    def test_returns_tensor(self, make_total_loss):
        tl, _m, cleanup = make_total_loss()
        try:
            model = SimpleModel()
            with patch(f"{MODULE}.cruising_speed", return_value=torch.tensor(25.0, requires_grad=True)):
                total = tl.get_total_loss(model, epoch_name=0)
            assert isinstance(total, torch.Tensor)
            assert total.numel() >= 1
            assert torch.isfinite(total).all()
        finally:
            cleanup()

    def test_weighted_losses_keys_stable_across_epochs(self, make_total_loss):
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            tl.get_total_loss(model, epoch_name=0)
            keys_0 = set(tl._last_weighted_losses.keys())

            tl.get_total_loss(model, epoch_name=1)
            keys_1 = set(tl._last_weighted_losses.keys())

            assert keys_0 == keys_1
        finally:
            cleanup()

    def test_total_loss_is_sum_of_weighted_components(self, make_total_loss):
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            total = tl.get_total_loss(model, epoch_name=0)

            recomputed = sum(v.item() for v in tl._last_weighted_losses.values())
            assert total.item() == pytest.approx(recomputed, rel=1e-5)
        finally:
            cleanup()

    def test_backward_compatible_with_training_loop_pattern(self, make_total_loss):
        """Simulate the WIRE_raschoff per-component backward pattern."""
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            optimizer.zero_grad()

            tl.get_total_loss(model, epoch_name=0)

            for weighted_loss in tl._last_weighted_losses.values():
                if not weighted_loss.requires_grad:
                    continue
                weighted_loss.backward(retain_graph=False)

            optimizer.step()
        finally:
            cleanup()

    def test_multiple_epochs_update_alm_state(self, make_total_loss):
        """ALM state (lambda, mu, nu) should evolve across epochs."""
        cfg = _make_losses_config(cse="False")
        tl, _m, cleanup = make_total_loss(losses_cfg=cfg)
        try:
            model = SimpleModel()
            lam_initial = tl.lamdas.clone()

            tl.get_total_loss(model, epoch_name=0)
            lam_after_1 = tl.lamdas.clone()

            tl.get_total_loss(model, epoch_name=1)
            lam_after_2 = tl.lamdas.clone()

            assert not torch.allclose(lam_initial, lam_after_1)
            assert not torch.allclose(lam_after_1, lam_after_2)
        finally:
            cleanup()


# ===========================================================================
# Test: Loss-competition dict exclusion of total_loss (Plan 020)
# ===========================================================================


class TestCompetitionDictExclusion:
    """total_loss must never appear as a competitor key in competition chart dicts.

    These tests reproduce the bug described in Analysis 020 and guard against
    regression after the fix in Plan 020.

    Note: tests use `object.__new__` to bypass TotalLoss / DiversityTrainer __init__,
    which requires heavy external dependencies. The make_total_loss fixture is NOT used
    here because it has a pre-existing issue where it attempts to patch MeshStatistics
    at module level but that symbol is only imported locally inside a method.
    """

    def test_total_loss_competition_dicts_exclude_total_loss(self) -> None:
        """TotalLoss._handle_telemetry must not pass total_loss into competition chart dicts.

        The four dicts (raw_floats, normalized_floats, weighted_floats, optimizer_floats)
        passed to _log_loss_competition_charts must all exclude any key named 'total_loss'.
        """
        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss  # noqa: PLC0415

        tl = object.__new__(TotalLoss)
        tl._use_wandb = True
        tl._use_renderer = False
        tl._use_cse = False
        tl._last_cruising_speed = None
        tl._memory_diagnostics = False
        tl._last_weighted_losses = {
            "gaot_loss": torch.tensor(2.0),
            "solar_v3B": torch.tensor(100.0),
        }
        tl._objective_losses = {}
        tl._objective_signs = {}
        tl._use_alm = False
        mock_norm = MagicMock()
        mock_norm.get_loss_scale.return_value = 1.0
        mock_norm.get_telemetry.return_value = {}
        tl._norm_manager = mock_norm

        # loss_log contains a raw_losses/total_loss that must NOT reach competition dicts.
        loss_log: dict[str, float] = {
            "raw_losses/gaot_loss": 2.0,
            "raw_losses/solar_v3B": 100.0,
            "raw_losses/total_loss": 102.0,  # contaminant — must NOT reach pie charts
        }
        total_loss_tensor = torch.tensor(102.0)

        captured: dict[str, dict] = {}

        def _spy(
            raw_f: dict,
            norm_f: dict,
            weighted_f: dict,
            opt_f: dict,
            _epoch: int,
        ) -> dict:
            captured["raw"] = raw_f
            captured["normalized"] = norm_f
            captured["weighted"] = weighted_f
            captured["optimizer"] = opt_f
            return {}

        with patch.object(tl, "_log_loss_competition_charts", side_effect=_spy):
            tl._handle_telemetry(
                total_loss=total_loss_tensor,
                loss_log=loss_log,
                epoch_name=0,
                model=MagicMock(),
                device=torch.device("cpu"),
            )

        assert captured, "Spy was never called — _log_loss_competition_charts not reached"
        assert "total_loss" not in captured["raw"], "raw_floats must not contain total_loss"
        assert "total_loss" not in captured["normalized"], "normalized_floats must not contain total_loss"
        assert "total_loss" not in captured["weighted"], "weighted_floats must not contain total_loss"
        assert "total_loss" not in captured["optimizer"], "optimizer_floats must not contain total_loss"

    def test_diversity_trainer_competition_dicts_exclude_total_loss(self) -> None:
        """DiversityTrainer._log must not pass total_loss into competition chart dicts.

        avg_log_data is built from TotalLossV2.weight() log_data which contains
        normalized_losses/total_loss and optimizer_losses/total_loss. The _log()
        method must exclude these before passing dicts to _log_competition_charts.
        """
        from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import DiversityTrainer  # noqa: PLC0415

        dt = object.__new__(DiversityTrainer)
        dt._use_wandb = True

        # avg_log_data mimics the dict from TotalLossV2.weight() — contains total_loss keys.
        avg_log_data: dict[str, float] = {
            "raw_losses/gaot_loss": 2.0,
            "raw_losses/solar_v3B": 100.0,
            "raw_losses/total_loss": 102.0,  # contaminant
            "normalized_losses/gaot_loss": 0.5,
            "normalized_losses/solar_v3B": 0.8,
            "normalized_losses/total_loss": 1.3,  # contaminant
            "weighted_losses/gaot_loss": 0.1,
            "weighted_losses/solar_v3B": 0.4,
            "optimizer_losses/gaot_loss": 0.1,
            "optimizer_losses/solar_v3B": 0.4,
            "optimizer_losses/total_loss": 0.5,  # contaminant
        }

        captured: dict[str, dict] = {}

        def _spy(raw_f: dict, norm_f: dict, weighted_f: dict, opt_f: dict, _epoch: int) -> dict:
            captured["raw"] = raw_f
            captured["normalized"] = norm_f
            captured["weighted"] = weighted_f
            captured["optimizer"] = opt_f
            return {}

        mock_alm = MagicMock()
        mock_alm.get_telemetry.return_value = {}
        mock_norm = MagicMock()
        mock_norm.get_telemetry.return_value = {}
        dt._alm_manager = mock_alm
        dt._norm_manager = mock_norm

        dm_module = "subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager"
        with patch(f"{dm_module}.wandb") as mock_wandb:
            mock_wandb.run = MagicMock()
            with patch.object(dt, "_log_competition_charts", side_effect=_spy):
                dt._log(avg_log_data, {}, {}, epoch=0)

        assert captured, "Spy was never called — _log_competition_charts not reached"
        assert "total_loss" not in captured["raw"], "raw_floats must not contain total_loss"
        assert "total_loss" not in captured["normalized"], "normalized_floats must not contain total_loss"
        assert "total_loss" not in captured["weighted"], "weighted_floats must not contain total_loss"
        assert "total_loss" not in captured["optimizer"], "optimizer_floats must not contain total_loss"
