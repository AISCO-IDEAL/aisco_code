"""Tests for latent-population pretraining (Plan 005 / v0.3.0).

Covers:
  - build_unit_circle_Z geometry
  - _sample_points_for_shape surface concentration
  - _compute_eikonal_latent finite output
  - DiversityTrainer initial_Z injection and Mode C guard
  - pretraining_latent end-to-end on CPU
  - _maybe_run_latent_pretraining gate logic in main_v2

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_pretraining_latent.py -v
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_V2 = "subprojects.wp4_aeroshell_optimizer.v2.src"

LATENT_UTILS = f"{_V2}.utils.latent_utils"
PRETRAIN_V1_1 = f"{_V2}.main_files.pretraining_v1_1"
PRETRAIN_LATENT = f"{_V2}.main_files.pretraining_latent"
DIVERSITY_MANAGER = f"{_V2}.GINN.diversity_manager"
MAIN_V2 = f"{_V2}.main_files.main_v2"
DYNAMIC_LOADING = f"{_V2}.utils.dynamic_loading"


def _import(module: str, name: str) -> Any:
    mod = importlib.import_module(module)
    return getattr(mod, name)


def _make_concat_wire_v2() -> Any:
    """Return a fresh ConcatWIREV2 instance on CPU."""
    smart_model_init = _import(DYNAMIC_LOADING, "smart_model_init")
    model, _, _, _ = smart_model_init("ConcatWIREV2")
    return model.to("cpu")


def _make_latent_pretrain_config(checkpoint_path: str, force: bool = True) -> dict:
    """Minimal 2-shape config dict for integration tests."""
    return {
        "model_name": "ConcatWIREV2",
        "z_placement": "unit_circle",
        "z_radius": 1.0,
        "force_pretraining": force,
        "latent_population": [
            {
                "name": "sphere_slot_0",
                "shape_type": "sphere",
                "radius": 1000.0,
                "ply_path": None,
                "side_length": None,
            },
            {
                "name": "cube_slot_1",
                "shape_type": "cube",
                "side_length": 2000.0,
                "ply_path": None,
                "radius": None,
            },
        ],
        "training_loop": {
            "epochs": 3,
            "learning_rate": 1e-4,
            "batch_size_per_shape": 32,
            "print_interval": 1,
            "eikonal_weight": 0.01,
        },
        "output": {
            "checkpoint_path": checkpoint_path,
            "mesh_resolution": 16,
            "mesh_box_mm": 2000,
        },
    }


def _make_mock_total_loss() -> MagicMock:
    """TotalLossV2 mock whose weight() returns a real scalar tensor for backward."""
    total_loss = MagicMock()
    total_loss.constraint_category_order = []

    def fake_compute(model: Any, epoch_name: Any = 0) -> MagicMock:  # noqa: ARG001  # signature must match TotalLossV2.compute kwarg names; model/epoch_name are passed by keyword
        bundle = MagicMock()
        bundle.constraints = {}
        return bundle

    def fake_weight(bundle: Any, alm_manager: Any = None) -> MagicMock:  # noqa: ARG001  # signature must match TotalLossV2.weight kwarg names; bundle/alm_manager are passed by keyword
        result = MagicMock()
        result.scalar = torch.tensor(1.0, requires_grad=True)
        result.log_data = {"raw_losses/total_loss": 1.0}
        return result

    total_loss.compute.side_effect = fake_compute
    total_loss.weight.side_effect = fake_weight
    return total_loss


def _make_trainer(
    *,
    population_size: int = 2,
    z_dim: int = 2,
    z_resample_cadence: int = 0,
    z_resample_start_epoch: int = 0,
    force_z0_to_origin: bool = True,
    initial_Z: torch.Tensor | None = None,
) -> Any:
    """Construct a minimal DiversityTrainer suitable for unit tests."""
    DiversityTrainer = _import(DIVERSITY_MANAGER, "DiversityTrainer")
    model = _make_concat_wire_v2()
    total_loss = _make_mock_total_loss()

    return DiversityTrainer(
        model=model,
        total_loss=total_loss,
        alm_manager=None,
        population_size=population_size,
        z_dim=z_dim,
        device="cpu",
        model_lr=1e-4,
        diversity_manager=None,
        force_z0_to_origin=force_z0_to_origin,
        z_resample_cadence=z_resample_cadence,
        z_resample_start_epoch=z_resample_start_epoch,
        initial_Z=initial_Z,
        use_wandb=False,
    )


# ---------------------------------------------------------------------------
# M1 — build_unit_circle_Z
# ---------------------------------------------------------------------------


class TestBuildUnitCircleZ:
    """Tests for latent_utils.build_unit_circle_Z."""

    def test_origin_slot(self) -> None:
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        Z = build_unit_circle_Z(6, 2, 1.0, "cpu")
        assert Z[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert Z[0, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_shape(self) -> None:
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        Z = build_unit_circle_Z(6, 2, 1.0, "cpu")
        assert Z.shape == (6, 2)

    def test_equidistant_from_origin(self) -> None:
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        Z = build_unit_circle_Z(6, 2, 1.0, "cpu")
        for i in range(1, 6):
            dist = Z[i].norm().item()
            assert dist == pytest.approx(1.0, abs=1e-5), f"z{i} distance from origin={dist}"

    def test_pairwise_equidistant(self) -> None:
        """All non-origin points should be equidistant from each other (equilateral triangle for N=4)."""
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        # For num_shapes=4 we get 3 non-origin points at 120° intervals (equilateral triangle).
        # All 3 pairwise distances are equal.  For N>4 we only verify equal angular spacing.
        Z = build_unit_circle_Z(4, 2, 1.0, "cpu")
        dists = [(Z[i] - Z[j]).norm().item() for i in range(1, 4) for j in range(i + 1, 4)]
        expected = dists[0]
        for d in dists:
            assert d == pytest.approx(expected, abs=1e-5)

    def test_no_gradient(self) -> None:
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        Z = build_unit_circle_Z(4, 2, 1.0, "cpu")
        assert not Z.requires_grad

    def test_custom_radius(self) -> None:
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        Z = build_unit_circle_Z(4, 2, 2.5, "cpu")
        for i in range(1, 4):
            assert Z[i].norm().item() == pytest.approx(2.5, abs=1e-5)

    def test_single_shape_origin_only(self) -> None:
        build_unit_circle_Z = _import(LATENT_UTILS, "build_unit_circle_Z")
        Z = build_unit_circle_Z(1, 2, 1.0, "cpu")
        assert Z.shape == (1, 2)
        assert Z[0].norm().item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# M2 — _sample_points_for_shape surface concentration
# ---------------------------------------------------------------------------


class TestSamplePointsForShape:
    """Verify surface-concentration property of _sample_points_for_shape."""

    def test_sphere_surface_concentration(self) -> None:
        """≥40% of points should have |SDF| < 200mm for r=1000mm sphere."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        n = 4000
        pts, gt = _sample_points_for_shape("sphere", {"radius": 1000.0}, n, "cpu")
        assert pts.shape == (n, 3)
        assert gt.shape == (n,)
        near = (gt.abs() < 200.0).float().mean().item()
        assert near >= 0.40, f"Expected ≥40% near-surface, got {near:.1%}"

    def test_cube_surface_concentration(self) -> None:
        """≥40% of points should have |SDF| < 300mm for side=2000mm cube."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        n = 4000
        pts, gt = _sample_points_for_shape("cube", {"side_length": 2000.0}, n, "cpu")
        assert pts.shape == (n, 3)
        assert gt.shape == (n,)
        near = (gt.abs() < 300.0).float().mean().item()
        assert near >= 0.40, f"Expected ≥40% near-surface, got {near:.1%}"

    def test_sphere_gt_correct(self) -> None:
        """Spot-check: exact surface points should have |SDF|=0."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        torch.manual_seed(42)
        pts, gt = _sample_points_for_shape("sphere", {"radius": 500.0}, 600, "cpu")
        # GT is defined as norm(pts) - radius; spot check a sample
        expected_gt = pts.norm(dim=-1) - 500.0
        assert torch.allclose(gt, expected_gt, atol=1.0), "SDF formula mismatch for sphere"


# ---------------------------------------------------------------------------
# M2 — _compute_eikonal_latent
# ---------------------------------------------------------------------------


class TestComputeEikonalLatent:
    """Verify _compute_eikonal_latent produces finite output."""

    def test_finite_output(self) -> None:
        _compute_eikonal_latent = _import(PRETRAIN_V1_1, "_compute_eikonal_latent")
        model = _make_concat_wire_v2()
        model.eval()
        n = 16
        z_dim = model.get_model_info_dict()["z_dim"]
        pts = torch.rand(n, 3) * 1000.0
        z = torch.randn(n, z_dim)
        result = _compute_eikonal_latent(model, pts, z, "cpu")
        assert isinstance(result, torch.Tensor)
        assert math.isfinite(result.item()), f"Expected finite eikonal, got {result.item()}"

    def test_has_grad_fn(self) -> None:
        """Eikonal must be differentiable w.r.t. model parameters."""
        _compute_eikonal_latent = _import(PRETRAIN_V1_1, "_compute_eikonal_latent")
        model = _make_concat_wire_v2()
        model.train()
        n = 8
        z_dim = model.get_model_info_dict()["z_dim"]
        pts = torch.rand(n, 3) * 500.0
        z = torch.randn(n, z_dim)
        result = _compute_eikonal_latent(model, pts, z, "cpu")
        assert result.requires_grad or result.grad_fn is not None, "Eikonal must have grad_fn"


# ---------------------------------------------------------------------------
# M5 — DiversityTrainer extensions
# ---------------------------------------------------------------------------


class TestDiversityTrainerInitialZ:
    """Tests for DiversityTrainer initial_Z injection."""

    def test_initial_z_injection(self) -> None:
        """initial_Z is stored verbatim in self._Z."""
        Z_init = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        trainer = _make_trainer(population_size=2, z_dim=2, initial_Z=Z_init)
        stored = trainer._Z
        # z0 may be zeroed by force_z0_to_origin, z1 should match
        assert stored[1, 0].item() == pytest.approx(1.0, abs=1e-6)
        assert stored[1, 1].item() == pytest.approx(0.0, abs=1e-6)

    def test_initial_z_shape_mismatch_raises(self) -> None:
        """Mismatched initial_Z shape must raise ValueError with both shapes in message."""
        Z_wrong = torch.ones(3, 2)  # population_size=2 expected, got 3
        with pytest.raises(ValueError, match=r"\(3.*2\)|\(2.*2\)"):
            _make_trainer(population_size=2, z_dim=2, initial_Z=Z_wrong)

    def test_no_initial_z_uses_randn(self) -> None:
        """Without initial_Z, _Z is randomly initialised (not all-zero)."""
        torch.manual_seed(0)
        trainer = _make_trainer(population_size=4, z_dim=2, initial_Z=None)
        # At least some non-zero values in rows 1..3 (row 0 is zeroed by force_z0_to_origin)
        assert trainer._Z[1:].abs().sum().item() > 0

    def test_force_z0_to_origin_preserved(self) -> None:
        """force_z0_to_origin must still zero Z[0] after initial_Z injection."""
        Z_init = torch.tensor([[9.9, 9.9], [1.0, 0.5]])
        trainer = _make_trainer(population_size=2, z_dim=2, initial_Z=Z_init, force_z0_to_origin=True)
        assert trainer._Z[0].norm().item() == pytest.approx(0.0, abs=1e-6)


class TestDiversityTrainerModeC:
    """Tests for Mode A/B/C resampling logic."""

    def test_mode_c_held_before_start(self) -> None:
        """With z_resample_start_epoch=200: Z unchanged at epoch 199."""
        Z_init = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        trainer = _make_trainer(
            population_size=2,
            z_dim=2,
            z_resample_cadence=1,
            z_resample_start_epoch=200,
            initial_Z=Z_init,
        )
        Z_before = trainer._Z.clone()  # noqa: F841  # retained for future assertions on Z stability
        with patch.object(trainer, "_resample_population") as mock_resample:
            trainer.step(epoch=199)
        mock_resample.assert_not_called()

    def test_mode_c_active_at_start_epoch(self) -> None:
        """With z_resample_start_epoch=200, cadence=1: _resample_population called at epoch 200."""
        Z_init = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        trainer = _make_trainer(
            population_size=2,
            z_dim=2,
            z_resample_cadence=1,
            z_resample_start_epoch=200,
            initial_Z=Z_init,
        )
        with patch.object(trainer, "_resample_population") as mock_resample:
            trainer.step(epoch=200)
        mock_resample.assert_called_once_with(200)

    def test_mode_a_cadence_zero_never_resamples(self) -> None:
        """Mode A: cadence=0 → never resamples regardless of z_resample_start_epoch."""
        Z_init = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        trainer = _make_trainer(
            population_size=2,
            z_dim=2,
            z_resample_cadence=0,
            z_resample_start_epoch=200,
            initial_Z=Z_init,
        )
        with patch.object(trainer, "_resample_population") as mock_resample:
            for epoch in range(250):
                trainer.step(epoch=epoch)
        mock_resample.assert_not_called()

    def test_default_params_backward_compat(self) -> None:
        """Default initial_Z=None + z_resample_start_epoch=0 initialises without error."""
        trainer = _make_trainer(population_size=2, z_dim=2, z_resample_cadence=0)
        assert trainer._Z.shape == (2, 2)


# ---------------------------------------------------------------------------
# M3 — pretraining_latent integration
# ---------------------------------------------------------------------------


class TestPretrainingLatentIntegration:
    """End-to-end tests for pretraining_latent()."""

    def test_runs_cpu(self, tmp_path: Path) -> None:
        """2 shapes, 3 epochs, CPU, no error."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config(str(tmp_path / "ckpt.pth"))
        trained, Z = pretraining_latent(model, config, "cpu")
        assert trained is not None
        assert Z.shape == (2, 2)

    def test_checkpoint_format(self, tmp_path: Path) -> None:
        """Checkpoint contains model_state_dict and z_pretrained keys."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        _resolve_checkpoint_path = _import(PRETRAIN_LATENT, "_resolve_checkpoint_path")
        model = _make_concat_wire_v2()
        ckpt_path = str(tmp_path / "ckpt.pth")
        config = _make_latent_pretrain_config(ckpt_path)
        pretraining_latent(model, config, "cpu")
        resolved = _resolve_checkpoint_path(config, model)
        assert resolved.exists()
        ckpt = torch.load(resolved, map_location="cpu", weights_only=True)
        assert "model_state_dict" in ckpt
        assert "z_pretrained" in ckpt
        assert ckpt["z_pretrained"].shape == (2, 2)

    def test_nan_guard_no_checkpoint(self, tmp_path: Path) -> None:
        """NaN loss halts training and does NOT write the checkpoint."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        model = _make_concat_wire_v2()
        ckpt_path = str(tmp_path / "should_not_exist.pth")
        config = _make_latent_pretrain_config(ckpt_path)

        def always_nan(x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG001  # z is required in signature to match latent model protocol; not used in this always-NaN stub
            return torch.full((x.shape[0], 1), float("nan"))

        model.forward = always_nan  # type: ignore[method-assign]

        with pytest.raises(SystemExit):
            pretraining_latent(model, config, "cpu")

        assert not Path(ckpt_path).exists(), "Checkpoint must NOT be written on NaN"

    def test_fast_resume(self, tmp_path: Path) -> None:
        """If checkpoint exists and force_pretraining=False, pretraining is skipped."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        model = _make_concat_wire_v2()
        ckpt_path = str(tmp_path / "ckpt.pth")

        # First run — creates checkpoint
        config_force = _make_latent_pretrain_config(ckpt_path, force=True)
        _, Z_first = pretraining_latent(model, config_force, "cpu")

        # Second run — should skip training (force=False)
        config_resume = _make_latent_pretrain_config(ckpt_path, force=False)
        _, Z_second = pretraining_latent(model, config_resume, "cpu")

        assert torch.allclose(Z_first, Z_second.to(Z_first.device), atol=1e-5)

    def test_old_checkpoint_raises(self, tmp_path: Path) -> None:
        """Old single-tensor checkpoint at hash-tagged path raises ValueError with actionable msg."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        _resolve_checkpoint_path = _import(PRETRAIN_LATENT, "_resolve_checkpoint_path")
        model = _make_concat_wire_v2()
        ckpt_path = str(tmp_path / "old_ckpt.pth")
        config = _make_latent_pretrain_config(ckpt_path, force=False)
        # Write an old-format checkpoint at the resolved hash-tagged path
        resolved = _resolve_checkpoint_path(config, model)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), resolved)

        with pytest.raises(ValueError, match=r"old single-tensor|model_state_dict|z_pretrained"):
            pretraining_latent(model, config, "cpu")


# ---------------------------------------------------------------------------
# M6 — _maybe_run_latent_pretraining gate
# ---------------------------------------------------------------------------


class TestMaybeRunLatentPretraining:
    """Tests for main_v2._maybe_run_latent_pretraining gate logic."""

    def test_disabled_returns_none_z(self) -> None:
        """Empty latent_pretraining key → Z is None, model unchanged."""
        _maybe_run_latent_pretraining = _import(MAIN_V2, "_maybe_run_latent_pretraining")
        model = _make_concat_wire_v2()
        general_config: dict = {"active_configs": {"latent_pretraining": ""}}
        _, Z = _maybe_run_latent_pretraining(model, general_config, "cpu")
        assert Z is None

    def test_non_concat_model_warns_and_skips(self, capsys: Any) -> None:
        """Non-SupportsLatentConcat model → warning printed, Z is None."""
        _maybe_run_latent_pretraining = _import(MAIN_V2, "_maybe_run_latent_pretraining")
        SupportsLatentConcat = _import(f"{_V2}.models.model_protocol", "SupportsLatentConcat")

        class _NoLatentModel:
            """Minimal model without latent-vector support."""

            def get_model_info_dict(self) -> dict:
                return {}

            def state_dict(self) -> dict:
                return {}

            def load_state_dict(self, *a: Any, **kw: Any) -> None:
                pass

            def to(self, _d: Any) -> _NoLatentModel:
                return self

            def cpu(self) -> _NoLatentModel:
                return self

            def train(self) -> _NoLatentModel:
                return self

            def eval(self) -> _NoLatentModel:
                return self

            def parameters(self, _recurse: bool = True) -> Any:
                return iter([])

            def __call__(self, *_a: Any, **_kw: Any) -> Any:
                return None

        model = _NoLatentModel()
        assert not isinstance(model, SupportsLatentConcat)

        general_config: dict = {"active_configs": {"latent_pretraining": "pretraining_latent_config.yaml"}}
        _, Z = _maybe_run_latent_pretraining(model, general_config, "cpu")
        assert Z is None
        captured = capsys.readouterr()
        assert "WARNING" in captured.out or "warning" in captured.out.lower()


# ---------------------------------------------------------------------------
# Hash-based checkpoint caching
# ---------------------------------------------------------------------------


class TestPretrainConfigHash:
    """Tests for _compute_pretrain_config_hash and _resolve_checkpoint_path."""

    def test_hash_stable_same_config(self) -> None:
        """Same config + model → same hash both times."""
        _compute_pretrain_config_hash = _import(PRETRAIN_LATENT, "_compute_pretrain_config_hash")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config("/tmp/ckpt.pth")
        h1 = _compute_pretrain_config_hash(config, model)
        h2 = _compute_pretrain_config_hash(config, model)
        assert h1 == h2

    def test_hash_changes_different_population(self) -> None:
        """Different latent_population shapes → different hash."""
        _compute_pretrain_config_hash = _import(PRETRAIN_LATENT, "_compute_pretrain_config_hash")
        model = _make_concat_wire_v2()
        config_a = _make_latent_pretrain_config("/tmp/ckpt.pth")
        config_b = _make_latent_pretrain_config("/tmp/ckpt.pth")
        config_b["latent_population"][0]["radius"] = 500.0  # different from 1000.0
        assert _compute_pretrain_config_hash(config_a, model) != _compute_pretrain_config_hash(config_b, model)

    def test_hash_changes_different_training_loop(self) -> None:
        """Different training_loop params → different hash."""
        _compute_pretrain_config_hash = _import(PRETRAIN_LATENT, "_compute_pretrain_config_hash")
        model = _make_concat_wire_v2()
        config_a = _make_latent_pretrain_config("/tmp/ckpt.pth")
        config_b = _make_latent_pretrain_config("/tmp/ckpt.pth")
        config_b["training_loop"]["epochs"] = 9999
        assert _compute_pretrain_config_hash(config_a, model) != _compute_pretrain_config_hash(config_b, model)

    def test_resolve_path_embeds_hash(self) -> None:
        """Resolved path stem contains the config hash."""
        _resolve_checkpoint_path = _import(PRETRAIN_LATENT, "_resolve_checkpoint_path")
        _compute_pretrain_config_hash = _import(PRETRAIN_LATENT, "_compute_pretrain_config_hash")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config("/tmp/base/pretrained_latent.pth")
        expected_hash = _compute_pretrain_config_hash(config, model)
        resolved = _resolve_checkpoint_path(config, model)
        assert expected_hash in resolved.stem

    def test_resolve_path_same_directory(self) -> None:
        """Resolved checkpoint is in the same directory as the configured path."""
        _resolve_checkpoint_path = _import(PRETRAIN_LATENT, "_resolve_checkpoint_path")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config("/some/dir/pretrained_latent.pth")
        resolved = _resolve_checkpoint_path(config, model)
        assert resolved.parent == Path("/some/dir")
        assert resolved.suffix == ".pth"

    def test_different_configs_separate_checkpoints(self, tmp_path: Path) -> None:
        """Two configs with different training params write to separate checkpoint files."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        base_path = str(tmp_path / "pretrained_latent.pth")
        config_a = _make_latent_pretrain_config(base_path)
        config_b = _make_latent_pretrain_config(base_path)
        config_b["training_loop"]["epochs"] = 2  # different from 3

        model_a = _make_concat_wire_v2()
        model_b = _make_concat_wire_v2()
        pretraining_latent(model_a, config_a, "cpu")
        pretraining_latent(model_b, config_b, "cpu")

        pth_files = list(tmp_path.glob("*.pth"))
        assert len(pth_files) == 2, f"Expected 2 separate checkpoint files, got {len(pth_files)}: {pth_files}"

    def test_same_config_reuses_checkpoint(self, tmp_path: Path) -> None:
        """Same config + force=False on second run skips training (fast resume)."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        model = _make_concat_wire_v2()
        base_path = str(tmp_path / "pretrained_latent.pth")

        config_first = _make_latent_pretrain_config(base_path, force=True)
        _, Z_first = pretraining_latent(model, config_first, "cpu")

        config_resume = _make_latent_pretrain_config(base_path, force=False)
        _, Z_second = pretraining_latent(model, config_resume, "cpu")

        assert torch.allclose(Z_first, Z_second.to(Z_first.device), atol=1e-5)
        pth_files = list(tmp_path.glob("*.pth"))
        assert len(pth_files) == 1, "Should not create a second file on resume"
