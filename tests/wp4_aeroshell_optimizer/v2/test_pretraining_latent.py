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

import numpy as np
import pytest
import torch
import trimesh

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
            "logging_interval": 1,
            "eikonal_weight": 0.01,
        },
        "output": {
            "checkpoint_path": checkpoint_path,
            "mesh_resolution": 16,
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
        radius = 500.0
        pts, gt = _sample_points_for_shape("sphere", {"radius": radius}, 600, "cpu")
        # Sphere is now centered at (0, 0, 100+radius) and grounded at Z=100
        sphere_center = torch.tensor([0.0, 0.0, 100.0 + radius])
        expected_gt = (pts - sphere_center).norm(dim=-1) - radius
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
# M1 — _load_ply_tensors watertightness guard
# ---------------------------------------------------------------------------


class TestLoadPlyTensorsWatertight:
    """Verify that _load_ply_tensors raises ValueError for non-watertight meshes."""

    def test_raises_if_not_watertight(self, tmp_path: Path) -> None:
        """A non-watertight PLY must raise ValueError before any vertex processing."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")

        # Build a minimal open mesh (non-watertight: two disconnected triangles, open boundary)
        verts = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)  # 2 disconnected tris = open mesh
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        assert not mesh.is_watertight, "Fixture mesh must not be watertight"

        ply_path = tmp_path / "open_mesh.ply"
        mesh.export(str(ply_path))

        with pytest.raises(ValueError, match="watertight"):
            _load_ply_tensors(str(ply_path), ply_scale=1.0)

    def test_watertight_mesh_loads_without_error(self, tmp_path: Path) -> None:
        """A watertight PLY must load successfully and return verts, faces, extent."""
        _load_ply_tensors = _import(PRETRAIN_LATENT, "_load_ply_tensors")

        sphere = trimesh.creation.icosphere(radius=500.0)
        assert sphere.is_watertight, "Fixture sphere must be watertight"

        ply_path = tmp_path / "sphere.ply"
        sphere.export(str(ply_path))

        verts, faces, extent = _load_ply_tensors(str(ply_path), ply_scale=1.0)
        assert verts.shape[0] > 0
        assert faces.shape[0] > 0
        assert extent > 0.0


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
        """Resolved path's parent folder and filename both contain config hash."""
        _resolve_checkpoint_path = _import(PRETRAIN_LATENT, "_resolve_checkpoint_path")
        _compute_pretrain_config_hash = _import(PRETRAIN_LATENT, "_compute_pretrain_config_hash")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config("/tmp/base/pretrained_latent.pth")
        expected_hash = _compute_pretrain_config_hash(config, model)
        resolved = _resolve_checkpoint_path(config, model)
        # Structure: /tmp/base/<hash>/pretrained_latent_<hash>.pth
        assert resolved.parent.name == expected_hash
        assert expected_hash in resolved.stem
        assert resolved.suffix == ".pth"

    def test_resolve_path_same_directory(self) -> None:
        """Resolved checkpoint is under the configured base directory."""
        _resolve_checkpoint_path = _import(PRETRAIN_LATENT, "_resolve_checkpoint_path")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config("/some/dir/pretrained_latent.pth")
        resolved = _resolve_checkpoint_path(config, model)
        # Structure: /some/dir/<hash>/pretrained_latent_<hash>.pth
        assert resolved.parent.parent == Path("/some/dir")
        assert resolved.suffix == ".pth"
        assert "pretrained_latent_" in resolved.name

    def test_different_configs_separate_checkpoints(self, tmp_path: Path) -> None:
        """Two configs with different training params write to separate checkpoint folders."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        base_path = str(tmp_path / "pretrained_latent.pth")
        config_a = _make_latent_pretrain_config(base_path)
        config_b = _make_latent_pretrain_config(base_path)
        config_b["training_loop"]["epochs"] = 2  # different from 3

        model_a = _make_concat_wire_v2()
        model_b = _make_concat_wire_v2()
        pretraining_latent(model_a, config_a, "cpu")
        pretraining_latent(model_b, config_b, "cpu")

        # New structure: tmp_path/<hash>/pretrained_latent.pth
        pth_files = list(tmp_path.glob("**/pretrained_latent_*.pth"))
        assert len(pth_files) == 2, f"Expected 2 separate checkpoint files in hash folders, got {len(pth_files)}: {pth_files}"

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
        # New structure: tmp_path/<hash>/pretrained_latent.pth
        pth_files = list(tmp_path.glob("**/pretrained_latent_*.pth"))
        assert len(pth_files) == 1, f"Should not create a second file on resume, got {len(pth_files)}: {pth_files}"


# ---------------------------------------------------------------------------
# M3 — _load_or_build_surface_cache SDF cache
# ---------------------------------------------------------------------------


class TestSDFCache:
    """Tests for _load_or_build_surface_cache (Plan 013 / M3)."""

    def _make_sphere_ply(self, tmp_path: Path, radius: float = 500.0) -> Path:
        """Export a watertight icosphere as PLY; return path."""
        sphere = trimesh.creation.icosphere(radius=radius)
        ply_path = tmp_path / "sphere.ply"
        sphere.export(str(ply_path))
        return ply_path

    def _make_ply_shape_params(self, ply_path: Path) -> dict:
        mesh = trimesh.load(str(ply_path))
        verts_np = np.array(mesh.vertices, dtype="float32")
        faces_np = np.array(mesh.faces, dtype="int64")
        verts_t = torch.tensor(verts_np)
        faces_t = torch.tensor(faces_np)
        extents = verts_np.max(axis=0) - verts_np.min(axis=0)
        return {
            "ply_verts": verts_t,
            "ply_faces": faces_t,
            "bbox_extent": float(np.linalg.norm(extents)),
            "surface_noise_sigma_factor": 0.005,
        }

    def test_cache_miss_builds_and_writes(self, tmp_path: Path) -> None:
        """On first call (cache miss), cache file is written to cache_dir."""
        _load_or_build_surface_cache = _import(PRETRAIN_LATENT, "_load_or_build_surface_cache")
        ply_path = self._make_sphere_ply(tmp_path)
        shape_params = self._make_ply_shape_params(ply_path)
        cache_dir = tmp_path / "sdf_cache"
        cache_dir.mkdir()

        pts, gt = _load_or_build_surface_cache(
            ply_path=str(ply_path),
            ply_scale=1.0,
            shape_params=shape_params,
            batch_size=100,
            n_dom=50,
            device="cpu",
            cache_dir=cache_dir,
        )
        assert pts.shape == (50, 3), f"Expected (50, 3), got {pts.shape}"
        assert gt.shape == (50,), f"Expected (50,), got {gt.shape}"

        cache_files = list(cache_dir.glob("*.pt"))
        assert len(cache_files) == 1, f"Expected 1 cache file, got {len(cache_files)}"

    def test_cache_hit_loads_same_pts(self, tmp_path: Path) -> None:
        """On second call (cache hit), the same pts and gt are returned without re-sampling."""
        _load_or_build_surface_cache = _import(PRETRAIN_LATENT, "_load_or_build_surface_cache")
        ply_path = self._make_sphere_ply(tmp_path)
        shape_params = self._make_ply_shape_params(ply_path)
        cache_dir = tmp_path / "sdf_cache"
        cache_dir.mkdir()

        kwargs: dict = {
            "ply_path": str(ply_path),
            "ply_scale": 1.0,
            "shape_params": shape_params,
            "batch_size": 100,
            "n_dom": 50,
            "device": "cpu",
            "cache_dir": cache_dir,
        }
        pts_first, gt_first = _load_or_build_surface_cache(**kwargs)
        pts_second, gt_second = _load_or_build_surface_cache(**kwargs)

        assert torch.allclose(pts_first, pts_second), "Cache hit must return identical pts"
        assert torch.allclose(gt_first, gt_second), "Cache hit must return identical gt"
        # Still only one cache file (no duplicate created on hit)
        assert len(list(cache_dir.glob("*.pt"))) == 1

    def test_param_change_creates_new_cache_file(self, tmp_path: Path) -> None:
        """Changing n_dom → new hash → new cache file (old file untouched)."""
        _load_or_build_surface_cache = _import(PRETRAIN_LATENT, "_load_or_build_surface_cache")
        ply_path = self._make_sphere_ply(tmp_path)
        shape_params = self._make_ply_shape_params(ply_path)
        cache_dir = tmp_path / "sdf_cache"
        cache_dir.mkdir()

        _load_or_build_surface_cache(
            ply_path=str(ply_path),
            ply_scale=1.0,
            shape_params=shape_params,
            batch_size=100,
            n_dom=50,
            device="cpu",
            cache_dir=cache_dir,
        )
        _load_or_build_surface_cache(
            ply_path=str(ply_path),
            ply_scale=1.0,
            shape_params=shape_params,
            batch_size=100,
            n_dom=30,  # changed n_dom → different n_surf → new hash
            device="cpu",
            cache_dir=cache_dir,
        )
        cache_files = list(cache_dir.glob("*.pt"))
        assert len(cache_files) == 2, f"Expected 2 separate cache files, got {len(cache_files)}"

    def test_none_cache_dir_skips_cache(self, tmp_path: Path) -> None:
        """cache_dir=None must return pts/gt without writing any file."""
        _load_or_build_surface_cache = _import(PRETRAIN_LATENT, "_load_or_build_surface_cache")
        ply_path = self._make_sphere_ply(tmp_path)
        shape_params = self._make_ply_shape_params(ply_path)

        pts, gt = _load_or_build_surface_cache(
            ply_path=str(ply_path),
            ply_scale=1.0,
            shape_params=shape_params,
            batch_size=80,
            n_dom=40,
            device="cpu",
            cache_dir=None,
        )
        assert pts.shape[0] == 40
        assert gt.shape[0] == 40
        # No files written anywhere in tmp_path (cache was disabled)
        pt_files = list(tmp_path.glob("**/*.pt"))
        assert len(pt_files) == 0, f"Expected no .pt files, found: {pt_files}"

    def test_cache_hit_moves_tensors_to_requested_device(self, tmp_path: Path) -> None:
        """Cache hit must move loaded CPU tensors onto requested device."""
        _load_or_build_surface_cache = _import(PRETRAIN_LATENT, "_load_or_build_surface_cache")
        ply_path = self._make_sphere_ply(tmp_path)
        shape_params = self._make_ply_shape_params(ply_path)
        cache_dir = tmp_path / "sdf_cache"
        cache_dir.mkdir()

        # First call builds cache on CPU.
        _load_or_build_surface_cache(
            ply_path=str(ply_path),
            ply_scale=1.0,
            shape_params=shape_params,
            batch_size=100,
            n_dom=50,
            device="cpu",
            cache_dir=cache_dir,
        )

        # Second call must load from cache and honor requested device.
        # Use "meta" to validate device transfer without requiring CUDA hardware.
        pts_meta, gt_meta = _load_or_build_surface_cache(
            ply_path=str(ply_path),
            ply_scale=1.0,
            shape_params=shape_params,
            batch_size=100,
            n_dom=50,
            device="meta",
            cache_dir=cache_dir,
        )

        assert pts_meta.device.type == "meta"
        assert gt_meta.device.type == "meta"

    def test_content_change_busts_cache(self, tmp_path: Path) -> None:
        """PLY content change (same path, different bytes) must produce a cache miss.

        This is the TDD red-phase test for Plan 028 M1 (content hash in cache key).
        After M1 is implemented, the PLY content hash is included in the cache key
        so that a same-name file with different bytes creates a new .pt file.
        """
        _load_or_build_surface_cache = _import(PRETRAIN_LATENT, "_load_or_build_surface_cache")
        ply_path = self._make_sphere_ply(tmp_path, radius=500.0)
        shape_params = self._make_ply_shape_params(ply_path)
        cache_dir = tmp_path / "sdf_cache"
        cache_dir.mkdir()

        kwargs: dict = {
            "ply_path": str(ply_path),
            "ply_scale": 1.0,
            "shape_params": shape_params,
            "batch_size": 100,
            "n_dom": 50,
            "device": "cpu",
            "cache_dir": cache_dir,
        }
        # First call: build cache from original PLY.
        _load_or_build_surface_cache(**kwargs)
        assert len(list(cache_dir.glob("*.pt"))) == 1

        # Overwrite PLY with a different sphere (same path, different byte content).
        new_sphere = trimesh.creation.icosphere(radius=300.0)
        new_sphere.export(str(ply_path))

        # Update shape_params to match new geometry.
        shape_params_new = self._make_ply_shape_params(ply_path)
        kwargs["shape_params"] = shape_params_new

        # Second call: new content must cause a cache miss → 2 .pt files in cache_dir.
        _load_or_build_surface_cache(**kwargs)
        cache_files = list(cache_dir.glob("*.pt"))
        assert len(cache_files) == 2, (
            f"Expected 2 .pt files after PLY content change (cache miss), got {len(cache_files)}. "
            "The cache key must include a PLY content hash (Plan 028 M1)."
        )


# ---------------------------------------------------------------------------
# BUG-001 — clear_mesh_cache inserted before per-slot extraction (Plan 018 M1)
# ---------------------------------------------------------------------------


class TestClearMeshCacheOnExtraction:
    """Tests that clear_mesh_cache() is called before each per-slot _extract_mesh call.

    BUG-001 root cause: the global _MESH_CACHE in cache.py is keyed without the
    active latent vector, so all slots hit the same cache entry and produce
    identical PLY files.  The fix inserts clear_mesh_cache() before each call.

    These tests verify the call-order contract without running full pretraining.
    """

    def test_clear_mesh_cache_importable_from_pretraining_latent_module(self) -> None:
        """clear_mesh_cache must be importable from within pretraining_latent's namespace.

        This test verifies that the import was added to pretraining_latent.py.
        It will FAIL until the import is present (TDD red phase).
        """
        mod = importlib.import_module(PRETRAIN_LATENT)
        assert hasattr(mod, "clear_mesh_cache"), (
            "clear_mesh_cache not found in pretraining_latent module namespace. "
            "Add: from subprojects.wp5.unified_mesh_extraction import clear_mesh_cache"
        )

    def test_post_training_export_clears_cache_per_slot(self, tmp_path: Path) -> None:
        """clear_mesh_cache() must be called once per slot in the post-training export loop.

        Strategy: patch clear_mesh_cache and _extract_mesh inside pretraining_latent,
        run pretraining (3 epochs, 2 slots), then assert clear_mesh_cache was called
        at least as many times as there are slots in the export loop.
        """
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config(str(tmp_path / "ckpt.pth"))

        mock_clear = MagicMock()
        # _extract_mesh must not actually write PLY files; stub out the save path
        mock_extract = MagicMock(return_value=None)

        with (
            patch(f"{PRETRAIN_LATENT}.clear_mesh_cache", mock_clear),
            patch(f"{PRETRAIN_LATENT}._extract_mesh", mock_extract),
        ):
            pretraining_latent(model, config, "cpu")

        num_slots = len(config["latent_population"])  # 2
        # At least one clear per slot in the export loop
        assert mock_clear.call_count >= num_slots, (
            f"Expected clear_mesh_cache() called >= {num_slots} times (one per export slot), "
            f"got {mock_clear.call_count}. "
            "Add clear_mesh_cache() inside the post-training export loop."
        )


# ---------------------------------------------------------------------------
# Plan 021 — _compute_input_grad_norms (TDD red phase)
# ---------------------------------------------------------------------------


class TestComputeInputGradNorms:
    """Tests for _compute_input_grad_norms diagnostic helper (Plan 021).

    These tests are written BEFORE the implementation (TDD red phase).
    They will fail with AttributeError until _compute_input_grad_norms exists.
    """

    def test_importable(self) -> None:
        """_compute_input_grad_norms must be importable from pretraining_latent."""
        mod = importlib.import_module(PRETRAIN_LATENT)
        assert hasattr(mod, "_compute_input_grad_norms"), (
            "_compute_input_grad_norms not found in pretraining_latent module. "
            "Add the helper function as specified in Plan 021 M1."
        )

    def test_returns_positive_norms_for_sensitive_model(self) -> None:
        """Both z_grad_norm and xyz_grad_norm should be > 0 for a standard ConcatWIREV2."""
        _compute_input_grad_norms = _import(PRETRAIN_LATENT, "_compute_input_grad_norms")
        model = _make_concat_wire_v2()
        model.eval()
        n = 8
        pts = torch.randn(n, 3) * 500.0  # mm-scale spatial coords
        z_dim = model.get_model_info_dict()["z_dim"]
        z = torch.randn(n, z_dim)
        gt_sdf = torch.zeros(n)  # zero GT → loss = mean((pred/1000)^2)

        z_norm, xyz_norm, z_warn = _compute_input_grad_norms(model, pts, z, gt_sdf, "cpu")

        assert not z_warn, "z gradient should not be None — z is concatenated into model input"
        assert z_norm > 0.0, f"z_grad_norm should be positive, got {z_norm}"
        assert xyz_norm > 0.0, f"xyz_grad_norm should be positive, got {xyz_norm}"

    def test_ratio_is_finite_and_positive(self) -> None:
        """grad_norm_ratio z_norm / (xyz_norm + eps) should be finite and positive."""
        _compute_input_grad_norms = _import(PRETRAIN_LATENT, "_compute_input_grad_norms")
        model = _make_concat_wire_v2()
        model.eval()
        n = 8
        pts = torch.randn(n, 3) * 500.0
        z_dim = model.get_model_info_dict()["z_dim"]
        z = torch.randn(n, z_dim)
        gt_sdf = torch.zeros(n)

        z_norm, xyz_norm, _ = _compute_input_grad_norms(model, pts, z, gt_sdf, "cpu")
        ratio = z_norm / (xyz_norm + 1e-8)

        assert math.isfinite(ratio), f"grad_norm_ratio should be finite, got {ratio}"
        assert ratio > 0.0, f"grad_norm_ratio should be positive, got {ratio}"

    def test_model_params_grad_not_set(self) -> None:
        """Probe must not accumulate .grad on model parameters (no optimizer pollution)."""
        _compute_input_grad_norms = _import(PRETRAIN_LATENT, "_compute_input_grad_norms")
        model = _make_concat_wire_v2()
        model.eval()
        # Ensure all .grad are None before the probe
        for p in model.parameters():
            p.grad = None

        n = 8
        pts = torch.randn(n, 3) * 500.0
        z_dim = model.get_model_info_dict()["z_dim"]
        z = torch.randn(n, z_dim)
        gt_sdf = torch.zeros(n)

        _compute_input_grad_norms(model, pts, z, gt_sdf, "cpu")

        # After probe, model parameter grads must still be None
        param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert param_grads == [], (
            f"Probe must not set .grad on model parameters. Found {len(param_grads)} parameters with non-None grad after probe."
        )

    def test_integration_print_contains_z_grad(self, tmp_path: Path) -> None:
        """Integration: epoch print line must contain 'z_grad=' when print_interval=1."""
        pretraining_latent = _import(PRETRAIN_LATENT, "pretraining_latent")
        model = _make_concat_wire_v2()
        config = _make_latent_pretrain_config(str(tmp_path / "ckpt.pth"))
        config["training_loop"]["epochs"] = 2
        config["training_loop"]["print_interval"] = 1

        written: list[str] = []
        mock_extract = MagicMock(return_value=None)

        with (
            patch(f"{PRETRAIN_LATENT}._extract_mesh", mock_extract),
            patch("tqdm.tqdm.write", side_effect=written.append),
        ):
            pretraining_latent(model, config, "cpu")

        epoch_lines = [ln for ln in written if "[latent_pretrain]" in ln and "Epoch" in ln]
        assert epoch_lines, f"No epoch print lines captured. All writes: {written}"
        assert any("z_grad=" in ln for ln in epoch_lines), "Expected 'z_grad=' in epoch print output. Epoch lines:\n" + "\n".join(
            epoch_lines
        )
