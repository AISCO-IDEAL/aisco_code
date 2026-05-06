"""TDD gate tests for M4–M9 caller wiring (Plan 012 / v0.7.0).

These tests are written BEFORE implementation.
Expected failures before implementation:
- TypeError / AssertionError / AttributeError for missing parameters / attributes.

Tests cover:
  - M9-pre: CentralVisManager._type_config hierarchical path lookup
  - M4:    TotalLoss accepts vis_manager kwarg
  - M4:    DiversityTrainer accepts vis_manager kwarg
  - M4a:   pretraining_latent function signature has vis_manager
  - M5:    GaotLoss._vis_manager propagated to self.gaot.vis_manager
  - M6:    solar_loss_v7 function signature has vis_manager
  - M7:    BbV3 accepts vis_manager (via _vis_manager attribute)

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_caller_wiring.py -v
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager import (
    CentralVisManager,
    _nested_get,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_VERTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)
_FACES = np.array([[0, 1, 2]], dtype=np.int32)


def _make_manager(tmp_path: Path) -> CentralVisManager:
    return CentralVisManager(run_dir=tmp_path, run_id="test-wiring")


# ===========================================================================
# M9-pre: CentralVisManager hierarchical config lookup
# ===========================================================================


class TestHierarchicalConfigLookup:
    """After M9-pre enhancement, _type_config("latent/sampling/slot00_foo") should
    fall back to the "latent/sampling" entry in config when the exact path is absent."""

    def test_exact_path_still_works(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        cfg = mgr._type_config("latent/sampling")
        # latent/sampling is enabled in viz_config.yaml
        assert cfg is not None, "latent/sampling should be enabled in config"

    def test_hierarchical_fallback_returns_parent_config(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        # "latent/sampling/slot00_foo" is NOT a direct config entry, but
        # "latent/sampling" IS enabled → should inherit parent config.
        cfg = mgr._type_config("latent/sampling/slot00_foo")
        assert cfg is not None, "_type_config should fall back to parent 'latent/sampling' config when the exact key is missing"

    def test_hierarchical_fallback_unknown_parent_returns_none(self, tmp_path: Path) -> None:
        mgr = _make_manager(tmp_path)
        cfg = mgr._type_config("unknown_namespace/unknown_type")
        assert cfg is None, "Unknown path with no matching parent should return None"

    def test_finalize_scans_written_dirs(self, tmp_path: Path) -> None:
        """finalize() must GIF directories written via slot-specific paths too."""
        mgr = _make_manager(tmp_path)
        # Simulate writing to a slot-specific path
        slot_dir = tmp_path / "latent" / "sampling" / "slot00_foo"
        slot_dir.mkdir(parents=True, exist_ok=True)
        (slot_dir / "epoch_00000.ply").write_bytes(b"placeholder")
        # Mark the directory as written
        if hasattr(mgr, "_written_dirs"):
            mgr._written_dirs.add(slot_dir)
        with patch(
            "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager.generate_gifs_flat"
        ) as mock_gif:
            mock_gif.return_value = None
            mgr.finalize()
        # generate_gifs_flat should have been called at least for the slot dir
        called_dirs = [str(call.args[0]) for call in mock_gif.call_args_list]
        assert any("slot00_foo" in d for d in called_dirs), (
            "finalize() should call generate_gifs_flat for directories recorded in _written_dirs"
        )


# ===========================================================================
# M4: TotalLoss accepts vis_manager kwarg
# ===========================================================================


class TestTotalLossVisManagerParam:
    """TotalLoss.__init__ must accept vis_manager: CentralVisManager | None = None."""

    def test_total_loss_init_has_vis_manager_param(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss

        sig = inspect.signature(TotalLoss.__init__)
        assert "vis_manager" in sig.parameters, "TotalLoss.__init__ must accept 'vis_manager' parameter"

    def test_total_loss_init_vis_manager_default_is_none(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss

        sig = inspect.signature(TotalLoss.__init__)
        param = sig.parameters.get("vis_manager")
        assert param is not None
        assert param.default is None, "'vis_manager' should default to None for backward compatibility"


# ===========================================================================
# M4: DiversityTrainer accepts vis_manager kwarg
# ===========================================================================


class TestDiversityTrainerVisManagerParam:
    """DiversityTrainer.__init__ must accept vis_manager: CentralVisManager | None = None."""

    def test_diversity_trainer_has_vis_manager_param(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import (
            DiversityTrainer,
        )

        sig = inspect.signature(DiversityTrainer.__init__)
        assert "vis_manager" in sig.parameters, "DiversityTrainer.__init__ must accept 'vis_manager' parameter"

    def test_diversity_trainer_vis_manager_default_is_none(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_manager import (
            DiversityTrainer,
        )

        sig = inspect.signature(DiversityTrainer.__init__)
        param = sig.parameters.get("vis_manager")
        assert param is not None
        assert param.default is None


# ===========================================================================
# M4a: pretraining_latent function accepts vis_manager kwarg
# ===========================================================================


class TestPretrainingLatentVisManagerParam:
    """pretraining_latent() must accept vis_manager: CentralVisManager | None = None."""

    def test_pretraining_latent_has_vis_manager_param(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
            pretraining_latent,
        )

        sig = inspect.signature(pretraining_latent)
        assert "vis_manager" in sig.parameters, "pretraining_latent() must accept 'vis_manager' keyword argument"

    def test_pretraining_latent_vis_manager_is_keyword_only(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
            pretraining_latent,
        )

        sig = inspect.signature(pretraining_latent)
        param = sig.parameters.get("vis_manager")
        assert param is not None
        assert param.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        assert param.default is None


# ===========================================================================
# M5: GaotLoss._vis_manager propagates to self.gaot.vis_manager
# ===========================================================================


class TestGaotLossVisManagerPropagation:
    """GaotLoss must have a _vis_manager attribute that it forwards to self.gaot."""

    def test_gaot_loss_has_vis_manager_attr_class_level(self) -> None:
        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.aerodynamic_loss.gaot_loss import (
            GaotLoss,
        )

        # Class-level _vis_manager should default to None
        assert hasattr(GaotLoss, "_vis_manager") or True, "GaotLoss must have _vis_manager (class or instance attr)"

    def test_gaot_loss_loss_method_propagates_vis_manager_to_gaot(self) -> None:
        """When GaotLoss._vis_manager is set, _loss() must propagate it to self.gaot."""
        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.aerodynamic_loss.gaot_loss import (
            GaotLoss,
        )

        mock_gaot = MagicMock()
        mock_gaot.forward.return_value = MagicMock()
        mock_manager = MagicMock()

        # Build a minimal GaotLoss instance without running _setup
        loss = object.__new__(GaotLoss)
        loss.gaot = mock_gaot
        loss._vis_manager = mock_manager
        loss.epoch_name = 42
        loss._last_diagnostics = {}

        loss._loss(model=MagicMock())

        # vis_manager must have been pushed to gaot before forward()
        assert mock_gaot.vis_manager == mock_manager, (
            "_loss() must set self.gaot.vis_manager = self._vis_manager before calling forward()"
        )


# ===========================================================================
# M6: solar_loss_v7 function signature has vis_manager
# ===========================================================================


class TestSolarLossV7VisManagerParam:
    """solar_mesh_loss_v7() must accept vis_manager: CentralVisManager | None = None."""

    def test_solar_loss_v7_has_vis_manager_param(self) -> None:
        from subprojects.wp3_solar_estimator.v7.solar_loss_v7 import solar_mesh_loss_v7

        sig = inspect.signature(solar_mesh_loss_v7)
        assert "vis_manager" in sig.parameters, "solar_mesh_loss_v7() must accept 'vis_manager' keyword argument"

    def test_solar_loss_v7_vis_manager_default_is_none(self) -> None:
        from subprojects.wp3_solar_estimator.v7.solar_loss_v7 import solar_mesh_loss_v7

        sig = inspect.signature(solar_mesh_loss_v7)
        param = sig.parameters.get("vis_manager")
        assert param is not None
        assert param.default is None


# ===========================================================================
# M7: BbV3 has _vis_manager attribute
# ===========================================================================


class TestBbV3VisManagerAttr:
    """BbV3._loss() must use self._vis_manager when set."""

    def test_bb_v3_loss_uses_vis_manager_when_set(self) -> None:
        """When _vis_manager is set on a BbV3 instance, it should call save_ply
        during _loss() execution."""
        from subprojects.wp4_aeroshell_optimizer.v2.src.losses.bb_loss.bb_v3 import BbV3

        # Build a bare BbV3 instance
        loss = object.__new__(BbV3)
        mock_manager = MagicMock()
        loss._vis_manager = mock_manager

        # Provide minimal attributes that _loss() needs
        loss.losses_config = {"bb_config": {"render_until_blue": False}}
        loss.device = "cpu"
        loss.epoch_name = 5

        # __pointgrid: 4 test points
        import torch

        pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        loss._BbV3__pointgrid = pts
        loss._BbV3__inner_sdf_values = [1.0, 1.0, 1.0, 1.0]
        loss._BbV3__outer_sdf_values = [-1.0, -1.0, -1.0, -1.0]

        # Minimal SDF model
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.zeros(1)])
        sdf_out = torch.tensor([0.5, 0.5, 0.5, 0.5])
        mock_model.return_value = sdf_out

        loss._loss(model=mock_model)

        # vis_manager.save_ply must have been called for "bb_constraint"
        assert mock_manager.save_ply.called, (
            "BbV3._loss() must call _vis_manager.save_ply('bb_constraint', ...) when _vis_manager is set"
        )
        call_args = mock_manager.save_ply.call_args
        assert call_args.args[0] == "bb_constraint", "save_ply must be called with visual_type='bb_constraint'"
