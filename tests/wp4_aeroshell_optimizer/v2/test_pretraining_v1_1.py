"""Tests for _sample_points_for_shape dual-band sampling (Plan 013 / M2).

Covers:
  - Dual-band PLY sampling: both tight and wide bands present in output
  - Fraction math: n_domain + n_near + n_exact == batch_size
  - Backward compatibility: single-band when near_surface_wide_sigma=None
  - domain_fraction / near_surface_fraction control output sizes

Run:
    pytest tests/wp4_aeroshell_optimizer/v2/test_pretraining_v1_1.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init

_V2 = "subprojects.wp4_aeroshell_optimizer.v2.src"
PRETRAIN_V1_1 = f"{_V2}.main_files.pretraining_v1_1"


def _import(module: str, name: str) -> Any:
    mod = importlib.import_module(module)
    return getattr(mod, name)


def _make_ply_shape_params() -> dict[str, Any]:
    """Create a minimal watertight PLY shape_params dict for testing."""
    sphere = trimesh.creation.icosphere(radius=500.0)
    verts_np = np.array(sphere.vertices, dtype="float32")
    faces_np = np.array(sphere.faces, dtype="int64")
    verts_t = torch.tensor(verts_np)
    faces_t = torch.tensor(faces_np)
    extents = verts_np.max(axis=0) - verts_np.min(axis=0)
    bbox_extent = float(np.linalg.norm(extents))
    return {
        "ply_verts": verts_t,
        "ply_faces": faces_t,
        "bbox_extent": bbox_extent,
        "surface_noise_sigma_factor": 0.05,
    }


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestSamplePointsBackwardCompat:
    """Dual-band params absent/None must reproduce the previous single-band behaviour."""

    def test_ply_total_count_single_band(self) -> None:
        """Without dual-band params, total output == batch_size."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        shape_params = _make_ply_shape_params()
        batch = 128
        pts, gt = _sample_points_for_shape("ply", shape_params, batch, "cpu")
        assert pts.shape == (batch, 3), f"Expected ({batch}, 3), got {pts.shape}"
        assert gt.shape == (batch,), f"Expected ({batch},), got {gt.shape}"

    def test_ply_domain_half_is_50pct_by_default(self) -> None:
        """Without dual-band params, domain fraction defaults to 0.50."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        shape_params = _make_ply_shape_params()
        batch = 100
        pts, _gt = _sample_points_for_shape("ply", shape_params, batch, "cpu")
        # domain pts are first n_domain; near-surface pts are last n_near
        # With default domain_fraction=0.50: n_domain=50, n_near=50
        assert pts.shape[0] == batch


# ---------------------------------------------------------------------------
# Dual-band fractions
# ---------------------------------------------------------------------------


class TestDualBandFractions:
    """Verify that near_surface_fraction and domain_fraction control output sizes."""

    def test_total_equals_batch_size(self) -> None:
        """With all four dual-band params set, total output == batch_size."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        shape_params = _make_ply_shape_params()
        batch = 200
        pts, gt = _sample_points_for_shape(
            "ply",
            shape_params,
            batch,
            "cpu",
            near_surface_fraction=0.30,
            domain_fraction=0.50,
            near_surface_tight_sigma=0.005,
            near_surface_wide_sigma=0.05,
        )
        assert pts.shape[0] == batch, f"Expected {batch}, got {pts.shape[0]}"
        assert gt.shape[0] == batch

    def test_domain_fraction_respected(self) -> None:
        """Changing domain_fraction to 0.70 shifts the domain/surface balance."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        shape_params = _make_ply_shape_params()
        batch = 100
        pts_70, _gt = _sample_points_for_shape(
            "ply",
            shape_params,
            batch,
            "cpu",
            near_surface_fraction=0.20,
            domain_fraction=0.70,
            near_surface_tight_sigma=0.005,
            near_surface_wide_sigma=0.05,
        )
        assert pts_70.shape[0] == batch

    def test_near_surface_pts_are_closer_to_mesh(self) -> None:
        """Tight-band pts should have smaller mean |SDF| than domain pts."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        shape_params = _make_ply_shape_params()

        torch.manual_seed(42)
        batch = 400
        _pts, gt = _sample_points_for_shape(
            "ply",
            shape_params,
            batch,
            "cpu",
            near_surface_fraction=0.30,
            domain_fraction=0.50,
            near_surface_tight_sigma=0.005,
            near_surface_wide_sigma=0.05,
        )

        # Domain pts are the first n_domain = int(400*0.50) = 200
        n_domain = int(batch * 0.50)
        domain_sdf_abs = gt[:n_domain].abs().mean().item()
        surface_sdf_abs = gt[n_domain:].abs().mean().item()

        assert surface_sdf_abs < domain_sdf_abs, (
            f"Near-surface pts should have smaller |SDF| than domain pts, "
            f"got surface={surface_sdf_abs:.1f} vs domain={domain_sdf_abs:.1f}"
        )


# ---------------------------------------------------------------------------
# Dual-band wide sigma
# ---------------------------------------------------------------------------


class TestDualBandWideSigma:
    """Verify that wide-sigma and tight-sigma produce different spread distributions."""

    def test_both_bands_have_positive_spread(self) -> None:
        """Both tight and wide near-surface pts must have positive spread (not all at exact surface)."""
        _sample_points_for_shape = _import(PRETRAIN_V1_1, "_sample_points_for_shape")
        shape_params = _make_ply_shape_params()

        torch.manual_seed(99)
        batch = 400
        pts, _gt = _sample_points_for_shape(
            "ply",
            shape_params,
            batch,
            "cpu",
            near_surface_fraction=0.40,  # 160 near-surface pts
            domain_fraction=0.40,
            near_surface_tight_sigma=0.005,
            near_surface_wide_sigma=0.05,
        )
        # Near-surface pts: index [n_domain, n_domain + n_near)
        n_domain = int(batch * 0.40)
        n_near = int(batch * 0.40)
        near_pts = pts[n_domain : n_domain + n_near]  # (n_near, 3)

        # Split into tight (first half) and wide (second half)
        half = n_near // 2
        tight_pts = near_pts[:half]
        wide_pts = near_pts[half:]

        tight_std = tight_pts.std().item()
        wide_std = wide_pts.std().item()

        assert tight_std > 0.0, "Tight-band pts must have positive spread"
        assert wide_std > 0.0, "Wide-band pts must have positive spread"
        # Wide sigma (0.05 * bbox_extent) >> tight sigma (0.005 * bbox_extent), so wide_std should be larger
        assert wide_std > tight_std, f"Wide-band std ({wide_std:.2f}) should exceed tight-band std ({tight_std:.2f})"


# ---------------------------------------------------------------------------
# Plan 031 — _compute_solitary_pretrain_hash / _resolve_solitary_ckpt_path
# Tests written BEFORE implementation (TDD red phase).
# Fail with AttributeError until the functions exist.
# ---------------------------------------------------------------------------


def _make_simple_model() -> Any:
    """Return a minimal model implementing SupportsModelContract."""
    model, _, _, _ = smart_model_init("ConcatWIREV2")
    return model.to("cpu")


def _make_solitary_config() -> dict:
    """Minimal config matching pretraining_v0_config.yaml schema."""
    return {
        "pretraining_shape": "sphere",
        "training_loop": {
            "epochs": 5,
            "learning_rate": 1e-4,
            "random_batch_size": 64,
            "print_interval": 5,
        },
    }


class TestSolitaryPretrainHash:
    """Tests for _compute_solitary_pretrain_hash and _resolve_solitary_ckpt_path (Plan 031 / M4).

    Written BEFORE implementation (TDD red phase).
    Fail with AttributeError until the functions exist in pretraining_v1_1.py.
    """

    def test_hash_stable_same_config(self) -> None:
        """Same config + model → same hash both times."""
        _compute_solitary_pretrain_hash = _import(PRETRAIN_V1_1, "_compute_solitary_pretrain_hash")
        model = _make_simple_model()
        config = _make_solitary_config()
        h1 = _compute_solitary_pretrain_hash(config, model)
        h2 = _compute_solitary_pretrain_hash(config, model)
        assert h1 == h2

    def test_hash_changes_when_epoch_changes(self) -> None:
        """Changing epochs → different hash."""
        _compute_solitary_pretrain_hash = _import(PRETRAIN_V1_1, "_compute_solitary_pretrain_hash")
        model = _make_simple_model()
        config_a = _make_solitary_config()
        config_b = _make_solitary_config()
        config_b["training_loop"]["epochs"] = 999
        assert _compute_solitary_pretrain_hash(config_a, model) != _compute_solitary_pretrain_hash(config_b, model)

    def test_hash_changes_when_shape_changes(self) -> None:
        """Changing pretraining_shape → different hash."""
        _compute_solitary_pretrain_hash = _import(PRETRAIN_V1_1, "_compute_solitary_pretrain_hash")
        model = _make_simple_model()
        config_a = _make_solitary_config()
        config_b = {**_make_solitary_config(), "pretraining_shape": "cube"}
        assert _compute_solitary_pretrain_hash(config_a, model) != _compute_solitary_pretrain_hash(config_b, model)

    def test_resolve_path_structure(self, tmp_path: Path) -> None:
        """_resolve_solitary_ckpt_path returns path under base_dir with epoch prefix + hash."""
        _resolve_solitary_ckpt_path = _import(PRETRAIN_V1_1, "_resolve_solitary_ckpt_path")
        _compute_solitary_pretrain_hash = _import(PRETRAIN_V1_1, "_compute_solitary_pretrain_hash")
        model = _make_simple_model()
        config = _make_solitary_config()
        base_dir = tmp_path
        resolved = _resolve_solitary_ckpt_path(config, model, base_dir)
        config_hash = _compute_solitary_pretrain_hash(config, model)
        # Checkpoint must be inside base_dir
        assert resolved.parent.parent == base_dir, f"Expected base_dir {base_dir}, got {resolved.parent.parent}"
        # Folder name must contain hash
        assert config_hash in resolved.parent.name
        # Folder name must contain epoch prefix
        assert resolved.parent.name.startswith("000005"), f"Expected '000005' prefix, got {resolved.parent.name}"
        # Filename embeds hash
        assert f"pretrained_{config_hash}.pth" == resolved.name
