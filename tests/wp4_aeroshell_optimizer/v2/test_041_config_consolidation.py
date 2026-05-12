"""
TDD tests for Plan 041 — Config Consolidation (M1, M2, M3).

Red phase: these tests are written BEFORE any implementation.
They verify the post-implementation state and MUST FAIL before implementation.

Tests:
  - M1: training_config.yaml exists, contains all expected keys from old GINN.yaml
         + new OQ-A keys (pretrain_population_size, z_init_mode, r_init_scale)
         + new OQ-B key (z_resample_mode)
         + training_v0_config.yaml is gone
         + GINN.yaml is gone
  - M1: Sweep resolution: _resolve_sweep_target prefix "training_config" works
  - M2: vis_config.yaml exists at v2/configs/, has latent_pretraining and latent_optimisation sections
         + old viz_config.yaml (in sampling_visualization/) is gone
         + CVM2 _DEFAULT_CONFIG_PATH points to the new file
  - M3: CentralVisManager not importable from active sampling_visualization package
         + CentralVisManagerV2 still importable
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization import (
    CentralVisManagerV2,
    NullVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import (
    _DEFAULT_CONFIG_PATH as _CVM2_DEFAULT_CONFIG_PATH,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_V2_CONFIGS = Path("/cluster/home/kdiehl/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/configs")
_V2_SAMPLING_VIS = Path("/cluster/home/kdiehl/AISCO_code/subprojects/wp4_aeroshell_optimizer/v2/src/sampling_visualization")

# ---------------------------------------------------------------------------
# M1 — training_config.yaml
# ---------------------------------------------------------------------------


class TestM1TrainingConfig:
    """Tests for the GINN.yaml + training_v0_config.yaml merge."""

    def test_training_config_exists(self) -> None:
        """training_config.yaml must exist after rename."""
        assert (_V2_CONFIGS / "training_config.yaml").exists(), (
            "training_config.yaml not found at v2/configs/. Run M1: rename training_v0_config.yaml and merge GINN.yaml."
        )

    def test_old_training_v0_config_gone(self) -> None:
        """training_v0_config.yaml must be gone after rename."""
        assert not (_V2_CONFIGS / "training_v0_config.yaml").exists(), (
            "training_v0_config.yaml still exists. It should have been renamed to training_config.yaml."
        )

    def test_ginn_yaml_gone(self) -> None:
        """GINN.yaml must be deleted after merge into training_config.yaml."""
        assert not (_V2_CONFIGS / "GINN.yaml").exists(), (
            "GINN.yaml still exists. Its keys should have been merged into training_config.yaml."
        )

    def test_training_params_keys_present(self) -> None:
        """All original training_params keys from training_v0_config.yaml must be present."""
        cfg = yaml.safe_load((_V2_CONFIGS / "training_config.yaml").read_text())
        params = cfg.get("training_params", {})
        required = [
            "learning_rate",
            "num_epochs",
            "mesh_save_interval",
            "marching_cubes_usage",
            "gradient_conflict_cadence",
            "gradient_conflict_output_dir",
        ]
        for key in required:
            assert key in params, f"training_params.{key} missing from training_config.yaml"

    def test_ginn_keys_present(self) -> None:
        """All original GINN.yaml keys must be present at top level of training_config.yaml."""
        cfg = yaml.safe_load((_V2_CONFIGS / "training_config.yaml").read_text())
        ginn_keys = [
            "population_size",
            "bbox_scale_factor",
            "diversity_num_points",
            "diversity_norm_order",
            "diversity_neighbor_agg_fn",
            "force_z0_to_origin",
            "z_resample_cadence",
            "z_resample_start_epoch",
            "gradient_clip_norm",
            "save_population_meshes",
        ]
        for key in ginn_keys:
            assert key in cfg, f"Key '{key}' (from GINN.yaml) missing from training_config.yaml"

    def test_oqa_pretrain_population_size_key(self) -> None:
        """OQ-A: pretrain_population_size gate key must be present."""
        cfg = yaml.safe_load((_V2_CONFIGS / "training_config.yaml").read_text())
        assert "pretrain_population_size" in cfg, "pretrain_population_size key missing. Required by OQ-A resolution."

    def test_oqa_z_init_mode_key(self) -> None:
        """OQ-A: z_init_mode key must be present."""
        cfg = yaml.safe_load((_V2_CONFIGS / "training_config.yaml").read_text())
        assert "z_init_mode" in cfg, "z_init_mode key missing. Required by OQ-A resolution (circle | wider_circle)."

    def test_oqa_r_init_scale_key(self) -> None:
        """OQ-A: r_init_scale key must be present."""
        cfg = yaml.safe_load((_V2_CONFIGS / "training_config.yaml").read_text())
        assert "r_init_scale" in cfg, "r_init_scale key missing. Required by OQ-A resolution."

    def test_oqb_z_resample_mode_key(self) -> None:
        """OQ-B: z_resample_mode key must be present."""
        cfg = yaml.safe_load((_V2_CONFIGS / "training_config.yaml").read_text())
        assert "z_resample_mode" in cfg, "z_resample_mode key missing. Required by OQ-B resolution (circle | grid)."

    def test_general_config_training_ref_updated(self) -> None:
        """general_config.yaml active_configs.training must reference training_config.yaml."""
        cfg = yaml.safe_load((_V2_CONFIGS / "general_config.yaml").read_text())
        training_ref = cfg.get("active_configs", {}).get("training", "")
        assert training_ref == "training_config.yaml", (
            f"general_config.active_configs.training is '{training_ref}', expected 'training_config.yaml'"
        )


# ---------------------------------------------------------------------------
# M2 — vis_config.yaml
# ---------------------------------------------------------------------------


class TestM2VisConfig:
    """Tests for the viz_config.yaml → vis_config.yaml move."""

    def test_vis_config_exists(self) -> None:
        """vis_config.yaml must exist at v2/configs/ after move."""
        assert (_V2_CONFIGS / "vis_config.yaml").exists(), (
            "vis_config.yaml not found at v2/configs/. "
            "Run M2: move v2/src/sampling_visualization/viz_config.yaml to v2/configs/vis_config.yaml."
        )

    def test_old_viz_config_gone(self) -> None:
        """viz_config.yaml must be gone from sampling_visualization/ after move."""
        assert not (_V2_SAMPLING_VIS / "viz_config.yaml").exists(), (
            "viz_config.yaml still exists in sampling_visualization/. It should have been moved to v2/configs/vis_config.yaml."
        )

    def test_latent_pretraining_section(self) -> None:
        """vis_config.yaml must have latent_pretraining section."""
        cfg = yaml.safe_load((_V2_CONFIGS / "vis_config.yaml").read_text())
        assert "latent_pretraining" in cfg, "latent_pretraining section missing from vis_config.yaml."
        lp = cfg["latent_pretraining"]
        for key in ["vis_cadence_pretraining", "vis_show_domain_pts", "vis_show_target_pts", "mesh_resolution", "make_gif"]:
            assert key in lp, f"latent_pretraining.{key} missing from vis_config.yaml"

    def test_latent_optimisation_section(self) -> None:
        """vis_config.yaml must have latent_optimisation section."""
        cfg = yaml.safe_load((_V2_CONFIGS / "vis_config.yaml").read_text())
        assert "latent_optimisation" in cfg, "latent_optimisation section missing from vis_config.yaml."
        lo = cfg["latent_optimisation"]
        for key in ["vis_cadence_optimisation", "vis_strategy", "vis_all_losses"]:
            assert key in lo, f"latent_optimisation.{key} missing from vis_config.yaml"

    def test_vis_latent_circle_survey_subsection(self) -> None:
        """vis_config.yaml must have latent_optimisation.vis_latent_circle_survey sub-section."""
        cfg = yaml.safe_load((_V2_CONFIGS / "vis_config.yaml").read_text())
        lo = cfg.get("latent_optimisation", {})
        assert "vis_latent_circle_survey" in lo, (
            "latent_optimisation.vis_latent_circle_survey sub-section missing from vis_config.yaml."
        )

    def test_latent_space_exploration_subsection(self) -> None:
        """vis_config.yaml must have latent_pretraining.latent_space_exploration sub-section."""
        cfg = yaml.safe_load((_V2_CONFIGS / "vis_config.yaml").read_text())
        lp = cfg.get("latent_pretraining", {})
        assert "latent_space_exploration" in lp, (
            "latent_pretraining.latent_space_exploration sub-section missing from vis_config.yaml."
        )

    def test_cvm2_default_config_path(self) -> None:
        """CVM2 _DEFAULT_CONFIG_PATH must resolve to the new vis_config.yaml location."""
        assert _CVM2_DEFAULT_CONFIG_PATH.exists(), (
            f"CVM2 _DEFAULT_CONFIG_PATH {_CVM2_DEFAULT_CONFIG_PATH} does not exist. "
            "Update central_vis_manager_v2.py to point to v2/configs/vis_config.yaml."
        )
        assert _CVM2_DEFAULT_CONFIG_PATH.name == "vis_config.yaml", (
            f"CVM2 _DEFAULT_CONFIG_PATH name is '{_CVM2_DEFAULT_CONFIG_PATH.name}', expected 'vis_config.yaml'."
        )

    def test_general_config_no_interpolator_key(self) -> None:
        """general_config.yaml active_configs must not have interpolator key."""
        cfg = yaml.safe_load((_V2_CONFIGS / "general_config.yaml").read_text())
        active = cfg.get("active_configs", {})
        assert "interpolator" not in active, (
            "interpolator key still present in general_config.active_configs. Remove it as part of M2."
        )

    def test_postprocessing_interpolator_archived(self) -> None:
        """postprocessing_interpolator_config.yaml must be in archive, not active configs."""
        assert not (_V2_CONFIGS / "postprocessing_interpolator_config.yaml").exists(), (
            "postprocessing_interpolator_config.yaml still in v2/configs/. Move it to v2/configs/archive/."
        )


# ---------------------------------------------------------------------------
# M3 — CVM1 archived
# ---------------------------------------------------------------------------


class TestM3CVM1Archived:
    """Tests for archiving CentralVisManager (CVM1)."""

    def test_cvm1_not_in_sampling_visualization_init(self) -> None:
        """CentralVisManager must not be importable from the sampling_visualization package."""
        mod = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization")
        assert not hasattr(mod, "CentralVisManager"), (
            "CentralVisManager still exported from sampling_visualization.__init__. Remove it from __all__ and the import line."
        )

    def test_cvm1_not_in_all(self) -> None:
        """CentralVisManager must not be in __all__ of sampling_visualization."""
        mod = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization")
        all_exports = getattr(mod, "__all__", [])
        assert "CentralVisManager" not in all_exports, "CentralVisManager still in __all__ of sampling_visualization.__init__."

    def test_cvm2_still_importable(self) -> None:
        """CentralVisManagerV2 must still be importable after CVM1 archival."""
        assert CentralVisManagerV2 is not None

    def test_null_vis_manager_still_importable(self) -> None:
        """NullVisManager must still be importable after CVM1 archival."""
        assert NullVisManager is not None

    def test_cvm1_archive_file_exists(self) -> None:
        """central_vis_manager.py must exist in the archive folder after move."""
        archive_path = _V2_SAMPLING_VIS / "archive" / "central_vis_manager.py"
        assert archive_path.exists(), (
            f"Archived CVM1 not found at {archive_path}. Move central_vis_manager.py to sampling_visualization/archive/."
        )
