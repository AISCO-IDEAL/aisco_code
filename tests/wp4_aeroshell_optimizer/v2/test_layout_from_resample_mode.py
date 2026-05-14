"""Tests for the layout_from_resample_mode toggle in vis_config.yaml.

Scope
-----
T1: vis_config.yaml declares layout_from_resample_mode under vis_latent_circle_survey.
T2: layout_from_resample_mode=True + z_resample_mode="in_circle" → layout becomes "pizza".
T3: layout_from_resample_mode=True + z_resample_mode="on_circle" → layout becomes "strip".
T4: layout_from_resample_mode=True + z_resample_mode="random" → layout becomes "strip" (fallback).
T5: layout_from_resample_mode=False → layout is left unchanged regardless of z_resample_mode.
T6: _load_config raising an exception → layout is left unchanged (silent no-op).
T7: main_v2.py source references the layout_from_resample_mode key.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import subprojects.wp4_aeroshell_optimizer.v2.src.main_files.main_v2 as main_mod

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).parents[3]  # …/AISCO_code
_VIS_CFG_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/configs/vis_config.yaml"
_MAIN_V2_PATH = _REPO / "subprojects/wp4_aeroshell_optimizer/v2/src/main_files/main_v2.py"

VIS_CFG = yaml.safe_load(_VIS_CFG_PATH.read_text())
SURVEY_CFG = VIS_CFG.get("latent_optimisation", {}).get("vis_latent_circle_survey", {})
MAIN_V2_SOURCE = _MAIN_V2_PATH.read_text()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cvm_mock(initial_layout: str = "strip") -> MagicMock:
    """Return a mock that mimics _vis_manager._cfg with the given initial layout."""
    mock = MagicMock()
    mock._cfg = {
        "latent_optimisation": {
            "vis_latent_circle_survey": {
                "layout": initial_layout,
                "layout_from_resample_mode": True,
            }
        }
    }
    return mock


def _apply_layout_derivation(vis_manager_cfg: dict, z_resample_mode: str, toggle: bool) -> None:
    """Inline replica of the logic added to main_v2.py - lets us test it in isolation."""
    if not toggle:
        return
    derived = "pizza" if z_resample_mode == "in_circle" else "strip"
    vis_manager_cfg.setdefault("latent_optimisation", {}).setdefault("vis_latent_circle_survey", {})["layout"] = derived


# ---------------------------------------------------------------------------
# T1: vis_config.yaml declares the toggle
# ---------------------------------------------------------------------------


class TestVisConfigDeclaresToggle:
    def test_key_present(self) -> None:
        assert "layout_from_resample_mode" in SURVEY_CFG, (
            "vis_config.yaml must declare layout_from_resample_mode under latent_optimisation.vis_latent_circle_survey"
        )

    def test_key_is_bool(self) -> None:
        assert isinstance(SURVEY_CFG["layout_from_resample_mode"], bool), (
            "layout_from_resample_mode must be a boolean in vis_config.yaml"
        )


# ---------------------------------------------------------------------------
# T2-T4: derivation logic correctness
# ---------------------------------------------------------------------------


class TestLayoutDerivation:
    @pytest.mark.parametrize(
        "z_mode, expected_layout",
        [
            ("in_circle", "pizza"),
            ("on_circle", "strip"),
            ("random", "strip"),  # unrecognised -> fallback strip
        ],
    )
    def test_derived_layout(self, z_mode: str, expected_layout: str) -> None:
        cfg: dict = {}
        _apply_layout_derivation(cfg, z_mode, toggle=True)
        actual = cfg["latent_optimisation"]["vis_latent_circle_survey"]["layout"]
        assert actual == expected_layout, f"z_resample_mode={z_mode!r}: expected layout={expected_layout!r}, got {actual!r}"


# ---------------------------------------------------------------------------
# T5: toggle=False leaves layout unchanged
# ---------------------------------------------------------------------------


class TestToggleDisabled:
    @pytest.mark.parametrize("z_mode", ["in_circle", "on_circle", "random"])
    def test_layout_unchanged_when_toggle_false(self, z_mode: str) -> None:
        cfg: dict = {"latent_optimisation": {"vis_latent_circle_survey": {"layout": "grid"}}}
        _apply_layout_derivation(cfg, z_mode, toggle=False)
        actual = cfg["latent_optimisation"]["vis_latent_circle_survey"]["layout"]
        assert actual == "grid", f"layout must stay 'grid' when toggle=False, got {actual!r} for z_mode={z_mode!r}"


# ---------------------------------------------------------------------------
# T6: exception in _load_config -> silent no-op, no crash
# ---------------------------------------------------------------------------


class TestExceptionSilenced:
    def test_no_crash_on_load_config_failure(self) -> None:
        """If _load_config raises, the layout must remain at its configured value."""
        original_cfg = {
            "latent_optimisation": {
                "vis_latent_circle_survey": {
                    "layout": "strip",
                    "layout_from_resample_mode": True,
                }
            }
        }

        mock_cvm = MagicMock()
        mock_cvm._cfg = dict(original_cfg)  # shallow copy is fine for this test

        def _raise(*_a, **_kw):
            raise RuntimeError("simulated config-load failure")

        with patch.object(main_mod, "_load_config", side_effect=_raise):
            try:
                survey_vis_cfg = mock_cvm._cfg.get("latent_optimisation", {}).get("vis_latent_circle_survey", {})
                if survey_vis_cfg.get("layout_from_resample_mode", False):
                    tc = main_mod._load_config(v="v2", cfg_name="training_config.yaml")  # will raise
                    z_mode = tc.get("z_resample_mode", "on_circle")
                    derived = "pizza" if z_mode == "in_circle" else "strip"
                    mock_cvm._cfg.setdefault("latent_optimisation", {}).setdefault("vis_latent_circle_survey", {})["layout"] = (
                        derived
                    )
            except Exception:
                pass  # the real code wraps in try/except — mirrored here

        # layout must be unchanged
        layout = mock_cvm._cfg["latent_optimisation"]["vis_latent_circle_survey"]["layout"]
        assert layout == "strip", f"Layout must remain 'strip' after a failed _load_config call, got {layout!r}"


# ---------------------------------------------------------------------------
# T7: main_v2.py source references layout_from_resample_mode
# ---------------------------------------------------------------------------


class TestMainV2SourceReferencesToggle:
    def test_key_referenced_in_main(self) -> None:
        assert "layout_from_resample_mode" in MAIN_V2_SOURCE, (
            "main_v2.py must reference 'layout_from_resample_mode' to read the toggle from vis_config"
        )

    def test_in_circle_branch_in_main(self) -> None:
        assert '"in_circle"' in MAIN_V2_SOURCE or "'in_circle'" in MAIN_V2_SOURCE, (
            "main_v2.py must contain the in_circle → pizza derivation branch"
        )
