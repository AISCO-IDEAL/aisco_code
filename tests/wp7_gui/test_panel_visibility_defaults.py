"""TDD tests for WP7 GUI v3 panel visibility defaults loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subprojects.wp7_gui.v3.shared.panel_visibility import load_panel_visibility_defaults


def test_load_panel_visibility_defaults_reads_single_file(tmp_path: Path) -> None:
    """Loader should parse one JSON file and return the same shown/hidden config shape."""
    config_path = tmp_path / "panel_visibility_defaults.json"
    expected = {
        "discovered_default_visible": False,
        "viewer": {
            "runs": {
                "panel_cp_field": {"label": "Cp Field", "defaultVisible": True},
                "panel_mesh_shape": {"label": "Aeroshell Mesh", "defaultVisible": True},
                "panel_mesh_shape_glb": {"label": "Aeroshell Mesh GLB", "defaultVisible": False},
            }
        },
        "launcher": {
            "completed": {
                "panel_cp_field": {"label": "Cp Field", "defaultVisible": True},
                "panel_mesh_shape": {"label": "Aeroshell Mesh", "defaultVisible": True},
                "panel_solar_placement": {"label": "Solar Placement", "defaultVisible": False},
                "panel_aeroshell_evolution": {"label": "Aeroshell Evolution", "defaultVisible": False},
            },
            "live": {
                "panel_cp_field": {"label": "Cp Field", "defaultVisible": True},
                "panel_mesh_shape": {"label": "Aeroshell Mesh", "defaultVisible": True},
                "panel_mesh_shape_glb": {"label": "Aeroshell Mesh GLB", "defaultVisible": False},
            },
        },
    }
    config_path.write_text(json.dumps(expected), encoding="utf-8")

    assert load_panel_visibility_defaults(config_path) == expected


def test_load_panel_visibility_defaults_rejects_ignored_block(tmp_path: Path) -> None:
    """Loader should reject ignored-panel behavior in this config model."""
    config_path = tmp_path / "panel_visibility_defaults.json"
    config_path.write_text(
        json.dumps(
            {
                "discovered_default_visible": False,
                "viewer": {"runs": {}},
                "launcher": {"completed": {}, "live": {}},
                "ignored": ["foo"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ignored"):
        load_panel_visibility_defaults(config_path)


def test_load_panel_visibility_defaults_expands_all_modes(tmp_path: Path) -> None:
    """Loader should let one defaults block drive viewer and launcher/live modes."""
    config_path = tmp_path / "panel_visibility_defaults.json"
    all_modes = {
        "panel_cp_field": {"label": "Cp Field", "defaultVisible": True},
        "panel_mesh_shape": {"label": "Aeroshell Mesh", "defaultVisible": True},
        "panel_mesh_shape_glb": {"label": "Aeroshell Mesh GLB", "defaultVisible": False},
        "panel_solar_placement": {"label": "Solar Placement", "defaultVisible": False},
        "panel_aeroshell_evolution": {"label": "Aeroshell Evolution", "defaultVisible": False},
    }
    config_path.write_text(
        json.dumps(
            {
                "discovered_default_visible": False,
                "all_modes": all_modes,
            }
        ),
        encoding="utf-8",
    )

    result = load_panel_visibility_defaults(config_path)

    assert result["viewer"]["runs"] == all_modes
    assert result["launcher"]["completed"] == all_modes
    assert result["launcher"]["live"] == all_modes
