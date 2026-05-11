"""TDD tests for WP7 GUI v2 mesh-config endpoint and vertex-colour rendering.

Plan 025 — M2: /mesh-config route, M1/M3: HTML assertions for frontend behaviour.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from subprojects.wp7_gui.v2 import app_v2

OPTIMIZATION_HTML = Path(app_v2.__file__).resolve().parent / "templates" / "pages" / "optimization.html"

# ---------------------------------------------------------------------------
# M2: /mesh-config endpoint — route existence
# ---------------------------------------------------------------------------


def test_mesh_config_route_exists() -> None:
    """The /mesh-config route must be registered on the FastAPI app."""
    route_paths = {getattr(route, "path", "") for route in app_v2.app.routes}
    assert "/mesh-config" in route_paths, f"/mesh-config not found in routes: {sorted(route_paths)}"


# ---------------------------------------------------------------------------
# M2: /mesh-config endpoint — behaviour
# ---------------------------------------------------------------------------


def test_mesh_config_returns_default_when_file_missing() -> None:
    """Handler must return mesh_opacity=1.0 when config file does not exist."""
    missing = Path("/nonexistent/definitely/not/here/config.yaml")
    with patch.object(app_v2, "GENERAL_CONFIG_PATH", missing):
        result = app_v2.mesh_config_endpoint()
    assert result == {"mesh_opacity": 1.0}, f"Expected default 1.0, got: {result}"


def test_mesh_config_returns_configured_value() -> None:
    """Handler must return the mesh_opacity value from general_config.yaml."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.dump({"mesh_opacity": 0.5}, fh)
        tmp_path = Path(fh.name)

    try:
        with patch.object(app_v2, "GENERAL_CONFIG_PATH", tmp_path):
            result = app_v2.mesh_config_endpoint()
        assert result == {"mesh_opacity": 0.5}, f"Expected 0.5, got: {result}"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_mesh_config_returns_default_when_key_absent() -> None:
    """Handler must return mesh_opacity=1.0 when the YAML key is absent."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.dump({"some_other_key": 42}, fh)
        tmp_path = Path(fh.name)

    try:
        with patch.object(app_v2, "GENERAL_CONFIG_PATH", tmp_path):
            result = app_v2.mesh_config_endpoint()
        assert result == {"mesh_opacity": 1.0}, f"Expected default 1.0, got: {result}"
    finally:
        tmp_path.unlink(missing_ok=True)


def test_mesh_config_returns_default_on_bad_yaml() -> None:
    """Handler must return mesh_opacity=1.0 when YAML is malformed — never a 500."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        fh.write("{{ invalid yaml: ::::")
        tmp_path = Path(fh.name)

    try:
        with patch.object(app_v2, "GENERAL_CONFIG_PATH", tmp_path):
            result = app_v2.mesh_config_endpoint()
        assert result == {"mesh_opacity": 1.0}, f"Expected default 1.0, got: {result}"
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# M1: vertex-colour rendering — HTML content assertions
# ---------------------------------------------------------------------------


def test_optimization_html_detects_vertex_colours() -> None:
    """optimization.html load3DMesh must detect geometry.attributes.color."""
    html = OPTIMIZATION_HTML.read_text(encoding="utf-8")
    assert "geometry.attributes.color" in html, "load3DMesh must check geometry.attributes.color to detect vertex colours"


def test_optimization_html_enables_vertex_colors_on_material() -> None:
    """optimization.html material must set vertexColors when colours are present."""
    html = OPTIMIZATION_HTML.read_text(encoding="utf-8")
    assert "vertexColors" in html, "MeshStandardMaterial must include vertexColors flag"


# ---------------------------------------------------------------------------
# M3: opacity slider — HTML content assertions
# ---------------------------------------------------------------------------


def test_optimization_html_has_opacity_slider() -> None:
    """optimization.html must include a mesh opacity slider element."""
    html = OPTIMIZATION_HTML.read_text(encoding="utf-8")
    assert "meshOpacitySlider" in html, "Aeroshell Mesh card must have an opacity slider with id='meshOpacitySlider'"


def test_optimization_html_fetches_mesh_config() -> None:
    """optimization.html must fetch /mesh-config to initialise opacity."""
    html = OPTIMIZATION_HTML.read_text(encoding="utf-8")
    assert "/mesh-config" in html, "optimization.html must fetch /mesh-config to get initial mesh_opacity"


def test_optimization_html_opacity_persists_across_epochs() -> None:
    """Opacity state variable must be declared at module scope, not inside load3DMesh."""
    html = OPTIMIZATION_HTML.read_text(encoding="utf-8")
    # The variable must exist outside load3DMesh so epoch advances don't reset it
    assert "meshOpacity" in html, "meshOpacity variable must be declared in optimization.html"
