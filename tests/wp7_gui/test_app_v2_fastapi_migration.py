"""TDD tests for WP7 GUI FastAPI/Uvicorn migration."""

from pathlib import Path

from fastapi import FastAPI

from subprojects.wp7_gui.v2 import app_v2

EXPECTED_ENDPOINTS = {
    "/run-local",
    "/run-local/logs",
    "/run-local/progress",
    "/run-local/artifacts",
    "/run-local/artifact-file",
    "/run-local/slow-gif",
}

TEMPLATES_DIR = Path(app_v2.__file__).resolve().parent / "templates"


def test_wp7_v2_exposes_fastapi_app_with_required_routes() -> None:
    """The WP7 v2 server must expose a FastAPI app with parity routes."""
    app = app_v2.app
    assert isinstance(app, FastAPI)

    route_paths = {getattr(route, "path", "") for route in app.routes}
    missing = EXPECTED_ENDPOINTS - route_paths
    assert not missing, f"Missing expected API routes: {sorted(missing)}"


def test_index_uses_top_logo_bar_and_separate_sidebar() -> None:
    """The shell should contain one top logo bar and a separate sidebar for navigation."""
    index_html = (TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")

    assert '<header class="top-bar"' in index_html
    assert '<div class="top-brand"' in index_html
    assert '<aside class="sidebar"' in index_html
    assert '<div class="brand"' not in index_html


def test_page_templates_do_not_repeat_logo_strip() -> None:
    """Inner pages should not render logo strips because shell owns the top bar."""
    optimization_html = (TEMPLATES_DIR / "pages" / "optimization.html").read_text(encoding="utf-8")
    solar_html = (TEMPLATES_DIR / "pages" / "solar.html").read_text(encoding="utf-8")
    logs_html = (TEMPLATES_DIR / "pages" / "logs.html").read_text(encoding="utf-8")

    assert "logo-container" not in optimization_html
    assert 'id="logo1"' not in optimization_html
    assert 'id="logo2"' not in optimization_html
    assert 'id="logo3"' not in optimization_html
    assert "/logo.png" not in optimization_html
    assert "/ideal.png" not in optimization_html
    assert "/aCe.png" not in optimization_html

    assert "/logo.png" not in solar_html
    assert "/ideal.png" not in solar_html
    assert "/aCe.png" not in solar_html
    assert "/logo.png" not in logs_html
    assert "/ideal.png" not in logs_html
    assert "/aCe.png" not in logs_html
