"""TDD tests for WP7 GUI FastAPI/Uvicorn migration."""

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


def test_wp7_v2_exposes_fastapi_app_with_required_routes() -> None:
    """The WP7 v2 server must expose a FastAPI app with parity routes."""
    app = app_v2.app
    assert isinstance(app, FastAPI)

    route_paths = {getattr(route, "path", "") for route in app.routes}
    missing = EXPECTED_ENDPOINTS - route_paths
    assert not missing, f"Missing expected API routes: {sorted(missing)}"
