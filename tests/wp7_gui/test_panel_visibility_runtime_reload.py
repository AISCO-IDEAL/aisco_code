"""TDD tests for runtime reloading of panel visibility defaults."""

from __future__ import annotations

from fastapi.testclient import TestClient

from subprojects.wp7_gui.v3.app_v3 import app
from subprojects.wp7_gui.v3.modes.launcher import routes as launcher_routes
from subprojects.wp7_gui.v3.modes.viewer import routes as viewer_routes


def test_launcher_output_reloads_panel_defaults_per_request(monkeypatch) -> None:
    """Launcher output route should load panel defaults for each request."""
    call_count = {"n": 0}

    def fake_loader(_path):
        call_count["n"] += 1
        return {
            "discovered_default_visible": False,
            "viewer": {"runs": {}},
            "launcher": {"completed": {}, "live": {}},
        }

    monkeypatch.setattr(launcher_routes, "load_panel_visibility_defaults", fake_loader)
    client = TestClient(app)

    response = client.get("/launcher/output?pipeline=live")

    assert response.status_code == 200
    assert call_count["n"] == 1


def test_viewer_index_reloads_panel_defaults_per_request(monkeypatch) -> None:
    """Viewer index route should load panel defaults for each request."""
    call_count = {"n": 0}

    def fake_loader(_path):
        call_count["n"] += 1
        return {
            "discovered_default_visible": False,
            "viewer": {"runs": {}},
            "launcher": {"completed": {}, "live": {}},
        }

    monkeypatch.setattr(viewer_routes, "load_panel_visibility_defaults", fake_loader)
    client = TestClient(app)

    response = client.get("/viewer")

    assert response.status_code == 200
    assert call_count["n"] == 1
