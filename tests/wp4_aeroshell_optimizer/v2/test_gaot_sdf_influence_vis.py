from __future__ import annotations

import numpy as np
import pytest

try:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import NullVisManager
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_v2 import CentralVisManagerV2
except Exception as exc:
    pytest.skip(f"central visualization optional dependencies unavailable: {exc}", allow_module_level=True)


def _make_vis_manager_stub() -> CentralVisManagerV2:
    manager = object.__new__(CentralVisManagerV2)
    manager.run_id = "test-run"
    manager._active_slot = None
    manager._visual_types = {
        "aero": {
            "sdf_influence": {
                "enabled": True,
                "colormap": "RdYlGn",
                "interval": 10,
                "alpha_enabled": None,
                "coordinate_system": None,
                "vmin": None,
                "vmax": None,
            }
        }
    }
    return manager


def test_is_due_respects_interval() -> None:
    manager = _make_vis_manager_stub()

    assert manager.is_due("aero/sdf_influence/jvp/raw_gradient", 0)
    assert not manager.is_due("aero/sdf_influence/jvp/raw_gradient", 9)
    assert manager.is_due("aero/sdf_influence/jvp/raw_gradient", 10)


def test_vis_aero_sdf_influence_maps_negative_red_positive_green(monkeypatch) -> None:
    manager = _make_vis_manager_stub()
    captured: dict[str, np.ndarray] = {}

    def fake_submit(visual_type, epoch, verts_np, faces_np, rgb_np, type_cfg):
        captured["visual_type"] = visual_type
        captured["rgb"] = rgb_np

    monkeypatch.setattr(manager, "_submit_ply", fake_submit)

    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    values = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

    manager.vis_aero_sdf_influence(0, "jvp", "raw_gradient", verts, faces, values)

    rgb = captured["rgb"]
    assert captured["visual_type"] == "aero/sdf_influence/jvp/raw_gradient"
    assert rgb[0, 0] > rgb[0, 1], "negative values should be redder than green"
    assert rgb[2, 1] > rgb[2, 0], "positive values should be greener than red"


def test_null_vis_manager_accepts_sdf_influence_api() -> None:
    manager = NullVisManager()

    assert not manager.is_due("aero/sdf_influence/jvp/raw_gradient", 0)
    manager.vis_aero_sdf_influence(0, "jvp", "raw_gradient", [], [], [])
