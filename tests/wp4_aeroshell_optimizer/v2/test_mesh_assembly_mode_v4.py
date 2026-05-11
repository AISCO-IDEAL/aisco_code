"""Tests for MeshAssemblyModeV4 per-camera frame_storage restructuring (Plan 026).

Covers:
- Nested ``frame_storage`` shape after __init__ and after update() calls.
- ``clear_frames()`` override: preserves n_cams empty inner lists (not flat empty list).
- ``num_stored_frames()`` override: returns sum across cameras (not outer-list length).
- ``finalize()`` creates ``cam{idx}/`` sub-directories, not a single root-level GIF.

No GPU, no CadQuery, no kaolin required.

Run::

    pytest tests/wp4_aeroshell_optimizer/v2/test_mesh_assembly_mode_v4.py -v
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

V4_MODULE = "subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.mesh_assembly_mode_v4"
BASE_LOAD_CONFIG = "subprojects.wp4_aeroshell_optimizer.v2.src.visuals.rendering_mode_base_class.load_config"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(n_cams: int) -> dict[str, Any]:
    """Minimal V4 config with *n_cams* numbered camera positions."""
    config: dict[str, Any] = {
        "look_at": [0, 0, 0],
        "render_mode": "rgb",
    }
    for i in range(n_cams):
        config[f"camera_position_{i + 1}"] = [float((i + 1) * 1000), 0.0, 0.0]
    return config


def _make_visualiser() -> MagicMock:
    """Minimal mock Visualiser — no GPU or file I/O required."""
    vis = MagicMock()
    vis.renderer.camera_mode = "look_at"
    vis.stored_frames = []

    def _touch_gif(file_path: str, **_kwargs: object) -> None:
        Path(file_path).touch()

    vis.frames_to_gif.side_effect = _touch_gif
    vis.save_image = MagicMock()
    return vis


def _patched_load_config(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {"rendering_modes": {"enabled": True}}


def _make_mode(n_cams: int) -> Any:
    """Instantiate MeshAssemblyModeV4 with *n_cams* cameras; all heavy deps mocked."""
    mod = importlib.import_module(V4_MODULE)
    config = _make_config(n_cams)
    vis = _make_visualiser()
    with patch(BASE_LOAD_CONFIG, side_effect=_patched_load_config):
        mode = mod.MeshAssemblyModeV4(vis, config)
    return mode


def _prime_base_frames(mode: Any) -> None:
    """Pre-fill base_frames so update() skips the heavy _build_base_frame() call."""
    n = len(mode.camera_positions)
    mode.base_frames = [np.zeros((4, 4, 3), dtype=np.float32) for _ in range(n)]
    mode.base_assembly_ids = [1] * n


def _inject_frames(mode: Any, n_frames: int) -> None:
    """Directly inject *n_frames* synthetic frames into the nested frame_storage.

    Used in finalize() tests to bypass update() rendering.
    Will silently no-op if frame_storage is not yet nested (pre-implementation).
    """
    frame = np.zeros((4, 4, 3), dtype=np.float32)
    for inner in mode.frame_storage:
        if isinstance(inner, list):
            inner.extend([frame.copy() for _ in range(n_frames)])


# ---------------------------------------------------------------------------
# Milestone 1 — frame_storage shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cams", [1, 3])
class TestFrameStorageShape:
    def test_frame_storage_is_nested_after_init(self, n_cams: int) -> None:
        """frame_storage must be a list of n_cams empty inner lists after __init__."""
        mode = _make_mode(n_cams)

        assert isinstance(mode.frame_storage, list), "frame_storage must be a list"
        assert len(mode.frame_storage) == n_cams, f"Expected outer length {n_cams}, got {len(mode.frame_storage)}"
        for cam_idx, inner in enumerate(mode.frame_storage):
            assert isinstance(inner, list), f"frame_storage[{cam_idx}] must be a list, got {type(inner)}"
            assert inner == [], f"frame_storage[{cam_idx}] must be empty after init, got {inner}"

    def test_frame_storage_nested_after_update(self, n_cams: int) -> None:
        """After N updates, each inner list has exactly N frames."""
        mode = _make_mode(n_cams)
        _prime_base_frames(mode)

        n_updates = 4
        for i in range(n_updates):
            mode.update({"assembly_id": 1, "iteration": i})

        assert len(mode.frame_storage) == n_cams, f"Outer list must have {n_cams} slots, got {len(mode.frame_storage)}"
        for cam_idx, inner in enumerate(mode.frame_storage):
            assert isinstance(inner, list), f"frame_storage[{cam_idx}] must be a list"
            assert len(inner) == n_updates, f"cam{cam_idx}: expected {n_updates} frames, got {len(inner)}"

    def test_clear_frames_preserves_structure(self, n_cams: int) -> None:
        """clear_frames() must produce n_cams empty inner lists, not a flat []."""
        mode = _make_mode(n_cams)
        _prime_base_frames(mode)
        for i in range(3):
            mode.update({"assembly_id": 1, "iteration": i})

        mode.clear_frames()

        assert len(mode.frame_storage) == n_cams, (
            f"After clear_frames(), outer list must still have {n_cams} slots, got {len(mode.frame_storage)}"
        )
        for cam_idx, inner in enumerate(mode.frame_storage):
            assert isinstance(inner, list), f"frame_storage[{cam_idx}] must still be a list after clear_frames()"
            assert inner == [], f"frame_storage[{cam_idx}] must be empty after clear_frames(), got {inner}"

    def test_num_stored_frames_returns_sum(self, n_cams: int) -> None:
        """num_stored_frames() returns total frames summed across all cameras."""
        mode = _make_mode(n_cams)
        _prime_base_frames(mode)

        n_updates = 5
        for i in range(n_updates):
            mode.update({"assembly_id": 1, "iteration": i})

        expected_total = n_cams * n_updates
        actual = mode.num_stored_frames()
        assert actual == expected_total, (
            f"Expected {expected_total} total frames, got {actual}. "
            f"num_stored_frames() must sum across cameras, not return outer list length."
        )


# ---------------------------------------------------------------------------
# Milestone 2 — finalize() per-camera subfolders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_cams", [1, 3])
class TestFinalizeSubfolders:
    def test_finalize_creates_cam_prefixed_gifs(self, tmp_path: Path, n_cams: int) -> None:
        """finalize() must write cam{idx}_<name>.gif directly in output_dir (no subdirs)."""
        mode = _make_mode(n_cams)
        _inject_frames(mode, n_frames=3)

        mode.finalize(tmp_path, file_name="test.gif")

        for idx in range(n_cams):
            expected = tmp_path / f"cam{idx}_test.gif"
            assert expected.is_file(), f"cam{idx}_test.gif not found; output_dir contains: {list(tmp_path.iterdir())}"

    def test_finalize_no_cam_subdirs(self, tmp_path: Path, n_cams: int) -> None:
        """finalize() must NOT create cam{idx}/ sub-directories."""
        mode = _make_mode(n_cams)
        _inject_frames(mode, n_frames=3)

        mode.finalize(tmp_path, file_name="test.gif")

        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert subdirs == [], f"Unexpected sub-directories: {subdirs}. GIFs must be written flat into output_dir."

    def test_finalize_empty_storage_produces_no_output(self, tmp_path: Path, n_cams: int) -> None:
        """finalize() with all-empty inner lists must produce no output at all."""
        mode = _make_mode(n_cams)
        # frame_storage is [[], [], ...] — no frames injected

        mode.finalize(tmp_path, file_name="test.gif")

        created = list(tmp_path.iterdir())
        assert created == [], f"Expected empty output_dir, found: {created}"
