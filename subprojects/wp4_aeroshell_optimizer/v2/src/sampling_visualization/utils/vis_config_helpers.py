"""Helpers for reading and querying viz_config.yaml visual_types entries."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def vis_type_config(visual_types: dict[str, Any], visual_type: str) -> dict[str, Any] | None:
    """Return the config dict for visual_type, or None if disabled/missing.

    Walks the slash-separated path (e.g. 'aero/cp_field') through the nested
    visual_types dict.  If no exact match is found, walks up the ancestors and
    inherits the first parent that has an 'enabled' key — this is how
    'mesh_shape/slot_00' inherits the 'mesh_shape' config.
    """
    def _get(key_path: str) -> dict[str, Any] | None:
        parts = key_path.split("/")
        node: Any = visual_types
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node if isinstance(node, dict) else None

    cfg = _get(visual_type)
    if cfg is None:
        parts = visual_type.split("/")
        for depth in range(len(parts) - 1, 0, -1):
            parent = _get("/".join(parts[:depth]))
            if parent is not None and "enabled" in parent:
                cfg = parent
                break

    if cfg is None or not cfg.get("enabled", False):
        return None
    return cfg


def vis_is_enabled(visual_types: dict[str, Any], visual_type: str) -> bool:
    return vis_type_config(visual_types, visual_type) is not None


def resolve_bool(per_type_val: bool | None, global_val: bool | None) -> bool:
    if per_type_val is not None:
        return bool(per_type_val)
    if global_val is not None:
        return bool(global_val)
    return False


def collect_gif_dirs(visual_types: dict[str, Any], run_dir: Path, written_dirs: set[Path]) -> list[Path]:
    """Return dirs under run_dir that contain epoch_XXXXX.ply files and should get GIFs."""
    flat_pattern = re.compile(r"^epoch_(\d{5})\.ply$")

    def _enabled_paths(node: dict[str, Any], prefix: str) -> list[str]:
        paths: list[str] = []
        for key, val in node.items():
            full = f"{prefix}/{key}" if prefix else key
            if isinstance(val, dict):
                if "enabled" in val:
                    if val.get("enabled", False):
                        paths.append(full)
                else:
                    paths.extend(_enabled_paths(val, full))
        return paths

    candidates = [run_dir / vt for vt in _enabled_paths(visual_types, "")]
    for d in written_dirs:
        if d not in candidates:
            candidates.append(d)

    return [
        d for d in candidates
        if d.is_dir() and any(flat_pattern.match(f.name) for f in d.iterdir())
    ]
