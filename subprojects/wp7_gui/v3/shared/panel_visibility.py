"""Panel visibility defaults loader for WP7 GUI v3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_panel_visibility_defaults(config_path: Path) -> dict[str, Any]:
    """Load shown/hidden default panel configuration from a single JSON file.

    Args:
        config_path: Absolute or relative path to panel defaults JSON.

    Returns:
        Parsed and validated panel visibility defaults dictionary.

    Raises:
        TypeError: If the file content contains invalid types.
        ValueError: If the file content is invalid for the shown/hidden model.
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("panel visibility defaults must be a JSON object")

    if "ignored" in data:
        raise ValueError("'ignored' behavior is not supported in panel visibility defaults")

    discovered_default = data.get("discovered_default_visible")
    if not isinstance(discovered_default, bool):
        raise TypeError("'discovered_default_visible' must be a boolean")

    # Preferred format: one all-modes block that is expanded into viewer/launcher.
    if "all_modes" in data:
        all_modes = data.get("all_modes")
        if not isinstance(all_modes, dict):
            raise ValueError("'all_modes' must be an object")
        return {
            "discovered_default_visible": discovered_default,
            "viewer": {"runs": all_modes},
            "launcher": {"completed": all_modes, "live": all_modes},
        }

    viewer = data.get("viewer")
    launcher = data.get("launcher")
    if not isinstance(viewer, dict) or not isinstance(launcher, dict):
        raise TypeError("either provide 'all_modes' or provide both 'viewer' and 'launcher' objects")

    return data
