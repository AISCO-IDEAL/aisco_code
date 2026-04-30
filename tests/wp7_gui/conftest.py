"""Pytest configuration for WP7 GUI tests.

Adds the wp7_gui/v2/ directory to sys.path so that app_v2.py's internal
`from functions.X import Y` imports resolve when running tests from repo root.
"""

import sys
from pathlib import Path

_V2_ROOT = Path(__file__).resolve().parents[2] / "subprojects" / "wp7_gui" / "v2"
sys.path.insert(0, str(_V2_ROOT))
