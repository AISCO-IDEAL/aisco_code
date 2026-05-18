"""Pytest configuration for WP1 CDF-AI tests.

Adds the GAOT3 root to sys.path so that validation_calculator.py's internal
`from src.X import Y` imports resolve when running tests from repo root.
"""

import sys
from pathlib import Path

_GAOT3_ROOT = Path(__file__).resolve().parents[2] / "subprojects" / "wp1_cfd_ai" / "AISCO_GAOT" / "GAOT3"
sys.path.insert(0, str(_GAOT3_ROOT))
