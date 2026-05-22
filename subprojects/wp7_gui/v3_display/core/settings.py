"""Settings for WP7 GUI v3_display.

Two constants are the primary things to adjust before a presentation:

    VIDEO_PATH      — absolute path to the .mp4 video file to show.
    LEADERBOARD_CSV — absolute path to the gui_runs_summary.csv produced by v3.
                      Defaults to v3/logs/gui_runs_summary.csv (sibling folder).

Everything else (host, port, static/template dirs) is derived automatically.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# ── ADJUST THESE BEFORE THE PRESENTATION ───────────────────────────────────
# ---------------------------------------------------------------------------

# Path to the .mp4 video file to display in the player widget.
VIDEO_PATH: Path = Path("/cluster/home/roarnold/AISCO_code/subprojects/wp7_gui/v3_display/video/gui_video.mp4")

# ---------------------------------------------------------------------------
# ── Derived paths (no need to change) ──────────────────────────────────────
# ---------------------------------------------------------------------------

_DISPLAY_ROOT: Path = Path(__file__).resolve().parents[1]
_V3_ROOT: Path = _DISPLAY_ROOT.parent / "v3"

# Leaderboard data — read from v3's live summary log by default.
LEADERBOARD_CSV: Path = _V3_ROOT / "logs" / "gui_runs_summary.csv"

# Reuse v3's compiled static assets (CSS, JS, images) directly.
STATIC_DIR: Path = _V3_ROOT / "frontend" / "static"

TEMPLATES_DIR: Path = _DISPLAY_ROOT / "frontend" / "templates"

HOST: str = "0.0.0.0"
PORT: int = 8005

# Leaderboard auto-refresh interval in milliseconds.
LEADERBOARD_REFRESH_MS: int = 5_000
