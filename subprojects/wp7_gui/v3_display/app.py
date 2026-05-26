"""WP7 GUI v3_display entrypoint.

Presentation-only app: one page with the run-name leaderboard (no metrics)
and an mp4 video player.

Run from repo root:
    python3 subprojects/wp7_gui/v3_display/app.py

Before running, set VIDEO_PATH in core/settings.py to point at your video file.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Ensure repo root is importable when run as a script.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from subprojects.wp7_gui.v3.shared.theme import build_template_theme_context  # noqa: E402
from subprojects.wp7_gui.v3_display.core.settings import (  # noqa: E402
    HOST,
    LEADERBOARD_CSV,
    LEADERBOARD_REFRESH_MS,
    PINNED_TOP_COUNT,
    PORT,
    ROTATING_WINDOW_SIZE,
    SPONSOR_LOGO_MAX_WIDTH_PERCENT,
    SPONSOR_LOGO_URLS,
    STATIC_DIR,
    TEMPLATES_DIR,
    VIDEO_PATH,
    WINDOW_ADVANCE_MS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="AISCO Display")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _clamp_percent(value: int) -> int:
    return max(1, min(100, value))


def _resolve_static_path(static_url: str) -> Path | None:
    if not static_url.startswith("/static/"):
        return None
    relative = static_url.removeprefix("/static/")
    return (STATIC_DIR / relative).resolve()


def _build_sponsor_logos() -> list[dict[str, Any]]:
    sponsor_logos: list[dict[str, Any]] = []
    for idx, logo_url in enumerate(SPONSOR_LOGO_URLS):
        static_path = _resolve_static_path(logo_url)
        exists = bool(static_path and static_path.exists())
        sponsor_logos.append(
            {
                "idx": idx,
                "url": logo_url,
                "exists": exists,
                "expected_path": str(static_path) if static_path else "",
                "filename": Path(logo_url).name,
            }
        )
    return sponsor_logos

# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------


@app.get("/")
def root(request: Request) -> Response:
    theme_ctx = build_template_theme_context()
    return templates.TemplateResponse(
        request,
        "display.html",
        {
            "page_title": "AISCO — Live Leaderboard",
            "leaderboard_refresh_ms": LEADERBOARD_REFRESH_MS,
            "pinned_top_count": PINNED_TOP_COUNT,
            "rotating_window_size": ROTATING_WINDOW_SIZE,
            "window_advance_ms": WINDOW_ADVANCE_MS,
            "video_available": VIDEO_PATH.exists(),
            "sponsor_logos": _build_sponsor_logos(),
            "sponsor_logo_max_width_percent": _clamp_percent(SPONSOR_LOGO_MAX_WIDTH_PERCENT),
            **theme_ctx,
        },
    )


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "video_path": str(VIDEO_PATH),
            "sponsor_logo_urls": list(SPONSOR_LOGO_URLS),
        }
    )


# ---------------------------------------------------------------------------
# Leaderboard API
# ---------------------------------------------------------------------------


def _parse_float_cell(raw: str | None) -> float | None:
    if not raw:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    with contextlib.suppress(ValueError):
        return float(stripped)
    return None


@app.get("/api/leaderboard")
def api_leaderboard() -> JSONResponse:
    """Return all run names sorted by cruising_speed_ms descending.

    Returns:
        JSON with:
        - ``entries``: list of dicts with ``rank`` and ``run_id`` only.
        - ``total_runs``: total number of runs in the CSV.
    """
    rows: list[dict[str, str]] = []
    if LEADERBOARD_CSV.exists():
        try:
            with LEADERBOARD_CSV.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except OSError as exc:
            LOGGER.warning("Could not read leaderboard CSV %s: %s", LEADERBOARD_CSV, exc)

    def _speed(row: dict[str, str]) -> float:
        return _parse_float_cell(row.get("cruising_speed_ms", "")) or 0.0

    sorted_rows = sorted(rows, key=_speed, reverse=True)

    entries: list[dict[str, Any]] = [{"rank": idx + 1, "run_id": row.get("run_id", "")} for idx, row in enumerate(sorted_rows)]

    return JSONResponse({"entries": entries, "total_runs": len(rows)})


# ---------------------------------------------------------------------------
# Video streaming
# ---------------------------------------------------------------------------


@app.get("/video")
def stream_video(request: Request) -> Response:
    """Stream the configured mp4 video with HTTP range support."""
    if not VIDEO_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Video not found at {VIDEO_PATH}")

    file_size = VIDEO_PATH.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse "bytes=start-end"
        range_value = range_header.strip().replace("bytes=", "")
        parts = range_value.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
        end = min(end, file_size - 1)
        chunk_size = end - start + 1

        def _iter_range() -> Any:
            with VIDEO_PATH.open("rb") as fh:
                fh.seek(start)
                remaining = chunk_size
                while remaining > 0:
                    data = fh.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        return StreamingResponse(
            _iter_range(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(chunk_size),
            },
        )

    # No range header — stream the whole file.
    def _iter_full() -> Any:
        with VIDEO_PATH.open("rb") as fh:
            while True:
                data = fh.read(65536)
                if not data:
                    break
                yield data

    return StreamingResponse(
        _iter_full(),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(file_size),
        },
    )


# ---------------------------------------------------------------------------
# Static assets — reuse v3's compiled assets
# ---------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    LOGGER.warning("Static directory not found: %s — CSS/JS will be missing", STATIC_DIR)


def main() -> None:
    print("--- AISCO Display GUI ---")
    print(f"Leaderboard CSV : {LEADERBOARD_CSV}")
    print(
        f"Video           : {VIDEO_PATH} ({'found' if VIDEO_PATH.exists() else 'NOT FOUND — update VIDEO_PATH in core/settings.py'})"
    )
    print(f"Sponsor width   : {SPONSOR_LOGO_MAX_WIDTH_PERCENT}%")
    for logo_url in SPONSOR_LOGO_URLS:
        logo_path = _resolve_static_path(logo_url)
        exists = bool(logo_path and logo_path.exists())
        print(f"Sponsor logo    : {logo_url} ({'found' if exists else 'NOT FOUND'})")
    print(f"Static assets   : {STATIC_DIR}")
    print(f"Templates       : {TEMPLATES_DIR}")
    print(f"Local URL       : http://localhost:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
