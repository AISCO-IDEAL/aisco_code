"""Shared artifact-serving API endpoints.

Mounted at /api so both launcher and viewer pages can fetch
/api/artifacts/file and /api/artifacts/slow-gif with the same URLs.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared import gif_serving
from subprojects.wp7_gui.v3.shared.run_paths import default_storage_root

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/artifacts/file")
def artifact_file(path: str = "") -> FileResponse:
    return gif_serving.serve_file(default_storage_root(), path)


@router.get("/artifacts/slow-gif")
def artifact_slow_gif(path: str = "", factor: float = 2.5) -> FileResponse:
    return gif_serving.serve_slow_gif(default_storage_root(), path, factor, SETTINGS.cache_dir)
