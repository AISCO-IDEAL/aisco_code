"""Shared artifact-serving API endpoints.

Mounted at /api so both launcher and viewer pages can fetch
/api/artifacts/file and /api/artifacts/slow-gif with the same URLs.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared import gif_serving
from subprojects.wp7_gui.v3.shared.run_paths import get_storage_root

router = APIRouter(prefix="/api", tags=["api"])


def _storage_root():
    return get_storage_root(
        SETTINGS.general_config_path,
        SETTINGS.default_local_storage_dir,
        SETTINGS.default_bulk_storage_dir,
        SETTINGS.repo_root.parent,
    )


@router.get("/artifacts/file")
def artifact_file(path: str = "") -> FileResponse:
    return gif_serving.serve_file(_storage_root(), path)


@router.get("/artifacts/slow-gif")
def artifact_slow_gif(path: str = "", factor: float = 2.5) -> FileResponse:
    return gif_serving.serve_slow_gif(_storage_root(), path, factor, SETTINGS.cache_dir)
