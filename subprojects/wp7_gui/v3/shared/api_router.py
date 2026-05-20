"""Shared artifact-serving API endpoints.

Mounted at /api so both launcher and viewer pages can fetch
/api/artifacts/file and /api/artifacts/slow-gif with the same URLs.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse

from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared import gif_serving
from subprojects.wp7_gui.v3.shared.run_paths import default_storage_root
from subprojects.wp7_gui.v3.shared.theme import get_gui_theme, get_mesh_assembly_config, load_vis_config

router = APIRouter(prefix="/api", tags=["api"])


@router.get("/artifacts/file")
def artifact_file(path: str = "") -> FileResponse:
    return gif_serving.serve_file(default_storage_root(), path)


@router.get("/artifacts/slow-gif")
def artifact_slow_gif(path: str = "", factor: float = 2.5) -> FileResponse:
    return gif_serving.serve_slow_gif(default_storage_root(), path, factor, SETTINGS.cache_dir)


@router.get("/artifacts/glb-config")
def artifact_glb_config() -> JSONResponse:
    vis_cfg = load_vis_config()
    assembly_cfg = get_mesh_assembly_config(vis_cfg)
    return JSONResponse(assembly_cfg)


@router.get("/theme-config")
def theme_config() -> JSONResponse:
    vis_cfg = load_vis_config()
    return JSONResponse(get_gui_theme(vis_cfg))
