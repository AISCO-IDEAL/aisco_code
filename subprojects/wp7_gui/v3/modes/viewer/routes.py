"""Viewer mode router — page views + API endpoints for finished runs."""

from __future__ import annotations

from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates

from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared.artifact_index import list_artifacts
from subprojects.wp7_gui.v3.shared.pretraining import PretrainingService
from subprojects.wp7_gui.v3.shared.run_paths import get_storage_root, list_all_runs

router = APIRouter(prefix="/viewer", tags=["viewer"])
_templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))
_pretraining = PretrainingService(SETTINGS.pretraining_storage_root)


def _storage_root():
    return get_storage_root(
        SETTINGS.general_config_path,
        SETTINGS.default_local_storage_dir,
        SETTINGS.default_bulk_storage_dir,
        SETTINGS.repo_root.parent,
    )


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@router.get("")
def viewer_index(request: Request):
    return _templates.TemplateResponse(
        request,
        "viewer/index.html",
        {
            "active_mode": "viewer",
            "active_page": "runs",
            "page_title": "Viewer — Runs",
        },
    )


@router.get("/pretraining")
def viewer_pretraining(request: Request):
    return _templates.TemplateResponse(
        request,
        "viewer/pretraining.html",
        {
            "active_mode": "viewer",
            "active_page": "pretraining",
            "page_title": "Viewer — Pretraining GIFs",
        },
    )


# ---------------------------------------------------------------------------
# Runs API
# ---------------------------------------------------------------------------
@router.get("/api/runs")
def api_list_runs() -> dict[str, Any]:
    storage_root = _storage_root()
    runs = list_all_runs(storage_root)
    return {"storage_root": str(storage_root), "runs": runs}


@router.get("/api/runs/{run_id:path}/artifacts")
def api_run_artifacts(run_id: str) -> dict[str, Any]:
    """List artifacts for a specific finished run.

    run_id is the URL-encoded relative path from storage_root:
    e.g.  model_hash%2Fwandb_run_name
    """
    storage_root = _storage_root()
    run_dir = (storage_root / unquote(run_id)).resolve()
    try:
        run_dir.relative_to(storage_root.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail={"message": "run_id is outside storage root"}) from None
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail={"message": "Run directory not found"})
    return list_artifacts(run_dir, storage_root)


# ---------------------------------------------------------------------------
# Pretraining API
# ---------------------------------------------------------------------------
@router.get("/api/pretraining/gifs")
def api_pretraining_gifs() -> dict[str, Any]:
    return _pretraining.list_gifs()


@router.get("/api/pretraining/gif-file")
def api_pretraining_gif_file(path: str = "") -> FileResponse:
    return _pretraining.get_gif_response(path)
