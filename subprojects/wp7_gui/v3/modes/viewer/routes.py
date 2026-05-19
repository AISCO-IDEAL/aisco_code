"""Viewer mode router — page views + API endpoints for finished runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared.artifact_index import list_artifacts
from subprojects.wp7_gui.v3.shared.panel_visibility import load_panel_visibility_defaults
from subprojects.wp7_gui.v3.shared.pretraining import PretrainingService
from subprojects.wp7_gui.v3.shared.run_paths import default_storage_root, get_storage_root_for_session, list_all_runs

router = APIRouter(prefix="/viewer", tags=["viewer"])
_templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))
_pretraining = PretrainingService(SETTINGS.pretraining_storage_root)

_OVERRIDE_PREFIX = "gui_override_"


def _resolve_cfg_folder(cfg_folder: str | None) -> Path | None:
    """Validate and resolve a cfg_folder query param to a Path, or None if invalid."""
    if not cfg_folder:
        return None
    try:
        candidate = Path(cfg_folder).resolve()
        candidate.relative_to(SETTINGS.config_overrides_root.resolve())
        if not candidate.name.startswith(_OVERRIDE_PREFIX):
            return None
        return candidate if candidate.is_dir() else None
    except ValueError:
        return None


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
            "panel_visibility_defaults_json": json.dumps(
                load_panel_visibility_defaults(SETTINGS.panel_visibility_defaults_path)
            ),
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
# Vis-config API — serves mesh_assembly_v4 style keys from vis_config.yaml so
# mesh_viewer.js can apply outer_lines_only rendering without hardcoded values.
# ---------------------------------------------------------------------------
@router.get("/api/vis-config")
def api_vis_config() -> JSONResponse:
    try:
        vis_cfg = yaml.safe_load((SETTINGS.wp4_configs_dir / "vis_config.yaml").read_text())
        assembly_cfg: dict[str, Any] = vis_cfg.get("visual_types", {}).get("mesh_assembly_v4", {})
    except Exception:
        assembly_cfg = {}
    return JSONResponse(content=assembly_cfg)


# ---------------------------------------------------------------------------
# Runs API
# ---------------------------------------------------------------------------
@router.get("/api/runs")
def api_list_runs(cfg_session: str | None = None, cfg_folder: str | None = None) -> dict[str, Any]:
    session_dir = _resolve_cfg_folder(cfg_folder)
    storage_root = get_storage_root_for_session(session_dir)
    runs = list_all_runs(storage_root)
    if cfg_session is not None:
        runs = [r for r in runs if r.get("cfg_session") == cfg_session]
    return {"storage_root": str(storage_root), "runs": runs}


@router.get("/api/runs/{run_id:path}/artifacts")
def api_run_artifacts(run_id: str) -> dict[str, Any]:
    """List artifacts for a specific finished run.

    run_id is the URL-encoded relative path from storage_root:
    e.g.  model_hash%2Fwandb_run_name
    """
    storage_root = default_storage_root()
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
