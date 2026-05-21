"""Viewer mode router — page views + API endpoints for finished runs."""

from __future__ import annotations

import json
import uuid as uuid_module
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared.artifact_index import list_artifacts
from subprojects.wp7_gui.v3.shared.latent_explorer import (
    extract_mesh_glb,
    find_last_epoch_checkpoint,
    is_checkpoint_path_allowed,
    load_checkpoint_info,
)
from subprojects.wp7_gui.v3.shared.panel_visibility import load_panel_visibility_defaults
from subprojects.wp7_gui.v3.shared.pretraining import PretrainingService
from subprojects.wp7_gui.v3.shared.run_paths import default_storage_root, get_storage_root_for_session, list_all_runs
from subprojects.wp7_gui.v3.shared.theme import build_template_theme_context, get_mesh_assembly_config, load_vis_config

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
    theme_ctx = build_template_theme_context()
    return _templates.TemplateResponse(
        request,
        "viewer/index.html",
        {
            "active_mode": "viewer",
            "active_page": "runs",
            "page_title": "Viewer — Runs",
            "panel_visibility_defaults_json": json.dumps(load_panel_visibility_defaults(SETTINGS.panel_visibility_defaults_path)),
            "mesh_priority_json": SETTINGS.mesh_priority_path.read_text(encoding="utf-8"),
            **theme_ctx,
        },
    )


@router.get("/pretraining")
def viewer_pretraining(request: Request):
    theme_ctx = build_template_theme_context()
    return _templates.TemplateResponse(
        request,
        "viewer/pretraining.html",
        {
            "active_mode": "viewer",
            "active_page": "pretraining",
            "page_title": "Viewer — Pretraining GIFs",
            **theme_ctx,
        },
    )


# ---------------------------------------------------------------------------
# Vis-config API — serves mesh_assembly_v4 style keys from vis_config.yaml so
# mesh_viewer.js can apply outer_lines_only rendering without hardcoded values.
# ---------------------------------------------------------------------------
@router.get("/api/vis-config")
def api_vis_config() -> JSONResponse:
    vis_cfg = load_vis_config()
    assembly_cfg: dict[str, Any] = get_mesh_assembly_config(vis_cfg)
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


# ---------------------------------------------------------------------------
# Latent Explorer page + API
# ---------------------------------------------------------------------------

_LATENT_MESH_DIR = SETTINGS.cache_dir / "latent_meshes"
_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _LoadCheckpointBody(BaseModel):
    checkpoint_path: str


class _QueryMeshBody(BaseModel):
    checkpoint_path: str
    z0: float
    z1: float
    resolution: int = 35


@router.get("/latent-explorer")
def viewer_latent_explorer(request: Request) -> Any:
    theme_ctx = build_template_theme_context()
    return _templates.TemplateResponse(
        request,
        "viewer/latent_explorer.html",
        {
            "active_mode": "launcher",
            "pipeline": "completed",
            "active_page": "latent-explorer",
            "page_title": "Latent Explorer",
            **theme_ctx,
        },
    )


@router.post("/api/latent/load-checkpoint")
def api_latent_load_checkpoint(body: _LoadCheckpointBody) -> JSONResponse:
    """Load a checkpoint and return latent-space metadata.

    The model is cached for subsequent mesh queries.
    """
    ckpt_path = Path(body.checkpoint_path)
    if not is_checkpoint_path_allowed(ckpt_path):
        raise HTTPException(
            status_code=400,
            detail={"message": f"Checkpoint path is not in an allowed directory: {ckpt_path}"},
        )
    try:
        ckpt_path = find_last_epoch_checkpoint(ckpt_path)
        device = torch.device(_DEFAULT_DEVICE)
        info = load_checkpoint_info(ckpt_path, device)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    return JSONResponse(
        content={
            "resolved_path": str(ckpt_path),
            "z_dim": info["z_dim"],
            "z_radius": info["z_radius"],
            "z_pretrained": info["z_pretrained"],
        }
    )


@router.post("/api/latent/query-mesh")
def api_latent_query_mesh(body: _QueryMeshBody) -> JSONResponse:
    """Extract the aeroshell mesh at the given latent point and return a URL."""
    ckpt_path = Path(body.checkpoint_path)
    if not is_checkpoint_path_allowed(ckpt_path):
        raise HTTPException(
            status_code=400,
            detail={"message": f"Checkpoint path is not in an allowed directory: {ckpt_path}"},
        )

    if not (1 <= body.resolution <= 100):
        raise HTTPException(status_code=400, detail={"message": "resolution must be between 1 and 100"})

    _LATENT_MESH_DIR.mkdir(parents=True, exist_ok=True)
    mesh_id = str(uuid_module.uuid4())
    glb_path = _LATENT_MESH_DIR / f"{mesh_id}.glb"

    try:
        device = torch.device(_DEFAULT_DEVICE)
        glb_bytes = extract_mesh_glb(
            ckpt_path=ckpt_path,
            z0=body.z0,
            z1=body.z1,
            resolution=body.resolution,
            device=device,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc)}) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc

    glb_path.write_bytes(glb_bytes)
    # The ?path=mesh.glb suffix triggers GLTFLoader in mesh_viewer.js
    mesh_url = f"/viewer/api/latent/mesh-file?id={mesh_id}&path=mesh.glb"
    return JSONResponse(content={"mesh_url": mesh_url})


@router.get("/api/latent/mesh-file")
def api_latent_mesh_file(id: str = "", path: str = "") -> FileResponse:  # noqa: ARG001  # path param consumed by mesh_viewer.js extension detection, not by this handler
    """Serve a previously generated GLB mesh file by its UUID."""
    if not id or "/" in id or "\\" in id or ".." in id:
        raise HTTPException(status_code=400, detail={"message": "Invalid mesh id"})
    glb_path = (_LATENT_MESH_DIR / f"{id}.glb").resolve()
    try:
        glb_path.relative_to(_LATENT_MESH_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail={"message": "Invalid mesh id"}) from None
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail={"message": "Mesh file not found"})
    return FileResponse(str(glb_path), media_type="model/gltf-binary")
