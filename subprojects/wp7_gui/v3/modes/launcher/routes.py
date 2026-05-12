"""Launcher mode router — page views + API endpoints."""

from __future__ import annotations

import asyncio
from typing import Any

import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from subprojects.wp7_gui.v3.core import fs_poller, log_parser, sse
from subprojects.wp7_gui.v3.core.process_manager import ProcessManager
from subprojects.wp7_gui.v3.core.settings import SETTINGS
from subprojects.wp7_gui.v3.shared.artifact_index import list_artifacts
from subprojects.wp7_gui.v3.shared.assembly_selection import AssemblySelectionService
from subprojects.wp7_gui.v3.shared.config_overrides import ConfigOverrideManager
from subprojects.wp7_gui.v3.shared.run_paths import default_storage_root, detect_current_run_dir

router = APIRouter(prefix="/launcher", tags=["launcher"])
_templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))

# ---------------------------------------------------------------------------
# Singletons shared across requests
# ---------------------------------------------------------------------------
_process_manager = ProcessManager(SETTINGS.logs_dir)
_config_manager = ConfigOverrideManager(
    root=SETTINGS.config_overrides_root,
    schema_path=SETTINGS.schema_path,
    wp4_configs_dir=SETTINGS.wp4_configs_dir,
)
_assembly_service_live = AssemblySelectionService(SETTINGS.repo_root, subdir="live")
_assembly_service_completed = AssemblySelectionService(SETTINGS.repo_root, subdir="completed")


# Assembly ID validation bounds — keep in sync with schema/gui_schema.json
_ASSEMBLY_ID_MIN = 1
_ASSEMBLY_ID_MAX = 51


def _assembly(pipeline: str) -> AssemblySelectionService:
    if pipeline == "completed":
        return _assembly_service_completed
    if pipeline == "live":
        return _assembly_service_live
    raise ValueError(f"Unknown pipeline value '{pipeline}'. Expected 'live' or 'completed'.")

_SUBPAGES = {
    "config-overrides": ("Launcher — Config Overrides", "launcher/config_overrides.html"),
    "assembly-selection": ("Launcher — Assembly Selection", "launcher/assembly_selection.html"),
    "optimization": ("Launcher — Live Run", "launcher/optimization.html"),
    "output": ("Launcher — Output", "launcher/output.html"),
}


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class RunStartRequest(BaseModel):
    assembly_id: int | str | None = None
    cfg_folder: str = ""


class OverrideValuesRequest(BaseModel):
    folder: str
    updates: dict[str, dict[str, Any]]


class CleanupRequest(BaseModel):
    folders: list[str] | None = None
    all: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_total_epochs() -> int:
    try:
        with SETTINGS.general_config_path.open(encoding="utf-8") as f:
            gen = yaml.safe_load(f) or {}
        training_name = gen.get("active_configs", {}).get("training", "training_v0_config.yaml")
        training_path = SETTINGS.wp4_configs_dir / training_name
        with training_path.open(encoding="utf-8") as f:
            tcfg = yaml.safe_load(f) or {}
        return int(tcfg.get("training_params", {}).get("num_epochs", 100))
    except Exception:
        return 100


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------
@router.get("")
def launcher_index(request: Request, pipeline: str = "live"):
    return _templates.TemplateResponse(
        request,
        "launcher/index.html",
        {
            "active_mode": "launcher",
            "active_page": "config-overrides",
            "page_title": "Launcher",
            "pipeline": pipeline,
        },
    )


@router.get("/{page_key}", name="launcher_subpage")
def launcher_subpage(request: Request, page_key: str, pipeline: str = "live"):
    if page_key not in _SUBPAGES:
        raise HTTPException(status_code=404, detail={"message": f"Unknown launcher page '{page_key}'"})
    title, template = _SUBPAGES[page_key]
    return _templates.TemplateResponse(
        request,
        template,
        {
            "active_mode": "launcher",
            "active_page": page_key,
            "page_title": title,
            "pipeline": pipeline,
        },
    )


# ---------------------------------------------------------------------------
# Run control API
# ---------------------------------------------------------------------------
@router.post("/api/run/start")
def run_start(payload: RunStartRequest) -> JSONResponse:
    assembly_id: int | None = None
    if payload.assembly_id not in (None, ""):
        try:
            assembly_id = int(str(payload.assembly_id).strip())
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail={"message": "assembly_id must be an integer"}) from None
        if not _ASSEMBLY_ID_MIN <= assembly_id <= _ASSEMBLY_ID_MAX:
            raise HTTPException(status_code=400, detail={"message": f"assembly_id must be {_ASSEMBLY_ID_MIN}-{_ASSEMBLY_ID_MAX}"})

    extra_args: list[str] = []
    if payload.cfg_folder:
        extra_args += ["--uses-gui", "--cfg-folder", payload.cfg_folder.strip()]

    env_overrides: dict[str, str] = {}
    if assembly_id is not None:
        env_overrides[SETTINGS.assembly_id_env_var] = str(assembly_id)

    try:
        pid, log_path = _process_manager.start(
            run_local_script=SETTINGS.run_local_script,
            repo_parent_dir=SETTINGS.repo_root.parent,
            extra_args=extra_args or None,
            env_overrides=env_overrides or None,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc

    return JSONResponse({"status": "started", "pid": pid, "log": str(log_path)})


@router.post("/api/run/stop")
def run_stop() -> JSONResponse:
    return JSONResponse(_process_manager.stop())


@router.get("/api/run/state")
def run_state() -> dict[str, Any]:
    state = _process_manager.get_state()
    return {
        "running": state["running"],
        "pid": state["pid"],
        "return_code": state["return_code"],
        "log_path": str(state["log_path"]) if state["log_path"] else None,
    }


@router.get("/api/run/logs")
def run_logs() -> dict[str, Any]:
    state = _process_manager.get_state()
    log_path = state["log_path"]
    logs, truncated = log_parser.tail_log(log_path) if log_path else ("", False)
    return {
        "logs": logs,
        "running": state["running"],
        "pid": state["pid"],
        "return_code": state["return_code"],
        "truncated": truncated,
    }


@router.get("/api/run/progress")
def run_progress() -> dict[str, Any]:
    state = _process_manager.get_state()
    log_path = state["log_path"]
    progress = (
        log_parser.get_progress(log_path, _get_total_epochs())
        if log_path
        else {"num_epochs": _get_total_epochs(), "current_epoch": -1, "progress_percent": 0.0, "completed": False}
    )
    return {**progress, "running": state["running"], "return_code": state["return_code"]}


@router.get("/api/run/artifacts")
def run_artifacts() -> dict[str, Any]:
    storage_root = default_storage_root()
    run_dir = detect_current_run_dir(storage_root)
    if run_dir is None:
        return {
            "run_dir": None,
            "storage_root": str(storage_root),
            "mesh_assembly_gifs": [],
            "solar_loss_gifs": [],
            "meshes": [],
            "message": "No run directory found yet.",
        }
    return {**list_artifacts(run_dir, storage_root), "message": "Artifacts loaded"}


@router.get("/api/run/stream/logs")
async def run_stream_logs() -> StreamingResponse:
    async def _gen():
        last_size = 0
        while True:
            state = _process_manager.get_state()
            log_path = state["log_path"]
            if log_path and log_path.exists():
                text, _ = log_parser.tail_log(log_path)
                if len(text) != last_size:
                    new_lines = text[last_size:]
                    last_size = len(text)
                    for line in new_lines.splitlines():
                        yield sse.event("log", {"line": line})
                    epoch = log_parser.parse_epoch(text)
                    if epoch >= 0:
                        yield sse.event("epoch", {"epoch": epoch, "total": _get_total_epochs()})
            if not state["running"]:
                yield sse.event("done", {"return_code": state["return_code"]})
                break
            await asyncio.sleep(1.0)

    return sse.stream_response(_gen())


@router.get("/api/run/stream/artifacts")
async def run_stream_artifacts() -> StreamingResponse:
    async def _gen():
        seen: dict[str, float] = {}
        while True:
            state = _process_manager.get_state()
            storage_root = default_storage_root()
            run_dir = detect_current_run_dir(storage_root)
            if run_dir is not None:
                new_events, seen = fs_poller.poll_once(run_dir, seen)
                for ev in new_events:
                    yield sse.event("new_artifact", ev)
            if not state["running"]:
                yield sse.event("done", {})
                break
            await asyncio.sleep(1.5)

    return sse.stream_response(_gen())


# ---------------------------------------------------------------------------
# Config-overrides API
# ---------------------------------------------------------------------------
@router.post("/api/config-overrides/init")
def cfg_init() -> dict[str, Any]:
    return _config_manager.init_session()


@router.get("/api/config-overrides/schema")
def cfg_schema() -> dict[str, Any]:
    return _config_manager.get_schema()


@router.get("/api/config-overrides/values")
def cfg_values_get(folder: str = "") -> dict[str, Any]:
    if not folder:
        raise HTTPException(status_code=400, detail={"message": "Missing 'folder' query parameter"})
    return _config_manager.get_values(folder)


@router.put("/api/config-overrides/values")
def cfg_values_put(payload: OverrideValuesRequest) -> dict[str, Any]:
    return _config_manager.set_values(payload.folder, payload.updates)


@router.get("/api/config-overrides/sessions")
def cfg_sessions() -> dict[str, Any]:
    return _config_manager.list_sessions()


@router.post("/api/config-overrides/cleanup")
def cfg_cleanup(payload: CleanupRequest) -> dict[str, Any]:
    return _config_manager.cleanup(payload.folders, payload.all)


# ---------------------------------------------------------------------------
# Assembly-selection API
# ---------------------------------------------------------------------------
@router.get("/api/assembly-selection/params")
def assembly_params(pipeline: str = "live") -> dict[str, dict[str, float]]:
    return _assembly(pipeline).get_params()


@router.post("/api/assembly-selection/update")
def assembly_update(payload: dict[str, Any], pipeline: str = "live") -> dict[str, Any]:
    return _assembly(pipeline).update_params(payload)


@router.post("/api/assembly-selection/add-wheel")
def assembly_add_wheel(pipeline: str = "live") -> dict[str, Any]:
    return _assembly(pipeline).add_wheel()


@router.post("/api/assembly-selection/remove-wheel")
def assembly_remove_wheel(pipeline: str = "live") -> dict[str, Any]:
    return _assembly(pipeline).remove_wheel()


@router.get("/api/assembly-selection/mesh")
def assembly_mesh(pipeline: str = "live") -> FileResponse:
    return _assembly(pipeline).get_mesh_response()


@router.get("/api/assembly-selection/checks")
def assembly_checks(pipeline: str = "live") -> dict:
    return _assembly(pipeline).get_checks()
