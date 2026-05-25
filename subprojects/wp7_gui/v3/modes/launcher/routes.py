"""Launcher mode router — page views + API endpoints."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import threading
import time
from pathlib import Path
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
from subprojects.wp7_gui.v3.shared.csv_metrics import CSVMetricsWriter
from subprojects.wp7_gui.v3.shared.panel_visibility import load_panel_visibility_defaults
from subprojects.wp7_gui.v3.shared.run_paths import (
    collect_run_dirs,
    detect_current_run_dir,
    get_storage_root_for_session,
    read_cfg_session,
)
from subprojects.wp7_gui.v3.shared.theme import build_template_theme_context

router = APIRouter(prefix="/launcher", tags=["launcher"])
_templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singletons shared across requests
# ---------------------------------------------------------------------------
_process_manager = ProcessManager(SETTINGS.logs_dir)
_csv_metrics = CSVMetricsWriter(SETTINGS.logs_dir / "gui_runs_summary.csv")
_runtime_state: dict[str, str] = {"active_cfg_folder": ""}  # updated on each run/start so progress/SSE can resolve epochs
_config_manager = ConfigOverrideManager(
    root=SETTINGS.config_overrides_root,
    schema_path=SETTINGS.schema_path,
    wp4_configs_dir=SETTINGS.wp4_configs_dir,
    assembly_map_path=SETTINGS.repo_root / "wp2_assembler" / "version4" / "config" / "assembly_map.yaml",
)
_assembly_service_live = AssemblySelectionService(SETTINGS.repo_root, subdir="live")
_assembly_service_completed = AssemblySelectionService(SETTINGS.repo_root, subdir="completed")


# Assembly ID validation bounds — keep in sync with schema/gui_schema.json
_ASSEMBLY_ID_MIN = 1
_ASSEMBLY_ID_MAX = 99999


def _assembly(pipeline: str) -> AssemblySelectionService:
    if pipeline == "completed":
        return _assembly_service_completed
    if pipeline == "live":
        return _assembly_service_live
    raise ValueError(f"Unknown pipeline value '{pipeline}'. Expected 'live' or 'completed'.")


_SUBPAGES = {
    "assembly-selection": ("1 - Launcher — Assembly Selection", "launcher/assembly_selection.html"),
    "config-overrides": ("2 - Launcher — Configure and Run", "launcher/config_overrides.html"),
    "output": ("3 - Launcher — Output", "launcher/output.html"),
    "leaderboard": ("4 - Launcher — Live Leaderboard", "launcher/leaderboard.html"),
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


class FolderRequest(BaseModel):
    folder: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_total_epochs(cfg_folder: str = "") -> int:
    """Return num_epochs, preferring the active override folder over wp4 defaults."""
    # 1. Try the override folder first (when a config session is active)
    if cfg_folder:
        try:
            override_dir = Path(cfg_folder)
            # Resolve the training config name from the override's general_config.yaml
            gen_path = override_dir / "general_config.yaml"
            training_name = "training_config.yaml"
            if gen_path.exists():
                with gen_path.open(encoding="utf-8") as f:
                    gen = yaml.safe_load(f) or {}
                training_name = gen.get("active_configs", {}).get("training", training_name)
            training_path = override_dir / training_name
            if training_path.exists():
                with training_path.open(encoding="utf-8") as f:
                    tcfg = yaml.safe_load(f) or {}
                epochs = tcfg.get("training_params", {}).get("num_epochs")
                if epochs is not None:
                    return int(epochs)
        except Exception:
            pass  # fall through to defaults

    # 2. Fall back to wp4 defaults
    try:
        with SETTINGS.general_config_path.open(encoding="utf-8") as f:
            gen = yaml.safe_load(f) or {}
        training_name = gen.get("active_configs", {}).get("training", "training_config.yaml")
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
    theme_ctx = build_template_theme_context()
    return _templates.TemplateResponse(
        request,
        "launcher/index.html",
        {
            "active_mode": "launcher",
            "active_page": "config-overrides",
            "page_title": "Launcher",
            "pipeline": pipeline,
            "panel_visibility_defaults_json": json.dumps(load_panel_visibility_defaults(SETTINGS.panel_visibility_defaults_path)),
            "mesh_priority_json": SETTINGS.mesh_priority_path.read_text(encoding="utf-8"),
            **theme_ctx,
        },
    )


@router.get("/{page_key}", name="launcher_subpage")
def launcher_subpage(request: Request, page_key: str, pipeline: str = "live"):
    if page_key not in _SUBPAGES:
        raise HTTPException(status_code=404, detail={"message": f"Unknown launcher page '{page_key}'"})
    title, template = _SUBPAGES[page_key]
    theme_ctx = build_template_theme_context()
    return _templates.TemplateResponse(
        request,
        template,
        {
            "active_mode": "launcher",
            "active_page": page_key,
            "page_title": title,
            "pipeline": pipeline,
            "panel_visibility_defaults_json": json.dumps(load_panel_visibility_defaults(SETTINGS.panel_visibility_defaults_path)),
            "mesh_priority_json": SETTINGS.mesh_priority_path.read_text(encoding="utf-8"),
            **theme_ctx,
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

    _runtime_state["active_cfg_folder"] = payload.cfg_folder.strip() if payload.cfg_folder else ""

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

    # Spawn a background thread that writes the final epoch CSV when the run
    # completes, ensuring coverage even when no SSE client is connected.
    threading.Thread(
        target=_csv_completion_writer,
        args=(log_path, _runtime_state["active_cfg_folder"]),
        daemon=True,
        name="csv-completion-writer",
    ).start()

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
    active_cfg_folder = _runtime_state["active_cfg_folder"]
    progress = (
        log_parser.get_progress(log_path, _get_total_epochs(active_cfg_folder))
        if log_path
        else {
            "num_epochs": _get_total_epochs(active_cfg_folder),
            "current_epoch": -1,
            "progress_percent": 0.0,
            "completed": False,
        }
    )
    return {**progress, "running": state["running"], "return_code": state["return_code"]}


# ---------------------------------------------------------------------------
# CSV metrics helpers
# ---------------------------------------------------------------------------


def _write_epoch_csv(
    active_cfg_folder: str,
    epoch: int,
    drag_value: float | None,
    solar_power_w: float | None,
    cruising_speed_ms: float | None,
) -> bool:
    """Write one epoch row to the per-run CSV and update the global summary.

    Returns True when the row was written, False when the run directory could
    not be resolved yet (e.g. the output folder has not been created by the
    training process).  Callers must not advance their ``last_written_epoch``
    counter on a False return so the epoch is retried on the next poll.

    Args:
        active_cfg_folder: Current GUI config override folder path string.
        epoch: Zero-based epoch index.
        drag_value: Drag force in Newtons, or None.
        solar_power_w: Solar output in Watts, or None.
        cruising_speed_ms: Cruising speed in m/s, or None.
    """
    cfg_folder = active_cfg_folder.strip()
    session_dir = Path(cfg_folder) if cfg_folder else None
    storage_root = get_storage_root_for_session(session_dir)
    cfg_session = Path(cfg_folder).name if cfg_folder else ""
    run_dir = detect_current_run_dir(storage_root, preferred_cfg_session=cfg_session or None)
    if run_dir is None:
        return False
    _csv_metrics.write_epoch_row(
        run_dir=run_dir,
        epoch=epoch,
        drag_value=drag_value,
        solar_power_w=solar_power_w,
        cruising_speed_ms=cruising_speed_ms,
    )
    _csv_metrics.write_summary_row(
        run_id=run_dir.name,
        cfg_session=cfg_session,
        epoch=epoch,
        drag_value=drag_value,
        solar_power_w=solar_power_w,
        cruising_speed_ms=cruising_speed_ms,
    )
    return True


@router.get("/api/run/metrics/latest")
def run_metrics_latest() -> dict[str, Any]:
    """Return the latest epoch metrics from the per-run CSV.

    Used by the frontend to restore live metric values on page load / reconnect,
    avoiding a blank sidebar until the next SSE epoch event arrives.

    Returns dict with keys: ``epoch``, ``drag_value``, ``solar_power_w``,
    ``cruising_speed_ms`` (all may be None if not yet written), plus
    ``run_dir`` (str or null) and ``found`` (bool).
    """
    active_cfg_folder = _runtime_state["active_cfg_folder"].strip() if _runtime_state["active_cfg_folder"] else ""
    session_dir = Path(active_cfg_folder) if active_cfg_folder else None
    storage_root = get_storage_root_for_session(session_dir)
    cfg_session = Path(active_cfg_folder).name if active_cfg_folder else None
    run_dir = detect_current_run_dir(storage_root, preferred_cfg_session=cfg_session or None)
    if run_dir is None:
        return {
            "found": False,
            "run_dir": None,
            "epoch": None,
            "drag_value": None,
            "solar_power_w": None,
            "cruising_speed_ms": None,
        }
    metrics = _csv_metrics.read_latest_metrics(run_dir)
    if not metrics:
        return {
            "found": False,
            "run_dir": str(run_dir),
            "epoch": None,
            "drag_value": None,
            "solar_power_w": None,
            "cruising_speed_ms": None,
        }
    return {
        "found": True,
        "run_dir": str(run_dir),
        "epoch": metrics.get("epoch"),
        "drag_value": metrics.get("drag_value"),
        "solar_power_w": metrics.get("solar_power_w"),
        "cruising_speed_ms": metrics.get("cruising_speed_ms"),
    }


@router.get("/api/run/metrics/history")
def run_metrics_history(limit: int = 120) -> dict[str, Any]:
    """Return trailing epoch metric arrays for the active run.

    Backend values stay in SI units (speed in m/s); clients may convert for display.
    """
    safe_limit = max(1, min(int(limit), 500))
    active_cfg_folder = _runtime_state["active_cfg_folder"].strip() if _runtime_state["active_cfg_folder"] else ""
    session_dir = Path(active_cfg_folder) if active_cfg_folder else None
    storage_root = get_storage_root_for_session(session_dir)
    cfg_session = Path(active_cfg_folder).name if active_cfg_folder else None
    run_dir = detect_current_run_dir(storage_root, preferred_cfg_session=cfg_session or None)
    if run_dir is None:
        return {
            "found": False,
            "run_dir": None,
            "history": {
                "epoch": [],
                "drag_value": [],
                "solar_power_w": [],
                "cruising_speed_ms": [],
            },
        }
    return {
        "found": True,
        "run_dir": str(run_dir),
        "history": _csv_metrics.read_metric_history(run_dir, limit=safe_limit),
    }


@router.get("/api/run/artifacts")
def run_artifacts() -> dict[str, Any]:
    active_cfg_folder = _runtime_state["active_cfg_folder"].strip() if _runtime_state["active_cfg_folder"] else ""
    session_dir = Path(active_cfg_folder) if active_cfg_folder else None
    storage_root = get_storage_root_for_session(session_dir)
    active_cfg_session = Path(active_cfg_folder).name if active_cfg_folder else None
    run_dir = detect_current_run_dir(storage_root, preferred_cfg_session=active_cfg_session)
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


# ---------------------------------------------------------------------------
# Leaderboard API
# ---------------------------------------------------------------------------


def _parse_float_cell(raw: str) -> float | None:
    """Parse a CSV cell string to float; returns None for missing/invalid values."""
    stripped = raw.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


@router.get("/api/leaderboard")
def api_leaderboard(current_run_id: str | None = None) -> JSONResponse:
    """Return leaderboard data from gui_runs_summary.csv.

    Reads all rows, sorts by cruising_speed_ms descending, returns the top 10
    plus the current run's entry and rank when it falls outside the top 10.
    When no current_run_id is supplied the most recently updated run (by
    timestamp) is used so the leaderboard always has a highlighted row.

    Args:
        current_run_id: Optional run_id to mark as the current run.  When
            omitted the latest run by timestamp is highlighted instead.

    Returns:
        JSON with:
        - ``entries``: list of up to 10 ranked dicts (rank, run_id, cruising_speed_ms,
          drag_value, solar_power_w, is_current).
        - ``current_entry``: the current run's dict with its actual rank, or null when
          the current run is already in the top 10.
        - ``effective_run_id``: the run_id that was ultimately highlighted.
        - ``total_runs``: total number of runs in the CSV.
    """
    summary_path = SETTINGS.logs_dir / "gui_runs_summary.csv"
    rows: list[dict[str, str]] = []
    if summary_path.exists():
        try:
            with summary_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except OSError as exc:
            LOGGER.warning("Could not read summary CSV %s: %s", summary_path, exc)

    # Resolve which run to highlight: explicit param takes priority, otherwise
    # fall back to the most recently updated run by timestamp.
    effective_run_id = current_run_id
    if not effective_run_id and rows:
        latest = max(rows, key=lambda r: r.get("timestamp", ""))
        effective_run_id = latest.get("run_id") or None

    def _speed(row: dict[str, str]) -> float:
        return _parse_float_cell(row.get("cruising_speed_ms", "")) or 0.0

    sorted_rows = sorted(rows, key=_speed, reverse=True)

    def _make_entry(rank: int, row: dict[str, str], is_current: bool) -> dict[str, Any]:
        return {
            "rank": rank,
            "run_id": row.get("run_id", ""),
            "cruising_speed_ms": _parse_float_cell(row.get("cruising_speed_ms", "")),
            "drag_value": _parse_float_cell(row.get("drag_value", "")),
            "solar_power_w": _parse_float_cell(row.get("solar_power_w", "")),
            "is_current": is_current,
        }

    top10 = sorted_rows[:10]
    current_in_top10 = False
    entries: list[dict[str, Any]] = []
    for idx, row in enumerate(top10):
        is_current = effective_run_id is not None and row.get("run_id") == effective_run_id
        if is_current:
            current_in_top10 = True
        entries.append(_make_entry(idx + 1, row, is_current))

    current_entry: dict[str, Any] | None = None
    if effective_run_id and not current_in_top10:
        for idx, row in enumerate(sorted_rows):
            if row.get("run_id") == effective_run_id:
                current_entry = _make_entry(idx + 1, row, True)
                break

    # If an explicit current_run_id was given but isn't in the CSV at all
    # (e.g. a SLURM/offline run), create an unranked placeholder so the
    # frontend can still highlight which run the user is viewing.
    if current_run_id and not current_in_top10 and current_entry is None:
        current_entry = {
            "rank": None,
            "run_id": current_run_id,
            "cruising_speed_ms": None,
            "drag_value": None,
            "solar_power_w": None,
            "is_current": True,
        }

    return JSONResponse(
        content={
            "entries": entries,
            "current_entry": current_entry,
            "total_runs": len(rows),
            "effective_run_id": effective_run_id,
        }
    )


@router.get("/api/leaderboard/sparklines")
def api_leaderboard_sparklines(run_ids: str = "", limit: int = 40) -> JSONResponse:
    """Return trailing per-run metric series for leaderboard sparkline rendering."""
    requested = [rid.strip() for rid in run_ids.split(",") if rid.strip()]
    if not requested:
        return JSONResponse(content={"histories": {}})

    active_cfg_folder = _runtime_state["active_cfg_folder"].strip() if _runtime_state["active_cfg_folder"] else ""
    session_dir = Path(active_cfg_folder) if active_cfg_folder else None
    storage_root = get_storage_root_for_session(session_dir)
    safe_limit = max(1, min(int(limit), 120))
    histories = _csv_metrics.read_histories_by_run_id(storage_root=storage_root, run_ids=requested, limit=safe_limit)
    return JSONResponse(content={"histories": histories})


@router.get("/api/leaderboard/stream")
async def leaderboard_stream() -> StreamingResponse:
    """SSE stream that fires a 'changed' event whenever gui_runs_summary.csv is modified.

    Clients subscribe once and receive a push notification each time a new epoch
    row is upserted into the summary, instead of polling on a fixed interval.
    An initial event is sent immediately on connect so the client loads data
    without waiting for the first file change.
    """

    async def _gen():
        summary_path = SETTINGS.logs_dir / "gui_runs_summary.csv"
        try:
            last_mtime: float = summary_path.stat().st_mtime if summary_path.exists() else 0.0
        except OSError:
            last_mtime = 0.0
        # Trigger an immediate load on connect.
        yield sse.event("changed", {})
        while True:
            await asyncio.sleep(1.0)
            try:
                mtime = summary_path.stat().st_mtime if summary_path.exists() else 0.0
            except OSError:
                mtime = last_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                yield sse.event("changed", {})

    return sse.stream_response(_gen())


# ---------------------------------------------------------------------------
# Background CSV completion writer
# ---------------------------------------------------------------------------


def _csv_completion_writer(log_path: Path, active_cfg_folder: str) -> None:
    """Background thread: write epoch CSV rows continuously as the run progresses.

    Polls the log file every few seconds while the run is active, flushing any
    newly completed epoch blocks to the per-run and summary CSVs.  ``include_last``
    is False during polling (the current epoch block may still be accumulating
    lines) and True on the final flush after the process exits, ensuring every
    epoch is captured.  ``last_written_epoch`` is threaded through all iterations
    so no epoch is ever written twice.

    Args:
        log_path: Path to the run log file being monitored.
        active_cfg_folder: GUI config override folder path captured at run start.
    """
    _poll_interval = 2.0  # seconds between state checks
    last_written_epoch: int = _sse_init_last_written_epoch(active_cfg_folder)
    while True:
        time.sleep(_poll_interval)
        should_stop = False
        try:
            state = _process_manager.get_state()
            current_log = state.get("log_path")
            if current_log != log_path:
                # A newer run has started — this thread is stale; stop.
                should_stop = True
            elif not state["running"]:
                # Process has exited — final flush includes the last epoch block.
                if log_path.exists():
                    final_text = log_path.read_text(encoding="utf-8", errors="replace")
                    _sse_flush_epoch_csv(final_text, active_cfg_folder, last_written_epoch, include_last=True)
                should_stop = True
            # Run still active — flush completed epoch blocks only.
            elif log_path.exists():
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                last_written_epoch = _sse_flush_epoch_csv(log_text, active_cfg_folder, last_written_epoch)
        except Exception as exc:
            LOGGER.warning("csv_completion_writer: unexpected error: %s", exc)
            should_stop = True
        if should_stop:
            return


# ---------------------------------------------------------------------------
# SSE log-stream helpers
# ---------------------------------------------------------------------------


def _sse_init_last_written_epoch(active_cfg_folder: str) -> int:
    """Return the last epoch already written to the per-run CSV, or -1.

    Only considers run directories explicitly tagged with the current cfg
    session.  Falling back to the mtime-newest run would incorrectly pre-seed
    ``last_written_epoch`` with a previous run's final epoch, causing all
    epochs of the new run to be silently skipped.
    """
    try:
        if active_cfg_folder:
            session_dir = Path(active_cfg_folder)
            storage_root = get_storage_root_for_session(session_dir)
            cfg_session = session_dir.name
            all_run_dirs = collect_run_dirs(storage_root)
            tagged = [p for p in all_run_dirs if read_cfg_session(p) == cfg_session]
            if not tagged:
                return -1
            run_dir = max(tagged, key=lambda p: p.stat().st_mtime)
            metrics = _csv_metrics.read_latest_metrics(run_dir)
            if metrics and metrics.get("epoch") is not None:
                return int(metrics["epoch"])  # type: ignore[arg-type]
    except Exception:
        pass
    return -1


def _sse_flush_epoch_csv(
    log_text: str,
    active_cfg_folder: str,
    last_written_epoch: int,
    *,
    include_last: bool = False,
) -> int:
    """Write CSV rows for completed epoch blocks found in *log_text*.

    Skips the final block by default because it may still be accumulating
    lines.  Pass ``include_last=True`` when the run has finished.

    Returns the updated ``last_written_epoch`` value.
    """
    epoch_blocks = log_parser.parse_epoch_blocks(log_text)
    all_epochs = sorted(epoch_blocks.keys())
    if include_last:
        epochs_to_write = all_epochs
    elif all_epochs and epoch_blocks[all_epochs[-1]].get("drag_value") is not None:
        # The last epoch block has its primary metric logged.  Since WP4 logs
        # metrics before writing the mesh file, drag_value being present means
        # the epoch is computationally complete — include it so that metrics
        # appear on the output page alongside the mesh.
        epochs_to_write = all_epochs
    else:
        # Last epoch block is still accumulating; defer until next poll.
        epochs_to_write = all_epochs[:-1]
    for ep in epochs_to_write:
        if ep > last_written_epoch:
            written = _write_epoch_csv(active_cfg_folder=active_cfg_folder, epoch=ep, **epoch_blocks[ep])
            if written:
                last_written_epoch = ep
    return last_written_epoch


@router.get("/api/run/stream/logs")
async def run_stream_logs() -> StreamingResponse:
    async def _gen():
        # Use the actual OS file size for change detection so that the stream
        # never stalls once the log exceeds the tail_log display cap.
        raw_file_size: int = -1  # log_path.stat().st_size
        sent_offset: int = 0  # char offset into the FULL log text sent so far

        while True:
            state = _process_manager.get_state()
            log_path = state["log_path"]
            active_cfg_folder = (_runtime_state["active_cfg_folder"] or "").strip()
            if log_path and log_path.exists():
                try:
                    cur_raw_size = log_path.stat().st_size
                except OSError:
                    cur_raw_size = raw_file_size
                if cur_raw_size != raw_file_size:
                    raw_file_size = cur_raw_size
                    full_text = log_path.read_text(encoding="utf-8", errors="replace")

                    # Send only new content; on the first connect cap history to
                    # the last 60 k chars so the client isn't flooded.
                    new_content = full_text[sent_offset:]
                    if sent_offset == 0 and len(full_text) > 60_000:
                        new_content = full_text[-60_000:]
                    sent_offset = len(full_text)

                    for line in new_content.splitlines():
                        yield sse.event("log", {"line": line})

                    epoch = log_parser.parse_epoch(full_text)
                    if epoch >= 0:
                        metrics = log_parser.parse_metrics(full_text)
                        yield sse.event(
                            "epoch",
                            {
                                "epoch": epoch,
                                "total": _get_total_epochs(active_cfg_folder),
                                "drag_value": metrics.get("drag_value"),
                                "solar_power_w": metrics.get("solar_power_w"),
                                "cruising_speed_ms": metrics.get("cruising_speed_ms"),
                            },
                        )

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
            active_cfg_folder = _runtime_state["active_cfg_folder"].strip() if _runtime_state["active_cfg_folder"] else ""
            session_dir = Path(active_cfg_folder) if active_cfg_folder else None
            storage_root = get_storage_root_for_session(session_dir)
            active_cfg_session = Path(active_cfg_folder).name if active_cfg_folder else None
            run_dir = detect_current_run_dir(storage_root, preferred_cfg_session=active_cfg_session)
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


@router.post("/api/config-overrides/reset")
def cfg_reset(payload: FolderRequest) -> dict[str, Any]:
    return _config_manager.reset_session(payload.folder)


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


@router.get("/api/assembly-selection/map-entry")
def assembly_map_entry(assembly_id: int, pipeline: str = "live") -> dict[str, Any]:
    return _assembly(pipeline).load_from_map_id(assembly_id)


@router.get("/api/assembly-selection/map-meta")
def assembly_map_meta(pipeline: str = "live") -> dict[str, int | None]:
    return _assembly(pipeline).get_map_meta()


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


@router.post("/api/assembly-selection/generate")
def assembly_generate(pipeline: str = "live") -> dict[str, Any]:
    return _assembly(pipeline).add_to_map()
