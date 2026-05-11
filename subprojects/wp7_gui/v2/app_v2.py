import contextlib
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import quote

import imageio.v2 as imageio
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from utilities.config_interface import fetch_wp4_configs

from subprojects.wp7_gui.v2.functions.assembly_selection_service import AssemblySelectionService
from subprojects.wp7_gui.v2.functions.pretraining_service import PretrainingService

# Prefer repository-local packages over site-packages for GUI/runtime helpers.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

REPO_ROOT = Path(__file__).resolve().parents[2]
GUI_ROOT = Path(__file__).resolve().parent
LEGACY_GUI_ROOT = GUI_ROOT.parent
TEMPLATES_DIR = GUI_ROOT / "templates"
PAGES_DIR = TEMPLATES_DIR / "pages"

WP4_PACKAGE_ROOT = REPO_ROOT / "wp4_aeroshell_optimizer"


def _detect_wp4_active_version() -> str:
    """Read ACTIVE_VERSION from wp4 package without importing heavy modules."""
    init_path = WP4_PACKAGE_ROOT / "__init__.py"
    try:
        text = init_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "v2"

    match = re.search(r"ACTIVE_VERSION\s*=\s*[\"']([^\"']+)[\"']", text)
    if match is None:
        return "v2"
    return match.group(1)


WP4_ACTIVE_VERSION = _detect_wp4_active_version()
WP4_ACTIVE_ROOT = WP4_PACKAGE_ROOT / WP4_ACTIVE_VERSION
_WP4_CONFIGS_DIR = WP4_ACTIVE_ROOT / "configs"
_WP4_SLURM_DIR = WP4_ACTIVE_ROOT / "slurm_scripts"

RUN_LOCAL_SCRIPT = _WP4_SLURM_DIR / "run_local.sh"
RUN_LOCAL_LOG = GUI_ROOT / "logs" / f"run_local_{timestamp}.log"
GENERAL_CONFIG_PATH = _WP4_CONFIGS_DIR / "general_config.yaml"
DEFAULT_LOCAL_STORAGE_DIR = REPO_ROOT.parent / "local_storage_dir"
DEFAULT_BULK_STORAGE_DIR = Path("/cluster/work/fuge/AISCO/bulk_storage")
PRETRAINING_STORAGE_ROOT = Path("/home/aisco/pretraining_gifs")
SLOW_GIF_CACHE_DIR = GUI_ROOT / "cache"
MESH_ASSEMBLY_DIR_CANDIDATES = ("mesh_assembly_mode_v4", "mesh_assembly_mode_v3", "mesh_assembly_mode_v2", "mesh_assembly_mode")


PROCESS_LOCK = threading.Lock()
STATE: dict[str, subprocess.Popen[str] | None] = {"run_local_process": None}
ASSEMBLY_ID_ENV_VAR = "AISCO_ASSEMBLY_ID"
GUI_PAGE_FILES: dict[str, Path] = {
    "introduction": PAGES_DIR / "introduction.html",
    "leaderboard": PAGES_DIR / "leaderboard.html",
    "pretraining": PAGES_DIR / "pretraining.html",
    "optimization": PAGES_DIR / "optimization.html",
    "gradient_viewer": PAGES_DIR / "gradient_viewer.html",
    "solar": PAGES_DIR / "gradient_viewer.html",
    "assembly_selection": PAGES_DIR / "assembly_selection.html",
    "testing": PAGES_DIR / "testing.html",
    "config_overrides": PAGES_DIR / "config_overrides.html",
}

# Config-override session management
CONFIG_OVERRIDES_ROOT = GUI_ROOT / "config_overrides"
GUI_OVERRIDE_PREFIX = "gui_override_"
GUI_SCHEMA_PATH = GUI_ROOT / "gui_schema.json"
COPY_EXCLUDE_SUBFOLDERS = ("sweeps",)


def _get_default_storage_root() -> Path:
    """Get the default storage root from configuration or fall back to default."""
    try:
        with GENERAL_CONFIG_PATH.open(encoding="utf-8") as f:
            general_cfg = yaml.safe_load(f) or {}
        storage_paths = general_cfg.get("storage_paths", {})
        run_locally = str(general_cfg.get("general_info", {}).get("run_locally", "False")).lower() == "true"
        
        if run_locally:
            raw_root = storage_paths.get("local_storage_dir", str(DEFAULT_LOCAL_STORAGE_DIR))
        else:
            raw_root = storage_paths.get("storage_dir", str(DEFAULT_BULK_STORAGE_DIR))
        
        root = Path(raw_root)
        return (REPO_ROOT.parent / root).resolve() if not root.is_absolute() else root.resolve()
    except Exception:
        return DEFAULT_LOCAL_STORAGE_DIR.resolve()


# Render output paths - these are derived from storage root subdirectories
# They will be populated with actual render outputs during run execution
_STORAGE_ROOT = _get_default_storage_root()
SOLAR_RENDERS_ROOT = _STORAGE_ROOT
MESH_ASSEMBLY_RENDERS_DIR = _STORAGE_ROOT
PRESSURE_VISUALS_ROOT = _STORAGE_ROOT


class RunLocalRequest(BaseModel):
    """Request payload for launching a local optimization run.

    Args:
        assembly_id: Optional assembly ID to optimize. Empty string is treated as random.

    Returns:
        None
    """

    assembly_id: int | str | None = None


class ConfigOverrideRequest(BaseModel):
    """Request payload for running with config overrides.

    Args:
        uses_gui: Whether running from GUI
        cfg_folder: Path to config override folder
    """

    uses_gui: bool = False
    cfg_folder: str = ""


class OverrideValuesRequest(BaseModel):
    """Body for PUT /config-overrides/values.

    updates is keyed by config filename relative to wp4/v2/configs/ (e.g. "training_v0_config.yaml"
    or "solar_configs/solar_loss_config.yaml"); each value is a flat dict from dotted leaf path to new value.
    """

    folder: str
    updates: dict[str, dict[str, Any]]


class CleanupRequest(BaseModel):
    folders: list[str] | None = None
    all: bool = False


class GUIRequestHandler:
    SOLAR_RENDER_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "aero": ("MeshSDF_grads_aero",),
        "solar": ("MeshSDF_grads_solar", "MeshSDF_grad_solar"),
        "total": ("MeshSDF_grads_total",),
    }

    @staticmethod
    def _coerce_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return default

    def _get_storage_root(self) -> Path:
        """Resolve artifact root from wp4 general_config run_locally + storage_paths."""
        try:
            with GENERAL_CONFIG_PATH.open(encoding="utf-8") as f:
                general_cfg = yaml.safe_load(f) or {}

            run_locally = self._coerce_bool(general_cfg.get("general_info", {}).get("run_locally", False), default=False)
            storage_paths = general_cfg.get("storage_paths", {})

            if run_locally:
                raw_root = storage_paths.get("local_storage_dir", str(DEFAULT_LOCAL_STORAGE_DIR))
            else:
                raw_root = storage_paths.get("storage_dir", str(DEFAULT_BULK_STORAGE_DIR))

            root = Path(raw_root)
            root = (REPO_ROOT.parent / root).resolve() if not root.is_absolute() else root.resolve()

            return root  # noqa: TRY300
        except Exception as error:
            LOGGER.warning("Falling back to default local storage root due to config read error: %s", error)
            return DEFAULT_LOCAL_STORAGE_DIR.resolve()

    def handle_wp2_params(self) -> dict[str, dict[str, float]]:
        return assembly_selection_service.get_params()

    def handle_wp2_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        return assembly_selection_service.update_params(payload)

    def handle_wp2_add_wheel(self) -> dict[str, Any]:
        return assembly_selection_service.add_wheel()

    def handle_wp2_remove_wheel(self) -> dict[str, Any]:
        return assembly_selection_service.remove_wheel()

    @staticmethod
    def handle_wp2_mesh() -> FileResponse:
        return assembly_selection_service.get_mesh_response()

    def handle_pretraining_gifs(self) -> dict[str, Any]:
        return pretraining_service.list_gifs()

    @staticmethod
    def handle_pretraining_gif_file(rel_path: str) -> FileResponse:
        return pretraining_service.get_gif_response(rel_path)

    def _get_solar_render_dir_for_group(self, group: str) -> Path | None:
        folder_candidates = self.SOLAR_RENDER_GROUPS.get(group)
        if folder_candidates is None:
            return None

        for folder in folder_candidates:
            candidate = (SOLAR_RENDERS_ROOT / folder).resolve()
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    @staticmethod
    def _extract_render_frame_index(filename: str) -> int | None:
        match = re.search(r"cam\d+_(\d+)_", filename)
        if match is None:
            return None
        return int(match.group(1))

    @staticmethod
    def _natural_sort_key(value: str) -> list[Any]:
        # Keep visual order aligned with typical folder explorers for numbered filenames.
        parts = re.split(r"(\d+)", value)
        return [int(part) if part.isdigit() else part.lower() for part in parts]

    def handle_solar_render_frames(self) -> dict[str, Any]:
        if not SOLAR_RENDERS_ROOT.exists() or not SOLAR_RENDERS_ROOT.is_dir():
            return {
                "base_dir": str(SOLAR_RENDERS_ROOT),
                "groups": {"aero": [], "solar": [], "total": []},
                "frame_indices": [],
                "message": "Solar renders root does not exist.",
            }

        groups_payload: dict[str, list[dict[str, Any]]] = {}
        frame_index_values: set[int] = set()
        valid_ext = {".png", ".jpg", ".jpeg", ".webp"}

        for group_key in ("aero", "solar", "total"):
            group_dir = self._get_solar_render_dir_for_group(group_key)
            entries: list[dict[str, Any]] = []

            if group_dir is not None:
                image_paths = [path for path in group_dir.iterdir() if path.is_file() and path.suffix.lower() in valid_ext]

                image_paths.sort(
                    key=lambda p: (
                        self._extract_render_frame_index(p.name) is None,
                        self._extract_render_frame_index(p.name) or 0,
                        p.name,
                    )
                )

                for image_path in image_paths:
                    frame_index = self._extract_render_frame_index(image_path.name)
                    if frame_index is not None:
                        frame_index_values.add(frame_index)

                    entries.append(
                        {
                            "name": image_path.name,
                            "index": frame_index,
                            "folder": group_dir.name,
                            "url": (f"/solar/render-file?group={quote(group_key)}&name={quote(image_path.name)}"),
                        }
                    )

            groups_payload[group_key] = entries

        return {
            "base_dir": str(SOLAR_RENDERS_ROOT),
            "groups": groups_payload,
            "frame_indices": sorted(frame_index_values),
            "message": "Solar renders loaded",
        }

    def handle_solar_render_file(self, group: str, filename: str) -> FileResponse:
        group_dir = self._get_solar_render_dir_for_group(group)
        if group_dir is None:
            raise HTTPException(status_code=400, detail={"message": f"Invalid group: {group}"})

        requested = (group_dir / filename).resolve()
        try:
            requested.relative_to(group_dir)
        except Exception:
            raise HTTPException(status_code=400, detail={"message": "Invalid image path"}) from None

        if not requested.exists() or not requested.is_file():
            raise HTTPException(status_code=404, detail={"message": "Render image not found"})

        mime_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        return FileResponse(str(requested), media_type=mime_type, headers={"Cache-Control": "no-store"})

    def handle_run_local(self, payload: RunLocalRequest) -> JSONResponse:
        raw_assembly_id = payload.assembly_id

        assembly_id: int | None = None
        if raw_assembly_id not in (None, ""):
            raw_assembly_id_str = str(raw_assembly_id).strip()
            try:
                assembly_id = int(raw_assembly_id_str)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail={"message": "assembly_id must be an integer between 1 and 51"}
                ) from None

            if not 1 <= assembly_id <= 51:
                raise HTTPException(status_code=400, detail={"message": "assembly_id must be in range 1..51"})

        if not RUN_LOCAL_SCRIPT.exists():
            LOGGER.error("Script missing: %s", RUN_LOCAL_SCRIPT)
            raise HTTPException(status_code=500, detail={"message": f"Optimization script not found at {RUN_LOCAL_SCRIPT}"})

        try:
            with PROCESS_LOCK:
                run_local_process = STATE["run_local_process"]
                if run_local_process is not None and run_local_process.poll() is None:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "running",
                            "message": "run_local.sh is already running",
                            "pid": run_local_process.pid,
                        },
                    )

                start_stamp = datetime.now().isoformat(timespec="seconds")
                run_label = "random assembly_id" if assembly_id is None else f"assembly_id={assembly_id}"
                RUN_LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
                RUN_LOCAL_LOG.write_text(
                    f"[{start_stamp}] Starting run_local.sh ({run_label})\n\n",
                    encoding="utf-8",
                )

                process_env = os.environ.copy()
                if assembly_id is not None:
                    process_env[ASSEMBLY_ID_ENV_VAR] = str(assembly_id)
                else:
                    process_env.pop(ASSEMBLY_ID_ENV_VAR, None)

                log_handle = RUN_LOCAL_LOG.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    ["bash", str(RUN_LOCAL_SCRIPT)],
                    cwd=str(REPO_ROOT.parent),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=process_env,
                    start_new_session=True,
                )
                log_handle.close()
                STATE["run_local_process"] = process

            LOGGER.info("Started optimization script. PID: %s", process.pid)
            return JSONResponse(
                status_code=200,
                content={
                    "status": "success",
                    "message": "Optimization process started",
                    "pid": process.pid,
                    "assembly_id": assembly_id,
                },
            )
        except HTTPException:
            raise
        except Exception as error:
            LOGGER.exception("Failed to launch subprocess")
            raise HTTPException(status_code=500, detail={"message": str(error)}) from error

    def handle_test_cli(self, payload: ConfigOverrideRequest) -> JSONResponse:
        """Execute main_v2.py with config override arguments."""
        if not RUN_LOCAL_SCRIPT.exists():
            LOGGER.error("Script missing: %s", RUN_LOCAL_SCRIPT)
            raise HTTPException(status_code=500, detail={"message": f"Script not found at {RUN_LOCAL_SCRIPT}"})

        try:
            # Build CLI arguments from request
            cli_args = []
            if payload.uses_gui:
                cli_args.append("--uses-gui")
            if payload.cfg_folder and payload.cfg_folder.strip():
                cli_args.extend(["--cfg-folder", payload.cfg_folder.strip()])
            # Start as a background process and stream to the standard run log so long-running
            # runs don't block the FastAPI worker or get killed by upstream timeouts.
            with PROCESS_LOCK:
                run_local_process = STATE["run_local_process"]
                if run_local_process is not None and run_local_process.poll() is None:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "running",
                            "message": "run_local.sh is already running",
                            "pid": run_local_process.pid,
                        },
                    )

                start_stamp = datetime.now().isoformat(timespec="seconds")
                run_label = f"config-overrides: {cli_args or 'none'}"
                RUN_LOCAL_LOG.parent.mkdir(parents=True, exist_ok=True)
                RUN_LOCAL_LOG.write_text(
                    f"[{start_stamp}] Starting run_local.sh ({run_label})\n\n",
                    encoding="utf-8",
                )

                process_env = os.environ.copy()

                log_handle = RUN_LOCAL_LOG.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    ["bash", str(RUN_LOCAL_SCRIPT), *cli_args],
                    cwd=str(REPO_ROOT.parent),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=process_env,
                    start_new_session=True,
                )
                log_handle.close()
                STATE["run_local_process"] = process

            LOGGER.info("Started config-override script. PID: %s, args: %s", process.pid, cli_args)
            return JSONResponse(
                status_code=200,
                content={
                    "status": "started",
                    "message": "Process started; poll /run-local/logs for output",
                    "pid": process.pid,
                    "cli_args": cli_args,
                },
            )
        except Exception as error:
            LOGGER.exception("Failed to run with config overrides")
            raise HTTPException(status_code=500, detail={"message": str(error)}) from error

    def handle_run_local_logs(self) -> dict[str, Any]:
        max_chars = 60000
        logs = ""
        truncated = False

        if RUN_LOCAL_LOG.exists():
            logs = RUN_LOCAL_LOG.read_text(encoding="utf-8", errors="replace")
            if len(logs) > max_chars:
                logs = logs[-max_chars:]
                truncated = True

        with PROCESS_LOCK:
            process = STATE["run_local_process"]
            running = process is not None and process.poll() is None
            pid = process.pid if process is not None else None
            return_code = None if running or process is None else process.poll()

        return {
            "logs": logs,
            "running": running,
            "pid": pid,
            "return_code": return_code,
            "truncated": truncated,
        }

    def handle_stop_run_local(self) -> JSONResponse:
        """Stop the currently running local process (if any)."""
        with PROCESS_LOCK:
            process = STATE.get("run_local_process")
            if process is None or process.poll() is not None:
                return JSONResponse(status_code=200, content={"status": "not_running", "message": "No active run to stop"})

            try:
                # Kill the whole process group to ensure children are terminated
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                with contextlib.suppress(Exception):
                    process.terminate()

            return JSONResponse(
                status_code=200, content={"status": "stopping", "pid": process.pid, "message": "Sent termination signal"}
            )

    def handle_run_local_progress(self) -> dict[str, Any]:
        num_epochs = self._get_num_epochs_from_training_config()
        current_epoch = -1
        progress_percent = 0.0
        completed = False

        if RUN_LOCAL_LOG.exists():
            logs = RUN_LOCAL_LOG.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"\[Epoch\s+(\d+)\]", logs)
            if matches:
                current_epoch = max(int(m) for m in matches)
                progress_percent = min(100.0, ((current_epoch + 1) / max(num_epochs, 1)) * 100.0)

            if "Model optimisation completed." in logs or "Completed job successfully." in logs:
                completed = True
                progress_percent = 100.0

        with PROCESS_LOCK:
            process = STATE["run_local_process"]
            running = process is not None and process.poll() is None
            return_code = None if running or process is None else process.poll()

        return {
            "running": running,
            "completed": completed,
            "return_code": return_code,
            "num_epochs": num_epochs,
            "current_epoch": current_epoch,
            "progress_percent": round(progress_percent, 1),
        }

    def handle_run_local_artifacts(self) -> dict[str, Any]:
        storage_root = self._get_storage_root()
        cam1_renders = self._list_cam1_render_images()
        pressure_field_images = self._list_pressure_field_images()
        run_dir = self._detect_current_run_dir()
        if run_dir is None:
            return {
                "run_dir": None,
                "mesh_assembly_gifs": [],
                "solar_loss_gifs": [],
                "cam1_renders": cam1_renders,
                "pressure_field_images": pressure_field_images,
                "meshes": [],
                "storage_root": str(storage_root),
                "message": "No completed run output directory found yet.",
            }

        mesh_assembly_dir = self._find_mesh_assembly_dir(run_dir)
        solar_loss_dir = run_dir / "solar_loss_rendering"
        mesh_dir = run_dir / "meshes"

        gifs = []
        if mesh_assembly_dir is not None and mesh_assembly_dir.exists():
            for gif_path in sorted(mesh_assembly_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel_path = gif_path.resolve().relative_to(storage_root.resolve()).as_posix()
                gifs.append(
                    {
                        "name": gif_path.name,
                        "rel_path": rel_path,
                        "url": f"/run-local/artifact-file?path={rel_path}",
                        "slow_url": f"/run-local/slow-gif?path={rel_path}&factor=2.5",
                    }
                )

        solar_loss_gifs = []
        if solar_loss_dir.exists():
            for gif_path in sorted(solar_loss_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel_path = gif_path.resolve().relative_to(storage_root.resolve()).as_posix()
                solar_loss_gifs.append(
                    {
                        "name": gif_path.name,
                        "rel_path": rel_path,
                        "url": f"/run-local/artifact-file?path={rel_path}",
                        "slow_url": f"/run-local/slow-gif?path={rel_path}&factor=2.5",
                    }
                )

        meshes = []
        if mesh_dir.exists():
            mesh_paths = sorted(list(mesh_dir.glob("*.ply")) + list(mesh_dir.glob("*.poly")))
            for mesh_path in mesh_paths:
                rel_path = mesh_path.resolve().relative_to(storage_root.resolve()).as_posix()
                epoch = self._extract_epoch(mesh_path.name)
                meshes.append(
                    {
                        "name": mesh_path.name,
                        "rel_path": rel_path,
                        "url": f"/run-local/artifact-file?path={rel_path}",
                        "epoch": epoch,
                    }
                )

            meshes.sort(key=lambda item: item["epoch"] if item["epoch"] is not None else -1)

        return {
            "run_dir": str(run_dir.resolve()),
            "storage_root": str(storage_root),
            "mesh_assembly_gifs": gifs,
            "solar_loss_gifs": solar_loss_gifs,
            "cam1_renders": cam1_renders,
            "pressure_field_images": pressure_field_images,
            "meshes": meshes,
            "message": "Artifacts loaded",
        }

    def _list_cam1_render_images(self) -> list[dict[str, Any]]:
        if not MESH_ASSEMBLY_RENDERS_DIR.exists() or not MESH_ASSEMBLY_RENDERS_DIR.is_dir():
            return []

        cam1_candidates: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            cam1_candidates.extend(MESH_ASSEMBLY_RENDERS_DIR.glob(f"cam1*{ext[1:]}"))

        cam1_candidates.sort(key=lambda p: (self._extract_render_frame_index(p.name) or -1, p.name))

        result: list[dict[str, Any]] = []
        for image_path in cam1_candidates:
            rel_path = image_path.resolve().relative_to(SOLAR_RENDERS_ROOT.resolve()).as_posix()
            result.append(
                {
                    "name": image_path.name,
                    "rel_path": rel_path,
                    "url": f"/run-local/wandb-visual-file?path={quote(rel_path)}",
                    "frame_index": self._extract_render_frame_index(image_path.name),
                }
            )
        return result

    def _list_pressure_field_images(self) -> list[dict[str, Any]]:
        if not PRESSURE_VISUALS_ROOT.exists() or not PRESSURE_VISUALS_ROOT.is_dir():
            return []

        all_candidates: list[Path] = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            all_candidates.extend(PRESSURE_VISUALS_ROOT.rglob(ext))

        pressure_candidates = [
            path for path in all_candidates if "pressure" in path.name.lower() or "pressure" in path.parent.name.lower()
        ]

        # Fallback for runs where pressure images are stored in the renders tree without explicit naming.
        if not pressure_candidates:
            pressure_candidates = [
                path for path in all_candidates if "mesh_assembly_mode_v4" not in {part.lower() for part in path.parts}
            ]

        pressure_candidates.sort(
            key=lambda p: self._natural_sort_key(p.resolve().relative_to(PRESSURE_VISUALS_ROOT.resolve()).as_posix())
        )

        result: list[dict[str, Any]] = []
        for image_path in pressure_candidates:
            rel_path = image_path.resolve().relative_to(PRESSURE_VISUALS_ROOT.resolve()).as_posix()
            result.append(
                {
                    "name": image_path.name,
                    "rel_path": rel_path,
                    "url": f"/run-local/wandb-pressure-file?path={quote(rel_path)}",
                    "epoch": self._extract_render_frame_index(image_path.name),
                }
            )
        return result

    def handle_wandb_visual_file(self, rel_path: str) -> FileResponse:
        rel_path = rel_path.lstrip("/")
        if not rel_path:
            raise HTTPException(status_code=400, detail={"message": "Missing path query parameter"})

        try:
            requested = (SOLAR_RENDERS_ROOT.resolve() / rel_path).resolve()
            requested.relative_to(SOLAR_RENDERS_ROOT.resolve())
        except Exception:
            raise HTTPException(status_code=400, detail={"message": "Invalid visual path"}) from None

        if not requested.exists() or not requested.is_file():
            raise HTTPException(status_code=404, detail={"message": "Visual file not found"})

        mime_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        return FileResponse(str(requested), media_type=mime_type, headers={"Cache-Control": "no-store"})

    def handle_wandb_pressure_file(self, rel_path: str) -> FileResponse:
        rel_path = rel_path.lstrip("/")
        if not rel_path:
            raise HTTPException(status_code=400, detail={"message": "Missing path query parameter"})

        try:
            requested = (PRESSURE_VISUALS_ROOT.resolve() / rel_path).resolve()
            requested.relative_to(PRESSURE_VISUALS_ROOT.resolve())
        except Exception:
            raise HTTPException(status_code=400, detail={"message": "Invalid pressure visual path"}) from None

        if not requested.exists() or not requested.is_file():
            raise HTTPException(status_code=404, detail={"message": "Pressure visual file not found"})

        mime_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        return FileResponse(str(requested), media_type=mime_type, headers={"Cache-Control": "no-store"})

    def handle_artifact_file(self, rel_path: str) -> FileResponse:
        rel_path = rel_path.lstrip("/")
        if not rel_path:
            raise HTTPException(status_code=400, detail={"message": "Missing path query parameter"})

        requested = self._resolve_storage_file(rel_path)
        if requested is None:
            raise HTTPException(status_code=400, detail={"message": "Invalid artifact path"})

        if not requested.exists() or not requested.is_file():
            raise HTTPException(status_code=404, detail={"message": "Artifact file not found"})

        mime_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        return FileResponse(str(requested), media_type=mime_type, headers={"Cache-Control": "no-store"})

    def handle_slow_gif(self, rel_path: str, factor_value: str = "2.5") -> FileResponse:
        rel_path = rel_path.lstrip("/")
        if not rel_path:
            raise HTTPException(status_code=400, detail={"message": "Missing path query parameter"})

        try:
            factor = float(factor_value)
            factor = max(1.0, min(factor, 10.0))
        except ValueError:
            raise HTTPException(status_code=400, detail={"message": "Invalid factor value"}) from None

        requested = self._resolve_storage_file(rel_path)
        if requested is None:
            raise HTTPException(status_code=400, detail={"message": "Invalid artifact path"})
        if not requested.exists() or not requested.is_file():
            raise HTTPException(status_code=404, detail={"message": "Artifact file not found"})
        if requested.suffix.lower() != ".gif":
            raise HTTPException(status_code=400, detail={"message": "Only GIF files are supported for slow playback"})

        cache_key = hashlib.sha1(f"{requested.resolve()}|{requested.stat().st_mtime_ns}|{factor:.2f}".encode()).hexdigest()
        SLOW_GIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = SLOW_GIF_CACHE_DIR / f"slow_{cache_key}.gif"

        if not cached_path.exists():
            if imageio is None:
                raise HTTPException(status_code=500, detail={"message": "imageio is required for slow GIF generation"})
            try:
                frames = imageio.mimread(requested)
                meta = imageio.get_reader(requested).get_meta_data()
                base_duration = float(meta.get("duration", 0.08))
                slow_duration = max(0.02, base_duration * factor)
            except Exception as error:
                raise HTTPException(status_code=500, detail={"message": f"Failed to build slowed GIF: {error}"}) from error

            if not frames:
                raise HTTPException(status_code=500, detail={"message": "Failed to decode GIF frames"})

            imageio.mimsave(cached_path, cast(list[Any], frames), duration=slow_duration)

        return FileResponse(str(cached_path), media_type="image/gif", headers={"Cache-Control": "no-store"})

    def _resolve_storage_file(self, rel_path: str) -> Path | None:
        try:
            storage_root = self._get_storage_root().resolve()
            requested = (storage_root / rel_path).resolve()
            requested.relative_to(storage_root)
        except Exception:
            return None
        else:
            return requested

    def _get_num_epochs_from_training_config(self) -> int:
        try:
            general_config_path = _WP4_CONFIGS_DIR / "general_config.yaml"
            with general_config_path.open(encoding="utf-8") as f:
                general_cfg = yaml.safe_load(f) or {}

            training_cfg_name = general_cfg.get("active_configs", {}).get("training", "training_v0_config.yaml")
            training_cfg_path = _WP4_CONFIGS_DIR / training_cfg_name
            with training_cfg_path.open(encoding="utf-8") as f:
                training_cfg = yaml.safe_load(f) or {}

            return int(training_cfg.get("training_params", {}).get("num_epochs", 100))
        except ModuleNotFoundError:
            return 100
        except Exception:
            return 100

    @staticmethod
    def _find_mesh_assembly_dir(run_path: Path) -> Path | None:
        for candidate in MESH_ASSEMBLY_DIR_CANDIDATES:
            candidate_dir = run_path / candidate
            if candidate_dir.exists() and candidate_dir.is_dir():
                return candidate_dir
        return None

    @staticmethod
    def _has_any_gif_artifacts(run_path: Path) -> bool:
        mesh_assembly_dir = GUIRequestHandler._find_mesh_assembly_dir(run_path)
        mesh_assembly_gifs = [] if mesh_assembly_dir is None else list(mesh_assembly_dir.glob("*.gif"))
        solar_loss_gifs = list((run_path / "solar_loss_rendering").glob("*.gif"))
        return bool(mesh_assembly_gifs or solar_loss_gifs)

    def _has_display_artifacts(self, run_path: Path) -> bool:
        meshes = list((run_path / "meshes").glob("*.ply")) + list((run_path / "meshes").glob("*.poly"))
        return self._has_any_gif_artifacts(run_path) or bool(meshes)

    def _collect_candidate_logs(self) -> list[Path]:
        candidate_logs: list[Path] = []
        logs_dirs = [GUI_ROOT / "logs"]
        legacy_logs_dir = LEGACY_GUI_ROOT / "logs"
        if legacy_logs_dir != logs_dirs[0]:
            logs_dirs.append(legacy_logs_dir)

        if RUN_LOCAL_LOG.exists():
            candidate_logs.append(RUN_LOCAL_LOG)

        for logs_dir in logs_dirs:
            if not logs_dir.exists():
                continue
            historical_logs = sorted(logs_dir.glob("run_local_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            for path in historical_logs:
                if path not in candidate_logs:
                    candidate_logs.append(path)

        return candidate_logs

    def _get_fallback_run_from_logs(self) -> Path | None:
        fallback_logged_run_with_meshes: Path | None = None
        for log_path in self._collect_candidate_logs():
            logs = log_path.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"Output directory:\s*(.+)", logs)
            if not matches:
                continue

            run_dir = Path(matches[-1].strip())
            run_dir = (REPO_ROOT.parent / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()
            if not (run_dir.exists() and run_dir.is_dir()):
                continue

            if self._has_any_gif_artifacts(run_dir):
                return run_dir
            if self._has_display_artifacts(run_dir) and fallback_logged_run_with_meshes is None:
                fallback_logged_run_with_meshes = run_dir

        return fallback_logged_run_with_meshes

    def _collect_run_dirs(self, storage_root: Path) -> list[Path]:
        run_dirs: list[Path] = []
        for model_dir in storage_root.iterdir():
            if not model_dir.is_dir():
                continue
            for child in model_dir.iterdir():
                has_meshes = (child / "meshes").exists()
                has_mesh_assembly = self._find_mesh_assembly_dir(child) is not None
                if child.is_dir() and (has_meshes or has_mesh_assembly):
                    run_dirs.append(child)
        return run_dirs

    def _detect_current_run_dir(self) -> Path | None:
        fallback_logged_run_with_meshes = self._get_fallback_run_from_logs()

        storage_root = self._get_storage_root()
        if not storage_root.exists():
            return None

        run_dirs = self._collect_run_dirs(storage_root)

        if not run_dirs:
            return fallback_logged_run_with_meshes

        run_dirs_with_gifs = [path for path in run_dirs if self._has_any_gif_artifacts(path)]
        if run_dirs_with_gifs:
            return max(run_dirs_with_gifs, key=lambda p: p.stat().st_mtime)

        run_dirs_with_artifacts = [path for path in run_dirs if self._has_display_artifacts(path)]
        if run_dirs_with_artifacts:
            return max(run_dirs_with_artifacts, key=lambda p: p.stat().st_mtime)

        if fallback_logged_run_with_meshes is not None:
            return fallback_logged_run_with_meshes

        return max(run_dirs, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _extract_epoch(filename: str) -> int | None:
        match = re.search(r"epoch(\d+)", filename)
        if match is None:
            match = re.search(r"_(\d+)\.(?:png|gif|ply|poly)$", filename)
        if match is None:
            return None
        return int(match.group(1))


class ConfigOverrideManager:
    """Manages GUI config-override sessions: copy YAMLs, validate edits, persist back, cleanup."""

    def _load_schema(self) -> dict[str, Any]:
        if not GUI_SCHEMA_PATH.exists():
            raise HTTPException(status_code=500, detail={"message": f"Schema manifest missing: {GUI_SCHEMA_PATH}"})
        try:
            with GUI_SCHEMA_PATH.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=500, detail={"message": f"Invalid gui_schema.json: {error}"}) from error

    def _resolve_session_folder(self, folder_str: str) -> Path:
        """Resolve a session folder argument and reject anything outside CONFIG_OVERRIDES_ROOT or missing the prefix."""
        try:
            candidate = Path(folder_str).resolve()
            root = CONFIG_OVERRIDES_ROOT.resolve()
            candidate.relative_to(root)
        except (ValueError, OSError) as error:
            raise HTTPException(
                status_code=400, detail={"message": f"Folder is outside the override root: {folder_str}"}
            ) from error
        if not candidate.name.startswith(GUI_OVERRIDE_PREFIX):
            raise HTTPException(
                status_code=400,
                detail={"message": f"Folder name must start with '{GUI_OVERRIDE_PREFIX}': {candidate.name}"},
            )
        return candidate

    def init_session(self) -> dict[str, Any]:
        CONFIG_OVERRIDES_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = CONFIG_OVERRIDES_ROOT / f"{GUI_OVERRIDE_PREFIX}{stamp}"
        if session_dir.exists():
            # Avoid clobbering on rapid double-clicks within the same second.
            session_dir = CONFIG_OVERRIDES_ROOT / f"{GUI_OVERRIDE_PREFIX}{stamp}_{os.getpid()}"
        session_dir.mkdir(parents=True)

        source_yaml_paths = fetch_wp4_configs(str(REPO_ROOT.parent), exclude_subfolders=COPY_EXCLUDE_SUBFOLDERS)
        wp4_configs_root = (REPO_ROOT / "wp4_aeroshell_optimizer" / "v2" / "configs").resolve()

        copied: list[str] = []
        for src_str in source_yaml_paths:
            src = Path(src_str).resolve()
            rel = src.relative_to(wp4_configs_root)
            dst = session_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(rel))

        return {
            "folder": str(session_dir),
            "name": session_dir.name,
            "copied_files": copied,
            "skipped_subfolders": list(COPY_EXCLUDE_SUBFOLDERS),
        }

    def get_schema(self) -> dict[str, Any]:
        return self._load_schema()

    def get_values(self, folder_str: str) -> dict[str, Any]:
        session_dir = self._resolve_session_folder(folder_str)
        schema = self._load_schema()
        result: dict[str, dict[str, Any]] = {}
        for cfg_name, cfg_entry in schema.get("configs", {}).items():
            yaml_path = session_dir / cfg_name
            if not yaml_path.exists():
                LOGGER.warning("Schema references missing YAML in session: %s", yaml_path)
                continue
            with yaml_path.open(encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}
            param_values: dict[str, Any] = {}
            for param in cfg_entry.get("params", []):
                dotted_path = param["path"]
                param_values[dotted_path] = self._read_dotted(cfg_dict, dotted_path.split("."))
            result[cfg_name] = param_values
        return {"folder": str(session_dir), "values": result}

    def set_values(self, payload: OverrideValuesRequest) -> dict[str, Any]:
        session_dir = self._resolve_session_folder(payload.folder)
        schema = self._load_schema()
        schema_configs = schema.get("configs", {})

        # Validate everything before writing anything — fail-fast keeps files consistent.
        validated: dict[str, list[tuple[list[str], Any]]] = {}
        for cfg_name, updates in payload.updates.items():
            if cfg_name not in schema_configs:
                raise HTTPException(status_code=400, detail={"message": f"Config not editable via GUI: {cfg_name}"})
            param_lookup = {p["path"]: p for p in schema_configs[cfg_name].get("params", [])}
            cfg_validated: list[tuple[list[str], Any]] = []
            for dotted_path, raw_value in updates.items():
                if dotted_path not in param_lookup:
                    raise HTTPException(
                        status_code=400,
                        detail={"message": f"Parameter not in schema whitelist: {cfg_name}::{dotted_path}"},
                    )
                coerced = self._validate_value(param_lookup[dotted_path], raw_value, cfg_name)
                cfg_validated.append((dotted_path.split("."), coerced))
            validated[cfg_name] = cfg_validated

        # Apply: load → mutate → dump. Keep original key order via yaml's default behaviour on dicts.
        written: list[str] = []
        for cfg_name, segments_and_values in validated.items():
            yaml_path = session_dir / cfg_name
            if not yaml_path.exists():
                raise HTTPException(status_code=404, detail={"message": f"YAML file missing in session: {cfg_name}"})
            with yaml_path.open(encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}
            for segments, value in segments_and_values:
                self._write_dotted(cfg_dict, segments, value)
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_dict, f, sort_keys=False, default_flow_style=False)
            written.append(cfg_name)

        return {"folder": str(session_dir), "written": written}

    def list_sessions(self) -> dict[str, Any]:
        if not CONFIG_OVERRIDES_ROOT.exists():
            return {"sessions": []}
        sessions = []
        for entry in sorted(CONFIG_OVERRIDES_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if entry.is_dir() and entry.name.startswith(GUI_OVERRIDE_PREFIX):
                sessions.append(  # noqa: PERF401
                    {
                        "name": entry.name,
                        "folder": str(entry),
                        "mtime": entry.stat().st_mtime,
                    }
                )
        return {"sessions": sessions}

    def cleanup(self, payload: CleanupRequest) -> dict[str, Any]:
        removed: list[str] = []
        if payload.all:
            if not CONFIG_OVERRIDES_ROOT.exists():
                return {"removed": []}
            for entry in CONFIG_OVERRIDES_ROOT.iterdir():
                if entry.is_dir() and entry.name.startswith(GUI_OVERRIDE_PREFIX):
                    shutil.rmtree(entry)
                    removed.append(entry.name)
            return {"removed": removed}

        for folder_str in payload.folders or []:
            session_dir = self._resolve_session_folder(folder_str)
            if session_dir.exists():
                shutil.rmtree(session_dir)
                removed.append(session_dir.name)
        return {"removed": removed}

    @staticmethod
    def _read_dotted(cfg_dict: dict, segments: list[str]) -> Any:
        d: Any = cfg_dict
        for seg in segments:
            if not isinstance(d, dict) or seg not in d:
                return None
            d = d[seg]
        return d

    @staticmethod
    def _write_dotted(cfg_dict: dict, segments: list[str], value: Any) -> None:
        d = cfg_dict
        for seg in segments[:-1]:
            if seg not in d or not isinstance(d[seg], dict):
                d[seg] = {}
            d = d[seg]
        d[segments[-1]] = value

    @staticmethod
    def _validate_value(param_schema: dict[str, Any], raw_value: Any, cfg_name: str) -> Any:  # noqa: PLR0912
        ptype = param_schema.get("type")
        path = param_schema.get("path", "?")
        loc = f"{cfg_name}::{path}"
        coerced: int | float

        if ptype == "int":
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                # Accept numeric strings too, but reject floats with non-zero fractional part.
                try:
                    coerced = int(raw_value)
                    if isinstance(raw_value, float) and not raw_value.is_integer():
                        raise ValueError  # noqa: TRY301
                except (TypeError, ValueError) as error:
                    raise HTTPException(status_code=400, detail={"message": f"{loc}: expected int, got {raw_value!r}"}) from error
            else:
                coerced = raw_value
            if "min" in param_schema and coerced < param_schema["min"]:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} < min {param_schema['min']}"})
            if "max" in param_schema and coerced > param_schema["max"]:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} > max {param_schema['max']}"})
            return coerced

        if ptype == "float":
            if isinstance(raw_value, bool):
                raise HTTPException(status_code=400, detail={"message": f"{loc}: expected float, got bool"})
            try:
                coerced = float(raw_value)
            except (TypeError, ValueError) as error:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: expected float, got {raw_value!r}"}) from error
            if "min" in param_schema and coerced < param_schema["min"]:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} < min {param_schema['min']}"})
            if "max" in param_schema and coerced > param_schema["max"]:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} > max {param_schema['max']}"})
            return coerced

        if ptype == "bool":
            if isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, str) and raw_value.lower() in {"true", "false"}:
                return raw_value.lower() == "true"
            raise HTTPException(status_code=400, detail={"message": f"{loc}: expected bool, got {raw_value!r}"})

        if ptype == "string":
            if not isinstance(raw_value, str):
                raise HTTPException(
                    status_code=400, detail={"message": f"{loc}: expected string, got {type(raw_value).__name__}"}
                )
            return raw_value

        if ptype == "enum":
            choices = param_schema.get("choices", [])
            if raw_value not in choices:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: {raw_value!r} not in choices {choices}"})
            return raw_value

        raise HTTPException(status_code=500, detail={"message": f"{loc}: unknown schema type {ptype!r}"})


app = FastAPI(title="AISCO GUI Server")
assembly_selection_service = AssemblySelectionService(repo_root=REPO_ROOT)
pretraining_service = PretrainingService(storage_root=PRETRAINING_STORAGE_ROOT)
handler = GUIRequestHandler()
override_manager = ConfigOverrideManager()


@app.post("/run-local")
def run_local_endpoint(payload: RunLocalRequest) -> JSONResponse:
    response = handler.handle_run_local(payload)
    return response


@app.post("/run-local/test-cli")
def test_cli_endpoint(payload: ConfigOverrideRequest) -> JSONResponse:
    response = handler.handle_test_cli(payload)
    return response


@app.get("/run-local/logs")
def run_local_logs_endpoint() -> dict[str, Any]:
    return handler.handle_run_local_logs()


@app.get("/run-local/progress")
def run_local_progress_endpoint() -> dict[str, Any]:
    return handler.handle_run_local_progress()


@app.get("/run-local/artifacts")
def run_local_artifacts_endpoint() -> dict[str, Any]:
    return handler.handle_run_local_artifacts()


@app.get("/run-local/artifact-file")
def artifact_file_endpoint(path: str = "") -> FileResponse:
    return handler.handle_artifact_file(path)


@app.get("/run-local/wandb-visual-file")
def run_local_wandb_visual_file_endpoint(path: str = "") -> FileResponse:
    return handler.handle_wandb_visual_file(path)


@app.get("/run-local/wandb-pressure-file")
def run_local_wandb_pressure_file_endpoint(path: str = "") -> FileResponse:
    return handler.handle_wandb_pressure_file(path)


@app.get("/run-local/slow-gif")
def slow_gif_endpoint(path: str = "", factor: str = "2.5") -> FileResponse:
    return handler.handle_slow_gif(path, factor)


@app.get("/pretraining/gifs")
def pretraining_gifs_endpoint() -> dict[str, Any]:
    return handler.handle_pretraining_gifs()


@app.get("/pretraining/gif-file")
def pretraining_gif_file_endpoint(path: str = "") -> FileResponse:
    return handler.handle_pretraining_gif_file(path)


@app.get("/solar/renders")
def solar_renders_endpoint() -> dict[str, Any]:
    return handler.handle_solar_render_frames()


@app.get("/solar/render-file")
def solar_render_file_endpoint(group: str = "", name: str = "") -> FileResponse:
    return handler.handle_solar_render_file(group, name)


@app.get("/assembly-selection/params")
def assembly_selection_params_endpoint() -> dict[str, dict[str, float]]:
    return handler.handle_wp2_params()


@app.post("/assembly-selection/update")
def assembly_selection_update_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    return handler.handle_wp2_update(payload)


@app.post("/assembly-selection/add-wheel")
def assembly_selection_add_wheel_endpoint() -> dict[str, Any]:
    return handler.handle_wp2_add_wheel()


@app.post("/assembly-selection/remove-wheel")
def assembly_selection_remove_wheel_endpoint() -> dict[str, Any]:
    return handler.handle_wp2_remove_wheel()


@app.get("/assembly-selection/mesh")
def assembly_selection_mesh_endpoint() -> FileResponse:
    return handler.handle_wp2_mesh()


@app.post("/run-local/stop")
def run_local_stop_endpoint() -> JSONResponse:
    return handler.handle_stop_run_local()


@app.post("/config-overrides/init")
def config_overrides_init_endpoint() -> dict[str, Any]:
    return override_manager.init_session()


@app.get("/config-overrides/schema")
def config_overrides_schema_endpoint() -> dict[str, Any]:
    return override_manager.get_schema()


@app.get("/config-overrides/values")
def config_overrides_values_get_endpoint(folder: str = "") -> dict[str, Any]:
    if not folder:
        raise HTTPException(status_code=400, detail={"message": "Missing 'folder' query parameter"})
    return override_manager.get_values(folder)


@app.put("/config-overrides/values")
def config_overrides_values_put_endpoint(payload: OverrideValuesRequest) -> dict[str, Any]:
    return override_manager.set_values(payload)


@app.get("/config-overrides/sessions")
def config_overrides_sessions_endpoint() -> dict[str, Any]:
    return override_manager.list_sessions()


@app.post("/config-overrides/cleanup")
def config_overrides_cleanup_endpoint(payload: CleanupRequest) -> dict[str, Any]:
    return override_manager.cleanup(payload)


@app.get("/ui/page/{page_key}")
def gui_page_endpoint(page_key: str) -> FileResponse:
    page_path = GUI_PAGE_FILES.get(page_key)
    if page_path is None:
        valid_pages = ", ".join(sorted(GUI_PAGE_FILES))
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown page '{page_key}'. Choose one of: {valid_pages}",
            },
        )

    if not page_path.exists() or not page_path.is_file():
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Configured page file not found: {page_path}",
            },
        )

    return FileResponse(str(page_path), media_type="text/html", headers={"Cache-Control": "no-store"})


if TEMPLATES_DIR.exists():
    app.mount("/", StaticFiles(directory=str(TEMPLATES_DIR), html=True), name="static")
else:
    LOGGER.warning("Templates directory missing: %s", TEMPLATES_DIR)


def main() -> None:
    host = "0.0.0.0"
    port = 8000

    print("--- AISCO GUI Server ---")
    print(f"Serving static files from: {TEMPLATES_DIR}")
    print(f"Local URL: http://localhost:{port}")
    print(f"To open in Windows: /mnt/c/Program\\ Files/Mozilla\\ Firefox/firefox.exe http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
