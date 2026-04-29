import hashlib
import logging
import mimetypes
import os
import re
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from functions.assembly_selection_service import AssemblySelectionService
from functions.pretraining_service import PretrainingService
from pydantic import BaseModel

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
PRETRAINING_STORAGE_ROOT = Path("/cluster/work/fuge/AISCO/latent_pretraining/runs/14f323b3fc/pretraining_latent_gifs")
SLOW_GIF_CACHE_DIR = GUI_ROOT / "cache"
MESH_ASSEMBLY_DIR_CANDIDATES = ("mesh_assembly_mode_v4", "mesh_assembly_mode_v3", "mesh_assembly_mode_v2", "mesh_assembly_mode")


PROCESS_LOCK = threading.Lock()
STATE: dict[str, subprocess.Popen[str] | None] = {"run_local_process": None}
ASSEMBLY_ID_ENV_VAR = "AISCO_ASSEMBLY_ID"
GUI_PAGE_FILES: dict[str, Path] = {
    "introduction": PAGES_DIR / "introduction.html",
    "pretraining": PAGES_DIR / "pretraining.html",
    "optimization": PAGES_DIR / "optimization.html",
    "solar": PAGES_DIR / "solar.html",
    "assembly_selection": PAGES_DIR / "assembly_selection.html",
    "logs": PAGES_DIR / "assembly_selection.html",
}


class RunLocalRequest(BaseModel):
    """Request payload for launching a local optimization run.

    Args:
        assembly_id: Optional assembly ID to optimize. Empty string is treated as random.

    Returns:
        None
    """

    assembly_id: int | str | None = None


class GUIRequestHandler:
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
        run_dir = self._detect_current_run_dir()
        if run_dir is None:
            return {
                "run_dir": None,
                "mesh_assembly_gifs": [],
                "solar_loss_gifs": [],
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
            "meshes": meshes,
            "message": "Artifacts loaded",
        }

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


app = FastAPI(title="AISCO GUI Server")
assembly_selection_service = AssemblySelectionService(repo_root=REPO_ROOT)
pretraining_service = PretrainingService(storage_root=PRETRAINING_STORAGE_ROOT)
handler = GUIRequestHandler()


@app.post("/run-local")
def run_local_endpoint(payload: RunLocalRequest) -> JSONResponse:
    response = handler.handle_run_local(payload)
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


@app.get("/run-local/slow-gif")
def slow_gif_endpoint(path: str = "", factor: str = "2.5") -> FileResponse:
    return handler.handle_slow_gif(path, factor)


@app.get("/pretraining/gifs")
def pretraining_gifs_endpoint() -> dict[str, Any]:
    return handler.handle_pretraining_gifs()


@app.get("/pretraining/gif-file")
def pretraining_gif_file_endpoint(path: str = "") -> FileResponse:
    return handler.handle_pretraining_gif_file(path)


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
    port = 8001

    print("--- AISCO GUI Server ---")
    print(f"Serving static files from: {TEMPLATES_DIR}")
    print(f"Local URL: http://localhost:{port}")
    print(f"To open in Windows: /mnt/c/Program\\ Files/Mozilla\\ Firefox/firefox.exe http://localhost:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
