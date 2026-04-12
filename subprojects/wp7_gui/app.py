import errno
import hashlib
import json
import logging
import mimetypes
import os
import re
import subprocess
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, unquote, urlparse

import imageio.v2 as imageio
import yaml

from subprojects.wp4_aeroshell_optimizer import CONFIGS_DIR as _WP4_CONFIGS_DIR
from subprojects.wp4_aeroshell_optimizer import SLURM_SCRIPTS_DIR as _WP4_SLURM_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = Path(__file__).resolve().parent

RUN_LOCAL_SCRIPT = _WP4_SLURM_DIR / "run_local.sh"
RUN_LOCAL_LOG = GUI_ROOT / "logs" / f"run_local_{timestamp}.log"
GENERAL_CONFIG_PATH = _WP4_CONFIGS_DIR / "general_config.yaml"
DEFAULT_LOCAL_STORAGE_DIR = REPO_ROOT.parent / "local_storage_dir"
DEFAULT_BULK_STORAGE_DIR = Path("/cluster/work/fuge/AISCO/bulk_storage")
SLOW_GIF_CACHE_DIR = GUI_ROOT / "cache"

PROCESS_LOCK = threading.Lock()
STATE: dict[str, subprocess.Popen[str] | None] = {"run_local_process": None}
ASSEMBLY_ID_ENV_VAR = "AISCO_ASSEMBLY_ID"


class GUIRequestHandler(SimpleHTTPRequestHandler):
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

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/run-local/logs":
            self.handle_run_local_logs()
            return
        if parsed.path == "/run-local/progress":
            self.handle_run_local_progress()
            return
        if parsed.path == "/run-local/artifacts":
            self.handle_run_local_artifacts()
            return
        if parsed.path == "/run-local/artifact-file":
            self.handle_artifact_file(parsed)
            return
        if parsed.path == "/run-local/slow-gif":
            self.handle_slow_gif(parsed)
            return

        super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/run-local":
            self.handle_run_local()
        else:
            self._send_json(404, {"message": "Endpoint not found"})

    def handle_run_local(self) -> None:
        body_data = self._read_json_body()
        raw_assembly_id = body_data.get("assembly_id")

        assembly_id: int | None = None
        if raw_assembly_id not in (None, ""):
            raw_assembly_id_str = str(raw_assembly_id).strip()
            try:
                assembly_id = int(raw_assembly_id_str)
            except (TypeError, ValueError):
                self._send_json(400, {"message": "assembly_id must be an integer between 1 and 51"})
                return

            if not 1 <= assembly_id <= 51:
                self._send_json(400, {"message": "assembly_id must be in range 1..51"})
                return

        if not RUN_LOCAL_SCRIPT.exists():
            LOGGER.error("Script missing: %s", RUN_LOCAL_SCRIPT)
            self._send_json(500, {"message": f"Optimization script not found at {RUN_LOCAL_SCRIPT}"})
            return

        try:
            with PROCESS_LOCK:
                run_local_process = STATE["run_local_process"]
                if run_local_process is not None and run_local_process.poll() is None:
                    self._send_json(
                        200,
                        {
                            "status": "running",
                            "message": "run_local.sh is already running",
                            "pid": run_local_process.pid,
                        },
                    )
                    return

                start_stamp = datetime.now().isoformat(timespec="seconds")
                run_label = "random assembly_id" if assembly_id is None else f"assembly_id={assembly_id}"
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
            self._send_json(
                200,
                {
                    "status": "success",
                    "message": "Optimization process started",
                    "pid": process.pid,
                    "assembly_id": assembly_id,
                },
            )
        except Exception as e:
            LOGGER.exception("Failed to launch subprocess")
            self._send_json(500, {"message": str(e)})

    def handle_run_local_logs(self) -> None:
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

        self._send_json(
            200,
            {
                "logs": logs,
                "running": running,
                "pid": pid,
                "return_code": return_code,
                "truncated": truncated,
            },
        )

    def handle_run_local_progress(self) -> None:
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

        self._send_json(
            200,
            {
                "running": running,
                "completed": completed,
                "return_code": return_code,
                "num_epochs": num_epochs,
                "current_epoch": current_epoch,
                "progress_percent": round(progress_percent, 1),
            },
        )

    def handle_run_local_artifacts(self) -> None:
        storage_root = self._get_storage_root()
        run_dir = self._detect_current_run_dir()
        if run_dir is None:
            self._send_json(
                200,
                {
                    "run_dir": None,
                    "mesh_assembly_gifs": [],
                    "solar_loss_gifs": [],
                    "meshes": [],
                    "storage_root": str(storage_root),
                    "message": "No completed run output directory found yet.",
                },
            )
            return

        mesh_assembly_dir = run_dir / "mesh_assembly_mode"
        solar_loss_dir = run_dir / "solar_loss_rendering"
        mesh_dir = run_dir / "meshes"

        gifs = []
        if mesh_assembly_dir.exists():
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

        self._send_json(
            200,
            {
                "run_dir": str(run_dir.resolve()),
                "storage_root": str(storage_root),
                "mesh_assembly_gifs": gifs,
                "solar_loss_gifs": solar_loss_gifs,
                "meshes": meshes,
                "message": "Artifacts loaded",
            },
        )

    def handle_artifact_file(self, parsed) -> None:
        query = parse_qs(parsed.query)
        raw_path = query.get("path", [""])[0]
        rel_path = unquote(raw_path).lstrip("/")
        if not rel_path:
            self._send_json(400, {"message": "Missing path query parameter"})
            return

        requested = self._resolve_storage_file(rel_path)
        if requested is None:
            self._send_json(400, {"message": "Invalid artifact path"})
            return

        if not requested.exists() or not requested.is_file():
            self._send_json(404, {"message": "Artifact file not found"})
            return

        mime_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        content = requested.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def handle_slow_gif(self, parsed) -> None:  # noqa: PLR0911
        query = parse_qs(parsed.query)
        raw_path = query.get("path", [""])[0]
        rel_path = unquote(raw_path).lstrip("/")
        if not rel_path:
            self._send_json(400, {"message": "Missing path query parameter"})
            return

        try:
            factor = float(query.get("factor", ["2.5"])[0])
            factor = max(1.0, min(factor, 10.0))
        except ValueError:
            self._send_json(400, {"message": "Invalid factor value"})
            return

        requested = self._resolve_storage_file(rel_path)
        if requested is None:
            self._send_json(400, {"message": "Invalid artifact path"})
            return
        if not requested.exists() or not requested.is_file():
            self._send_json(404, {"message": "Artifact file not found"})
            return
        if requested.suffix.lower() != ".gif":
            self._send_json(400, {"message": "Only GIF files are supported for slow playback"})
            return

        cache_key = hashlib.sha1(f"{requested.resolve()}|{requested.stat().st_mtime_ns}|{factor:.2f}".encode()).hexdigest()
        SLOW_GIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = SLOW_GIF_CACHE_DIR / f"slow_{cache_key}.gif"

        if not cached_path.exists():
            if imageio is None:
                self._send_json(500, {"message": "imageio is required for slow GIF generation"})
                return
            try:
                frames = imageio.mimread(requested)
                if not frames:
                    self._send_json(500, {"message": "Failed to decode GIF frames"})
                    return

                meta = imageio.get_reader(requested).get_meta_data()
                base_duration = float(meta.get("duration", 0.08))
                slow_duration = max(0.02, base_duration * factor)
                imageio.mimsave(cached_path, cast(list[Any], frames), duration=slow_duration)
            except Exception as error:
                self._send_json(500, {"message": f"Failed to build slowed GIF: {error}"})
                return

        content = cached_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/gif")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

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
    def _has_any_gif_artifacts(run_path: Path) -> bool:
        mesh_assembly_gifs = list((run_path / "mesh_assembly_mode").glob("*.gif"))
        solar_loss_gifs = list((run_path / "solar_loss_rendering").glob("*.gif"))
        return bool(mesh_assembly_gifs or solar_loss_gifs)

    def _has_display_artifacts(self, run_path: Path) -> bool:
        meshes = list((run_path / "meshes").glob("*.ply")) + list((run_path / "meshes").glob("*.poly"))
        return self._has_any_gif_artifacts(run_path) or bool(meshes)

    def _collect_candidate_logs(self) -> list[Path]:
        logs_dir = GUI_ROOT / "logs"
        candidate_logs: list[Path] = []
        if RUN_LOCAL_LOG.exists():
            candidate_logs.append(RUN_LOCAL_LOG)
        if logs_dir.exists():
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
                has_mesh_assembly = (child / "mesh_assembly_mode").exists()
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

    def _read_json_body(self) -> dict:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except ValueError:
            return {}

        if content_length <= 0:
            return {}

        raw_body = self.rfile.read(content_length)
        try:
            decoded = raw_body.decode("utf-8")
            return json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _send_json(self, status_code: int, payload: dict) -> None:
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # Useful if GUI is served differently
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def create_server(host: str, start_port: int, max_tries: int = 20) -> tuple[ThreadingHTTPServer, int]:
    for port in range(start_port, start_port + max_tries):
        try:
            server = ThreadingHTTPServer((host, port), GUIRequestHandler)
        except OSError as error:
            if error.errno == errno.EADDRINUSE:
                LOGGER.warning("Port %s is already in use, trying next port...", port)
                continue
            raise
        else:
            return server, port

    msg = f"No free port found in range {start_port}-{start_port + max_tries - 1}"
    raise RuntimeError(msg)


def main() -> None:
    templates_dir = Path(__file__).resolve().parent / "templates"

    if not templates_dir.exists():
        print(f"Warning: {templates_dir} not found. Static files may not serve correctly.")
    else:
        os.chdir(templates_dir)

    host = "0.0.0.0"
    start_port = 8000
    server, port = create_server(host, start_port)

    print("--- AISCO GUI Server ---")
    print(f"Serving static files from: {templates_dir}")
    print(f"Local URL: http://localhost:{port}")
    print(f"To open in Windows: /mnt/c/Program\\ Files/Mozilla\\ Firefox/firefox.exe http://localhost:{port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
