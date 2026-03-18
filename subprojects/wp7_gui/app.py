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
from urllib.parse import parse_qs, unquote, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = Path(__file__).resolve().parent
RUN_LOCAL_SCRIPT = REPO_ROOT / "wp4_aeroshell_optimizer" / "v1" / "slurm_scripts" / "run_local.sh"
RUN_LOCAL_LOG = GUI_ROOT / "logs" / f"run_local_{timestamp}.log"
LOCAL_STORAGE_DIR = REPO_ROOT.parent / "local_storage_dir"
SLOW_GIF_CACHE_DIR = GUI_ROOT / "cache"

PROCESS_LOCK = threading.Lock()
STATE: dict[str, subprocess.Popen[str] | None] = {"run_local_process": None}


class GUIRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/run-local/logs":
            self.handle_run_local_logs()
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
        elif self.path == "/run-command":
            self.handle_run_command()
        else:
            self._send_json(404, {"message": "Endpoint not found"})

    def handle_run_local(self) -> None:
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
                RUN_LOCAL_LOG.write_text(
                    f"[{start_stamp}] Starting run_local.sh\n\n",
                    encoding="utf-8",
                )

                log_handle = RUN_LOCAL_LOG.open("a", encoding="utf-8")
                process = subprocess.Popen(
                    ["bash", str(RUN_LOCAL_SCRIPT)],
                    cwd=str(REPO_ROOT.parent),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
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

    def handle_run_command(self) -> None:
        body_data = self._read_json_body()
        command = str(body_data.get("command", "")).strip()

        if not command:
            self._send_json(400, {"message": "Command is empty"})
            return

        try:
            completed = subprocess.run(
                ["bash", "-lc", command],
                cwd=str(REPO_ROOT.parent),
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._send_json(408, {"message": "Command timed out after 300 seconds"})
            return
        except Exception as error:
            self._send_json(500, {"message": f"Failed to execute command: {error}"})
            return

        max_chars = 60000
        stdout = completed.stdout[-max_chars:]
        stderr = completed.stderr[-max_chars:]

        self._send_json(
            200,
            {
                "message": "Command executed",
                "command": command,
                "return_code": completed.returncode,
                "stdout": stdout,
                "stderr": stderr,
            },
        )

    def handle_run_local_artifacts(self) -> None:
        run_dir = self._detect_current_run_dir()
        if run_dir is None:
            self._send_json(
                200,
                {
                    "run_dir": None,
                    "mesh_assembly_gifs": [],
                    "meshes": [],
                    "message": "No completed run output directory found yet.",
                },
            )
            return

        mesh_assembly_dir = run_dir / "mesh_assembly_mode"
        mesh_dir = run_dir / "meshes"
        mesh_steps_dir = run_dir / "solar_loss_rendering" / "sim_renders"

        gifs = []
        if mesh_assembly_dir.exists():
            for gif_path in sorted(mesh_assembly_dir.glob("*.gif"), key=lambda p: p.stat().st_mtime, reverse=True):
                rel_path = gif_path.resolve().relative_to(LOCAL_STORAGE_DIR.resolve()).as_posix()
                gifs.append(
                    {
                        "name": gif_path.name,
                        "rel_path": rel_path,
                        "url": f"/run-local/artifact-file?path={rel_path}",
                        "slow_url": f"/run-local/slow-gif?path={rel_path}&factor=2.5",
                    }
                )

        meshes = []
        if mesh_dir.exists():
            for mesh_path in sorted(mesh_dir.glob("*.ply")):
                rel_path = mesh_path.resolve().relative_to(LOCAL_STORAGE_DIR.resolve()).as_posix()
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

        mesh_step_images = []
        if mesh_steps_dir.exists():
            for image_path in sorted(mesh_steps_dir.glob("mesh_selection_side_*.png")):
                rel_path = image_path.resolve().relative_to(LOCAL_STORAGE_DIR.resolve()).as_posix()
                epoch = self._extract_epoch(image_path.name)
                mesh_step_images.append(
                    {
                        "name": image_path.name,
                        "rel_path": rel_path,
                        "url": f"/run-local/artifact-file?path={rel_path}",
                        "epoch": epoch,
                    }
                )

            mesh_step_images.sort(key=lambda item: item["epoch"] if item["epoch"] is not None else -1)

        self._send_json(
            200,
            {
                "run_dir": run_dir.resolve().relative_to(REPO_ROOT.parent.resolve()).as_posix(),
                "mesh_assembly_gifs": gifs,
                "meshes": meshes,
                "mesh_step_images": mesh_step_images,
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

        try:
            local_root = LOCAL_STORAGE_DIR.resolve()
            requested = (local_root / rel_path).resolve()
            requested.relative_to(local_root)
        except Exception:
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

    def handle_slow_gif(self, parsed) -> None:
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

        requested = self._resolve_local_storage_file(rel_path)
        if requested is None:
            self._send_json(400, {"message": "Invalid artifact path"})
            return
        if not requested.exists() or not requested.is_file():
            self._send_json(404, {"message": "Artifact file not found"})
            return
        if requested.suffix.lower() != ".gif":
            self._send_json(400, {"message": "Only GIF files are supported for slow playback"})
            return

        cache_key = hashlib.sha1(f"{requested.resolve()}|{requested.stat().st_mtime_ns}|{factor:.2f}".encode("utf-8")).hexdigest()
        SLOW_GIF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = SLOW_GIF_CACHE_DIR / f"slow_{cache_key}.gif"

        if not cached_path.exists():
            try:
                import imageio.v2 as imageio

                frames = imageio.mimread(requested)
                if not frames:
                    self._send_json(500, {"message": "Failed to decode GIF frames"})
                    return

                meta = imageio.get_reader(requested).get_meta_data()
                base_duration = float(meta.get("duration", 0.08))
                slow_duration = max(0.02, base_duration * factor)
                imageio.mimsave(cached_path, frames, format="GIF", duration=slow_duration)
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

    def _resolve_local_storage_file(self, rel_path: str) -> Path | None:
        try:
            local_root = LOCAL_STORAGE_DIR.resolve()
            requested = (local_root / rel_path).resolve()
            requested.relative_to(local_root)
            return requested
        except Exception:
            return None

    def _detect_current_run_dir(self) -> Path | None:
        if RUN_LOCAL_LOG.exists():
            logs = RUN_LOCAL_LOG.read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"Output directory:\s*(.+)", logs)
            if matches:
                run_dir = Path(matches[-1].strip())
                if not run_dir.is_absolute():
                    run_dir = (REPO_ROOT.parent / run_dir).resolve()
                else:
                    run_dir = run_dir.resolve()

                if run_dir.exists() and run_dir.is_dir():
                    return run_dir

        if not LOCAL_STORAGE_DIR.exists():
            return None

        run_dirs = []
        for model_dir in LOCAL_STORAGE_DIR.iterdir():
            if not model_dir.is_dir():
                continue
            for child in model_dir.iterdir():
                if child.is_dir() and (child / "mesh_assembly_mode").exists():
                    run_dirs.append(child)

        if not run_dirs:
            return None

        return max(run_dirs, key=lambda p: p.stat().st_mtime)

    @staticmethod
    def _extract_epoch(filename: str) -> int | None:
        match = re.search(r"epoch(\d+)", filename)
        if match is None:
            match = re.search(r"_(\d+)\.(?:png|gif|ply)$", filename)
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
