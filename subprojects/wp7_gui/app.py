import errno
import json
import logging
import os
import subprocess
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

REPO_ROOT = Path(__file__).resolve().parents[1]
GUI_ROOT = Path(__file__).resolve().parent
RUN_LOCAL_SCRIPT = REPO_ROOT / "wp4_aeroshell_optimizer" / "v1" / "slurm_scripts" / "run_local.sh"
RUN_LOCAL_LOG = GUI_ROOT / "logs" / f"run_local_{timestamp}.log"

PROCESS_LOCK = threading.Lock()
STATE: dict[str, subprocess.Popen[str] | None] = {"run_local_process": None}

class GUIRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/run-local/logs":
            self.handle_run_local_logs()
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
        self.send_header("Access-Control-Allow-Origin", "*") # Useful if GUI is served differently
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
