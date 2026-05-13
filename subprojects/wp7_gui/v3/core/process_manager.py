"""Single-run subprocess manager.

Owns exactly one active WP4 run at a time: spawns run_local.sh, tracks the
process + log file, and kills the process group on stop(). Thread-safe.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import TypedDict

LOGGER = logging.getLogger(__name__)


class RunState(TypedDict):
    running: bool
    pid: int | None
    return_code: int | None
    log_path: Path | None


class ProcessManager:
    def __init__(self, logs_dir: Path) -> None:
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._log_path: Path | None = None
        self._logs_dir = logs_dir

    def start(
        self,
        run_local_script: Path,
        repo_parent_dir: Path,
        extra_args: list[str] | None = None,
        env_overrides: dict[str, str] | None = None,
    ) -> tuple[int, Path]:
        """Spawn run_local.sh and return (pid, log_path).

        Args:
            run_local_script: absolute path to run_local.sh
            repo_parent_dir: cwd for the subprocess (parent of AISCO_code)
            extra_args: appended after the script path (e.g. ["--uses-gui", "--cfg-folder", ...])
            env_overrides: env vars to set/clear on the subprocess environment

        Returns:
            (pid, log_path) of the started process
        """
        if not run_local_script.exists():
            raise FileNotFoundError(f"run_local.sh not found: {run_local_script}")

        with self._lock:
            if self._process is not None and self._process.poll() is None:
                assert self._log_path is not None
                return self._process.pid, self._log_path

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._logs_dir.mkdir(parents=True, exist_ok=True)
            log_path = self._logs_dir / f"run_local_{stamp}.log"
            log_path.write_text(
                f"[{datetime.now().isoformat(timespec='seconds')}] Starting run_local.sh\n\n",
                encoding="utf-8",
            )

            process_env = os.environ.copy()
            for key, value in (env_overrides or {}).items():
                if value is None:
                    process_env.pop(key, None)
                else:
                    process_env[key] = value

            log_handle = log_path.open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    ["bash", str(run_local_script), *(extra_args or [])],
                    cwd=str(repo_parent_dir),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=process_env,
                    start_new_session=True,
                )
            finally:
                log_handle.close()

            self._process = process
            self._log_path = log_path

        LOGGER.info("Started run_local.sh PID=%s log=%s", process.pid, log_path)
        return process.pid, log_path

    def stop(self) -> dict[str, object]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return {"status": "not_running", "message": "No active run to stop"}
            try:
                os.killpg(self._process.pid, signal.SIGTERM)
            except Exception:
                with contextlib.suppress(Exception):
                    self._process.terminate()
            return {"status": "stopping", "pid": self._process.pid, "message": "Sent termination signal"}

    def get_state(self) -> RunState:
        with self._lock:
            if self._process is None:
                return RunState(running=False, pid=None, return_code=None, log_path=self._log_path)
            running = self._process.poll() is None
            return RunState(
                running=running,
                pid=self._process.pid,
                return_code=None if running else self._process.poll(),
                log_path=self._log_path,
            )

    @property
    def log_path(self) -> Path | None:
        return self._log_path
