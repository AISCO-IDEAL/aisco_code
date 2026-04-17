#!/usr/bin/env python
"""Run GAOT2 and GAOT3 through their real entrypoints and configs.

This deliberately does not import model internals or synthesize data. It answers
the practical question: do both versions execute from the same public surface,
``python main.py -c config/main_config.yaml``?
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[5]
GAOT_ROOT = REPO_ROOT / "subprojects" / "wp1_cfd_ai" / "AISCO_GAOT"
DEFAULT_GAOT2_CONFIG = Path("subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/config/main_config.yaml")
DEFAULT_GAOT3_CONFIG = Path("subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/config/main_config.yaml")
DEFAULT_LOG_DIR = Path(tempfile.gettempdir()) / "gaot2_gaot3_main_equivalence"
EQUIVALENCE_EPOCHS = 3
EQUIVALENCE_SUBSAMPLING_SEED = 42


@dataclass(frozen=True)
class MainRun:
    version: str
    cwd: Path
    command: list[str]
    config_path: Path
    stdout_path: Path
    stderr_path: Path
    returncode: int
    timed_out: bool


def _tail(path: Path, max_chars: int = 5000) -> str:
    if not path.exists():
        return "<missing>"
    content = path.read_text(encoding="utf-8", errors="replace")
    if len(content) <= max_chars:
        return content
    return f"... truncated ...\n{content[-max_chars:]}"


def _env_for(version_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(REPO_ROOT), str(REPO_ROOT / "subprojects"), str(version_dir)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("WANDB_DISABLED", "true")
    return env


def _repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _repo_relative(path: Path) -> Path:
    if path.is_absolute():
        try:
            return path.relative_to(REPO_ROOT)
        except ValueError:
            return path
    return path


def _write_equivalence_config(version: str, source_config: Path, log_dir: Path) -> Path:
    """Write a temporary config with deterministic subsampling and short training."""
    source_config = _repo_path(source_config).resolve()
    with source_config.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected config mapping in {source_config}, got {type(cfg).__name__}")

    dataset = cfg.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset mapping in {source_config}")
    dataset["deterministic_subsampling"] = True
    dataset["subsampling_seed"] = EQUIVALENCE_SUBSAMPLING_SEED

    optimizer = cfg.setdefault("optimizer", {})
    if not isinstance(optimizer, dict):
        raise TypeError(f"Expected optimizer mapping in {source_config}")
    optimizer_args = optimizer.setdefault("args", {})
    if not isinstance(optimizer_args, dict):
        raise TypeError(f"Expected optimizer.args mapping in {source_config}")
    optimizer_args["epoch"] = EQUIVALENCE_EPOCHS

    effective_dir = log_dir / "effective_configs"
    effective_dir.mkdir(parents=True, exist_ok=True)
    effective_config = effective_dir / f"{version.lower()}_equivalence_config.yaml"
    with effective_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return effective_config


def _run_main(version: str, config_path: Path, log_dir: Path, timeout: int | None) -> MainRun:
    version_dir = GAOT_ROOT / version
    main_path = version_dir / "main.py"
    config_arg = _repo_relative(config_path)
    config_path = _repo_path(config_arg).resolve()
    if not main_path.exists():
        raise FileNotFoundError(f"{version} main.py not found: {main_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"{version} config not found: {config_path}")

    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{version.lower()}_stdout.log"
    stderr_path = log_dir / f"{version.lower()}_stderr.log"
    command = [sys.executable, str(main_path.relative_to(REPO_ROOT)), "-c", str(config_arg)]

    print(f"\n[{version}]")
    print(f"cwd: {REPO_ROOT}")
    print(f"config: {config_arg}")
    print(f"command: {' '.join(command)}")
    print(f"stdout: {stdout_path}")
    print(f"stderr: {stderr_path}")

    timed_out = False
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=_env_for(version_dir),
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = 124

    print(f"{version} return code: {returncode}")
    return MainRun(
        version=version,
        cwd=REPO_ROOT,
        command=command,
        config_path=config_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        returncode=returncode,
        timed_out=timed_out,
    )


def _print_failure_report(runs: list[MainRun]) -> None:
    print("\nGAOT2/GAOT3 main execution failed.")
    for run in runs:
        if run.returncode == 0:
            continue
        timeout_note = " timed out" if run.timed_out else ""
        print(f"\n[{run.version}] exit code {run.returncode}{timeout_note}")
        print(f"command: {' '.join(run.command)}")
        print(f"stdout log: {run.stdout_path}")
        print(f"stderr log: {run.stderr_path}")
        print("\n--- stdout tail ---")
        print(_tail(run.stdout_path))
        print("\n--- stderr tail ---")
        print(_tail(run.stderr_path))


def run_gaot_main_equivalence(
    gaot2_config: Path | None = None,
    gaot3_config: Path | None = None,
    log_dir: Path = DEFAULT_LOG_DIR,
    timeout: int | None = None,
) -> None:
    """Execute both real GAOT main/config pairs and require both to succeed."""
    source_config_paths = {
        "GAOT2": gaot2_config or DEFAULT_GAOT2_CONFIG,
        "GAOT3": gaot3_config or DEFAULT_GAOT3_CONFIG,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    config_paths = {
        version: _write_equivalence_config(version, source_config, log_dir)
        for version, source_config in source_config_paths.items()
    }

    print(
        "\nEquivalence overrides: "
        f"dataset.deterministic_subsampling=True, "
        f"dataset.subsampling_seed={EQUIVALENCE_SUBSAMPLING_SEED}, "
        f"optimizer.args.epoch={EQUIVALENCE_EPOCHS}"
    )

    runs = [_run_main(version, config_paths[version], log_dir, timeout) for version in ("GAOT2", "GAOT3")]
    failed = [run for run in runs if run.returncode != 0]
    if failed:
        _print_failure_report(runs)
        raise AssertionError("At least one GAOT main/config pair failed to execute.")

    print("\nBoth GAOT main/config pairs executed successfully.")
    for run in runs:
        print(f"{run.version}: stdout={run.stdout_path} stderr={run.stderr_path}")


def test_gaot2_gaot3_main_configs_execute() -> None:
    """Pytest guard for the expensive real-config run.

    Set RUN_GAOT_MAIN_EQUIVALENCE=1 to enable this from pytest. Calling this
    file directly always runs the check.
    """
    if os.environ.get("RUN_GAOT_MAIN_EQUIVALENCE") != "1":
        try:
            import pytest
        except ImportError as exc:  # pragma: no cover - pytest is expected here
            raise RuntimeError("pytest is required to skip this integration test") from exc
        pytest.skip("Set RUN_GAOT_MAIN_EQUIVALENCE=1 to execute real GAOT main configs.")

    timeout = os.environ.get("GAOT_MAIN_TIMEOUT_SECONDS")
    run_gaot_main_equivalence(timeout=int(timeout) if timeout else None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute GAOT2 and GAOT3 main.py with their real configs.")
    parser.add_argument("--gaot2-config", type=Path, default=None, help=f"GAOT2 config path. Defaults to {DEFAULT_GAOT2_CONFIG.as_posix()}.")
    parser.add_argument("--gaot3-config", type=Path, default=None, help=f"GAOT3 config path. Defaults to {DEFAULT_GAOT3_CONFIG.as_posix()}.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for stdout/stderr logs.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout per main.py run, in seconds.")
    args = parser.parse_args()

    run_gaot_main_equivalence(
        gaot2_config=args.gaot2_config,
        gaot3_config=args.gaot3_config,
        log_dir=args.log_dir,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
