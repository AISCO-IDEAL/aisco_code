"""Central paths and constants for v3.

All hardcoded paths from v2's app_v2.py module-level live here so the rest of
v3 only takes a Settings instance. Storage-root resolution that depends on
wp4's general_config.yaml lives in shared.run_paths (it must re-read on every
call to match v2 behavior).
"""

from __future__ import annotations

import os
import re
import socket
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import Environment as Wp4Environment
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import detect_environment as detect_wp4_environment


class RuntimeEnvironment(Enum):
    """Supported execution environments for local launch script selection."""

    AISCO_BEAST = "aisco_beast"
    WSL = "wsl"
    EULER = "euler"
    SLURM = "slurm"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    gui_root: Path
    wp4_package_root: Path
    wp4_active_version: str
    wp4_configs_dir: Path
    wp4_slurm_dir: Path
    run_local_script: Path
    general_config_path: Path

    default_local_storage_dir: Path
    default_bulk_storage_dir: Path
    pretraining_storage_root: Path

    templates_dir: Path
    static_dir: Path
    schema_path: Path
    panel_visibility_defaults_path: Path
    mesh_priority_path: Path
    logs_dir: Path
    cache_dir: Path
    config_overrides_root: Path

    host: str
    port: int
    runtime_environment: str = RuntimeEnvironment.UNKNOWN.value

    assembly_id_env_var: str = "AISCO_ASSEMBLY_ID"


def _detect_wp4_active_version(wp4_package_root: Path) -> str:
    init_path = wp4_package_root / "__init__.py"
    try:
        text = init_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "v2"
    match = re.search(r"ACTIVE_VERSION\s*=\s*[\"']([^\"']+)[\"']", text)
    return match.group(1) if match else "v2"


def _detect_runtime_environment() -> RuntimeEnvironment:
    """Detect whether we run on Beast/WSL or cluster-like hosts."""
    # Keep GUI detection aligned with WP4 runtime logic when possible.
    with_wp4 = {
        Wp4Environment.AISCO_BEAST: RuntimeEnvironment.AISCO_BEAST,
        Wp4Environment.WSL: RuntimeEnvironment.WSL,
        Wp4Environment.EULER: RuntimeEnvironment.EULER,
        Wp4Environment.SLURM: RuntimeEnvironment.SLURM,
    }
    detected = with_wp4.get(detect_wp4_environment())
    if detected is not None:
        return detected

    # Defensive fallback in case WP4 detection returns UNKNOWN.
    # WSL check first — hostname can mirror the Windows host name.
    if any(var in os.environ for var in ("WSL_DISTRO_NAME", "WSL_INTEROP", "WSLENV")):
        return RuntimeEnvironment.WSL
    for probe in ("/proc/sys/kernel/osrelease", "/proc/version"):
        try:
            text = Path(probe).read_text(encoding="utf-8", errors="ignore").lower()
            if "microsoft" in text or "wsl" in text:
                return RuntimeEnvironment.WSL
        except OSError:
            continue

    hostname = socket.gethostname().strip().lower()
    if hostname in {"aisco-beast", "aisco_beast"} or hostname.startswith("aisco-beast"):
        return RuntimeEnvironment.AISCO_BEAST
    if hostname.startswith("eu-"):
        return RuntimeEnvironment.EULER
    if "SLURM_JOB_ID" in os.environ:
        return RuntimeEnvironment.SLURM
    return RuntimeEnvironment.UNKNOWN


def build_settings() -> Settings:
    gui_root = Path(__file__).resolve().parents[1]
    repo_root = gui_root.parents[1]
    wp4_package_root = repo_root / "wp4_aeroshell_optimizer"
    wp4_active_version = _detect_wp4_active_version(wp4_package_root)
    wp4_active_root = wp4_package_root / wp4_active_version
    wp4_configs_dir = wp4_active_root / "configs"
    runtime_environment = _detect_runtime_environment()

    if runtime_environment == RuntimeEnvironment.WSL:
        run_local_script = wp4_active_root / "slurm_scripts" / "run_local_wsl.sh"
    elif runtime_environment == RuntimeEnvironment.AISCO_BEAST:
        run_local_script = wp4_active_root / "slurm_scripts" / "run_local_beast.sh"
    else:
        run_local_script = wp4_active_root / "slurm_scripts" / "run_local_cluster.sh"

    return Settings(
        repo_root=repo_root,
        gui_root=gui_root,
        wp4_package_root=wp4_package_root,
        wp4_active_version=wp4_active_version,
        wp4_configs_dir=wp4_configs_dir,
        wp4_slurm_dir=wp4_active_root / "slurm_scripts",
        run_local_script=run_local_script,
        general_config_path=wp4_configs_dir / "general_config.yaml",
        default_local_storage_dir=repo_root.parent / "local_storage_dir",
        default_bulk_storage_dir=Path("/cluster/work/fuge/AISCO/bulk_storage"),
        pretraining_storage_root=Path("/cluster/work/fuge/AISCO/latent_pretraining/runs/14f323b3fc/pretraining_latent_gifs"),
        templates_dir=gui_root / "frontend" / "templates",
        static_dir=gui_root / "frontend" / "static",
        schema_path=gui_root / "schema" / "gui_schema.json",
        panel_visibility_defaults_path=gui_root / "schema" / "panel_visibility_defaults.json",
        mesh_priority_path=gui_root / "schema" / "mesh_priority.json",
        logs_dir=gui_root / "logs",
        cache_dir=gui_root / "cache",
        config_overrides_root=gui_root / "config_overrides",
        host="0.0.0.0",
        port=8001,
        runtime_environment=runtime_environment.value,
    )


SETTINGS = build_settings()
