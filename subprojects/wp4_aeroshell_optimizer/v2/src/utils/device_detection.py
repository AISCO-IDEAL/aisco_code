"""
Environment and device detection utilities.

Detects whether the code is running on:
- Local AISCO Beast machine
- WSL (Windows Subsystem for Linux)
- Euler cluster (ETH HPC)
- SLURM job (generic cluster)
"""

import os
import socket
from enum import Enum
from pathlib import Path


class Environment(Enum):
    """Enumeration of supported runtime environments."""

    AISCO_BEAST = "aisco_beast"  # Local AISCO Beast machine
    WSL = "wsl"  # Windows Subsystem for Linux
    EULER = "euler"  # ETH Euler cluster
    SLURM = "slurm"  # Generic SLURM cluster
    UNKNOWN = "unknown"  # Could not determine


def _is_wsl() -> bool:
    """Best-effort WSL detection covering env vars and kernel release string."""
    if any(var in os.environ for var in ("WSL_DISTRO_NAME", "WSL_INTEROP", "WSLENV")):
        return True
    for probe in ("/proc/sys/kernel/osrelease", "/proc/version"):
        try:
            text = Path(probe).read_text(encoding="utf-8", errors="ignore").lower()
            if "microsoft" in text or "wsl" in text:
                return True
        except OSError:
            continue
    return False


def detect_environment() -> Environment:
    """
    Detect the runtime environment.

    Returns:
        Environment: The detected environment type.
    """
    # WSL check takes priority (hostname can mirror the Windows host name).
    if _is_wsl():
        return Environment.WSL

    # Check hostname
    hostname = socket.gethostname().strip().lower()
    if hostname in {"aisco-beast", "aisco_beast"} or hostname.startswith("aisco-beast"):
        return Environment.AISCO_BEAST
    if hostname.startswith("eu-") or hostname.startswith("euler"):
        return Environment.EULER

    # Check for SLURM environment
    if "SLURM_JOB_ID" in os.environ:
        return Environment.SLURM

    return Environment.UNKNOWN


def is_local_machine() -> bool:
    """
    Check if running on a local machine (not a cluster).

    Returns:
        bool: True if running locally (AISCO Beast or WSL), False otherwise.
    """
    env = detect_environment()
    return env in (Environment.AISCO_BEAST, Environment.WSL)


def is_cluster() -> bool:
    """
    Check if running on a cluster.

    Returns:
        bool: True if running on a cluster (Euler or SLURM), False otherwise.
    """
    env = detect_environment()
    return env in (Environment.EULER, Environment.SLURM)


def get_config_base_path() -> Path:
    """
    Get the base path where AISCO_code is located, accounting for local vs cluster setup.

    Beast: ~/Documents/AISCO_code
    WSL:   ~/AISCO_code  (no Documents sub-folder)
    Cluster: ~/AISCO_code

    Returns:
        Path: The base directory that contains the AISCO_code folder.
    """
    env = detect_environment()
    home = Path.home()

    if env == Environment.AISCO_BEAST:
        return home / "Documents"
    else:
        return home


def get_config_path(v: str, cfg_name: str) -> Path:
    """
    Get the full path to a configuration file.

    Args:
        v: Version string (e.g., "v0", "v1", "v2")
        cfg_name: Configuration filename (e.g., "general_config.yaml")

    Returns:
        Path: The full path to the configuration file.
    """
    base_path = get_config_base_path()
    return base_path / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / v / "configs" / cfg_name


def get_run_locally_value() -> str:
    """
    Get the value to set for 'run_locally' flag based on detected environment.

    Returns:
        str: "True" if running locally, "False" if on cluster.
    """
    return "True" if is_local_machine() else "False"
