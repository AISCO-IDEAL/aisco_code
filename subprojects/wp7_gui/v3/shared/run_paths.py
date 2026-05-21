"""Storage-root resolution and run-directory discovery.

Ports v2's GUIRequestHandler storage logic. Re-reads general_config.yaml on
every call to pick up run_locally flag changes without a server restart.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import Environment as RuntimeEnv
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import detect_environment
from subprojects.wp7_gui.v3.core.settings import SETTINGS

LOGGER = logging.getLogger(__name__)

MESH_ASSEMBLY_CANDIDATES = (
    "mesh_assembly_v4",
    "mesh_assembly_mode_v4",
    "mesh_assembly_mode_v3",
    "mesh_assembly_mode_v2",
    "mesh_assembly_mode",
)


def _resolve_storage_path(raw_path: str | None, repo_parent: Path) -> Path | None:
    if raw_path is None:
        return None
    p = Path(raw_path)
    return (repo_parent / p).resolve() if not p.is_absolute() else p.resolve()


def _is_local_runtime() -> bool:
    env = detect_environment()
    return env in {RuntimeEnv.AISCO_BEAST, RuntimeEnv.WSL}


def get_storage_root(  # noqa: PLR0911
    general_config_path: Path,
    default_local: Path,
    default_bulk: Path,
    repo_parent: Path,
) -> Path:
    """Resolve the artifact storage root from wp4's general_config.yaml.

    Re-reads the file on every call (matches v2 behavior so toggling
    run_locally takes effect immediately).

    Selection policy:
    - run_locally=True  -> beast_storage_dir, then local_storage_dir
    - run_locally=False -> storage_dir
    """
    try:
        with general_config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        run_locally_raw = cfg.get("general_info", {}).get("run_locally", False)
        # Accept both real booleans and legacy quoted strings from YAML files.
        if isinstance(run_locally_raw, bool):
            run_locally = run_locally_raw
        else:
            run_locally = str(run_locally_raw).strip().lower() in {"1", "true", "yes", "on"}
        storage = cfg.get("storage_paths", {})
        local_runtime = _is_local_runtime()

        key = "beast_storage_dir" if run_locally else "storage_dir"
        raw = storage.get(key)
        if raw is None and run_locally:
            raw = storage.get("local_storage_dir")
        if raw is None:
            # If config is incomplete but host is local, prefer local storage for GUI discovery.
            if local_runtime:
                local_resolved = _resolve_storage_path(storage.get("local_storage_dir"), repo_parent)
                if local_resolved is not None:
                    return local_resolved
                return default_local.resolve()
            return (default_local if run_locally else default_bulk).resolve()

        resolved = _resolve_storage_path(raw, repo_parent)
        assert resolved is not None, "Resolved storage path unexpectedly None."

        # On local Beast workflows, some configs keep run_locally=False while
        # storing outputs under beast_storage_dir. If the selected root is
        # unavailable, transparently fall back to beast_storage_dir.
        if resolved.exists():
            return resolved

        beast_raw = storage.get("beast_storage_dir")
        if beast_raw is not None:
            beast_resolved = _resolve_storage_path(beast_raw, repo_parent)
            assert beast_resolved is not None, "Resolved beast storage path unexpectedly None."
            if beast_resolved.exists():
                return beast_resolved

        # Local Beast/WSL runs often keep run_locally=False while still writing to local paths.
        if local_runtime:
            local_resolved = _resolve_storage_path(storage.get("local_storage_dir"), repo_parent)
            if local_resolved is not None and local_resolved.exists():
                return local_resolved
            return default_local.resolve()

        # Cluster-style fallback when configured roots do not exist.
        return default_bulk.resolve()
    except Exception as exc:
        LOGGER.warning("Config read error, using default storage root: %s", exc)
        return default_local.resolve()


def default_storage_root() -> Path:
    """Return the storage root resolved from SETTINGS (convenience wrapper)."""
    return get_storage_root(
        SETTINGS.general_config_path,
        SETTINGS.default_local_storage_dir,
        SETTINGS.default_bulk_storage_dir,
        SETTINGS.repo_root.parent,
    )


def get_storage_root_for_session(session_dir: Path | None) -> Path:
    """Resolve storage root from a session's general_config.yaml, falling back to default."""
    if session_dir is not None:
        session_cfg = session_dir / "general_config.yaml"
        if session_cfg.exists():
            return get_storage_root(
                session_cfg,
                SETTINGS.default_local_storage_dir,
                SETTINGS.default_bulk_storage_dir,
                SETTINGS.repo_root.parent,
            )
    return default_storage_root()


def find_mesh_assembly_dir(run_path: Path) -> Path | None:
    for candidate in MESH_ASSEMBLY_CANDIDATES:
        d = run_path / candidate
        if d.is_dir():
            return d
    return None


def has_any_gif_artifacts(run_path: Path) -> bool:
    mesh_dir = find_mesh_assembly_dir(run_path)
    mesh_gifs = [] if mesh_dir is None else list(mesh_dir.glob("*.gif"))
    solar_gifs = list((run_path / "solar_loss_rendering").glob("*.gif"))
    return bool(mesh_gifs or solar_gifs)


def has_display_artifacts(run_path: Path) -> bool:
    mesh_dir = run_path / "meshes"
    meshes = list(mesh_dir.glob("*.ply")) + list(mesh_dir.glob("*.poly")) + list(mesh_dir.glob("*.glb"))
    return has_any_gif_artifacts(run_path) or bool(meshes)


def collect_run_dirs(storage_root: Path) -> list[Path]:
    if not storage_root.exists():
        return []
    run_dirs: list[Path] = []
    try:
        model_dirs = [d for d in storage_root.iterdir() if d.is_dir()]
    except OSError as exc:
        LOGGER.warning("Cannot list storage root %s: %s", storage_root, exc)
        return []
    for model_dir in model_dirs:
        try:
            run_dirs.extend(
                child
                for child in model_dir.iterdir()
                if child.is_dir() and ((child / "meshes").exists() or find_mesh_assembly_dir(child) is not None)
            )
        except OSError as exc:
            LOGGER.warning("Cannot list model dir %s: %s", model_dir, exc)
    return run_dirs


def detect_current_run_dir(storage_root: Path, preferred_cfg_session: str | None = None) -> Path | None:
    """Return the current run dir.

    Selection policy:
    1) If preferred_cfg_session is provided, return the newest run tagged with
       that GUI config session.
    2) Otherwise, return the newest run by modification time.
    """
    run_dirs = collect_run_dirs(storage_root)
    if not run_dirs:
        return None

    if preferred_cfg_session:
        tagged = [p for p in run_dirs if read_cfg_session(p) == preferred_cfg_session]
        if tagged:
            return max(tagged, key=lambda p: p.stat().st_mtime)

    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def read_cfg_session(run_path: Path) -> str | None:
    """Return the basename of the cfg session tagged in gui_cfg_session.txt, or None."""
    tag_file = run_path / "gui_cfg_session.txt"
    try:
        # .name normalises to the folder basename regardless of whether the file
        # stores a full path or just the bare session folder name.
        return Path(tag_file.read_text(encoding="utf-8").strip()).name
    except OSError:
        return None


def list_all_runs(storage_root: Path) -> list[dict[str, object]]:
    """Return all run dirs with metadata, sorted newest-first."""
    run_dirs = collect_run_dirs(storage_root)
    return [
        {
            "path": str(d),
            "name": d.name,
            "model_hash": d.parent.name,
            "mtime": d.stat().st_mtime,
            "has_gifs": has_any_gif_artifacts(d),
            "has_meshes": (d / "meshes").exists(),
            "cfg_session": read_cfg_session(d),
            "run_id": f"{d.parent.name}/{d.name}",
        }
        for d in sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    ]


def extract_epoch(filename: str) -> int | None:
    match = re.search(r"epoch(\d+)", filename)
    if match is None:
        match = re.search(r"_(\d+)\.(?:png|gif|ply|poly|glb)$", filename)
    return int(match.group(1)) if match else None
