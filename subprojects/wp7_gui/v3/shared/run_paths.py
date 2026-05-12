"""Storage-root resolution and run-directory discovery.

Ports v2's GUIRequestHandler storage logic. Re-reads general_config.yaml on
every call to pick up run_locally flag changes without a server restart.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml

from subprojects.wp7_gui.v3.core.settings import SETTINGS

LOGGER = logging.getLogger(__name__)

MESH_ASSEMBLY_CANDIDATES = (
    "mesh_assembly_mode_v4",
    "mesh_assembly_mode_v3",
    "mesh_assembly_mode_v2",
    "mesh_assembly_mode",
)


def get_storage_root(
    general_config_path: Path,
    default_local: Path,
    default_bulk: Path,
    repo_parent: Path,
) -> Path:
    """Resolve the artifact storage root from wp4's general_config.yaml.

    Re-reads the file on every call (matches v2 behavior so toggling
    run_locally takes effect immediately).
    """
    try:
        with general_config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        run_locally_raw = cfg.get("general_info", {}).get("run_locally", False)
        # yaml.safe_load returns a Python bool for well-formed YAML booleans
        run_locally = run_locally_raw is True
        storage = cfg.get("storage_paths", {})

        raw = storage.get("local_storage_dir" if run_locally else "storage_dir", None)
        if raw is None:
            return (default_local if run_locally else default_bulk).resolve()

        root = Path(raw)
        return (repo_parent / root).resolve() if not root.is_absolute() else root.resolve()
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
    meshes = list((run_path / "meshes").glob("*.ply")) + list((run_path / "meshes").glob("*.poly"))
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


def detect_current_run_dir(storage_root: Path) -> Path | None:
    """Return the most-recently-modified run dir that has artifacts."""
    run_dirs = collect_run_dirs(storage_root)
    if not run_dirs:
        return None
    with_gifs = [p for p in run_dirs if has_any_gif_artifacts(p)]
    if with_gifs:
        return max(with_gifs, key=lambda p: p.stat().st_mtime)
    with_artifacts = [p for p in run_dirs if has_display_artifacts(p)]
    if with_artifacts:
        return max(with_artifacts, key=lambda p: p.stat().st_mtime)
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
        match = re.search(r"_(\d+)\.(?:png|gif|ply|poly)$", filename)
    return int(match.group(1)) if match else None
