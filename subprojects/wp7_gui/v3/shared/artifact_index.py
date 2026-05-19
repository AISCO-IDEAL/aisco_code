"""Enumerate artifacts inside a run directory.

from __future__ import annotations
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from subprojects.wp7_gui.v3.shared.run_paths import extract_epoch, find_mesh_assembly_dir

# Sub-directories whose GIFs are already captured by dedicated top-level keys.
_HANDLED_DIRS = frozenset(
    {
        "mesh_assembly_v4",
        "mesh_assembly_mode_v4",
        "mesh_assembly_mode_v3",
        "mesh_assembly_mode_v2",
        "mesh_assembly_mode",
        "solar_loss_rendering",
        "meshes",
        "models",
    }
)


def _humanize_path(rel: str) -> str:
    """Convert a folder path like 'mesh_gradients/gaot_loss' → 'Mesh Gradients – Gaot Loss'."""
    parts = [p.replace("_", " ").title() for p in rel.split("/")]
    return " \u2013 ".join(parts)


# This aggregator intentionally scans multiple artifact types and directory depths
# in one pass to keep API payload construction in a single place.
def list_artifacts(run_dir: Path, storage_root: Path) -> dict[str, object]:  # noqa: PLR0912, PLR0915
    """Return artifact lists for a given run directory.

    Returns:
        dict with keys: run_dir (str), storage_root (str), mesh_assembly_gifs,
        solar_loss_gifs, meshes, meshes_glb, extra_gifs — each a list of
        artifact dicts with name, rel_path, url (and slow_url / label for gifs).
    """
    storage_root_resolved = storage_root.resolve()
    run_dir_resolved = run_dir.resolve()

    def _rel(p: Path) -> str:
        try:
            return p.resolve().relative_to(storage_root_resolved).as_posix()
        except ValueError:
            return p.name

    mesh_assembly_dir = find_mesh_assembly_dir(run_dir_resolved)
    solar_loss_dir = run_dir_resolved / "solar_loss_rendering"
    mesh_dir = run_dir_resolved / "meshes"

    gifs = []
    if mesh_assembly_dir is not None and mesh_assembly_dir.exists():
        for p in sorted(mesh_assembly_dir.glob("*.gif"), key=lambda f: f.stat().st_mtime, reverse=True):
            rel = _rel(p)
            gifs.append(
                {
                    "name": p.name,
                    "rel_path": rel,
                    "url": f"/api/artifacts/file?path={rel}",
                    "slow_url": f"/api/artifacts/slow-gif?path={rel}&factor=2.5",
                }
            )

    solar_gifs = []
    if solar_loss_dir.exists():
        for p in sorted(solar_loss_dir.glob("*.gif"), key=lambda f: f.stat().st_mtime, reverse=True):
            rel = _rel(p)
            solar_gifs.append(
                {
                    "name": p.name,
                    "rel_path": rel,
                    "url": f"/api/artifacts/file?path={rel}",
                    "slow_url": f"/api/artifacts/slow-gif?path={rel}&factor=2.5",
                }
            )

    meshes: list[dict[str, object]] = []
    meshes_glb: list[dict[str, object]] = []
    seen_mesh_rel_paths: set[str] = set()

    def _append_mesh_file(p: Path, target: list[dict[str, object]]) -> None:
        rel = _rel(p)
        if rel in seen_mesh_rel_paths:
            return
        seen_mesh_rel_paths.add(rel)
        target.append(
            {
                "name": p.name,
                "rel_path": rel,
                "url": f"/api/artifacts/file?path={rel}",
                "epoch": extract_epoch(p.name),
            }
        )

    if mesh_dir.exists():
        for p in sorted(list(mesh_dir.glob("*.ply")) + list(mesh_dir.glob("*.poly"))):
            _append_mesh_file(p, meshes)
        for p in sorted(mesh_dir.glob("*.glb")):
            _append_mesh_file(p, meshes_glb)

    # Some runs write renderable PLYs under mesh_assembly_v4 (or sibling candidates)
    # instead of meshes/. Include them so launcher/output and viewer/archive can render.
    if mesh_assembly_dir is not None and mesh_assembly_dir.exists():
        for p in sorted(mesh_assembly_dir.glob("*.ply")):
            _append_mesh_file(p, meshes)
        for p in sorted(mesh_assembly_dir.glob("*.glb")):
            _append_mesh_file(p, meshes_glb)

    meshes.sort(key=lambda m: cast(int, m["epoch"]) if isinstance(m.get("epoch"), int) else -1)
    meshes_glb.sort(key=lambda m: cast(int, m["epoch"]) if isinstance(m.get("epoch"), int) else -1)

    # --- Extra GIFs: scan all subdirs (up to 2 levels) not already handled ---
    extra_gifs: list[dict[str, object]] = []

    def _add_gif_from_dir(subdir: Path, label: str) -> None:
        """Add one GIF entry for subdir: prefer animation.gif, else first *.gif found."""
        preferred = subdir / "animation.gif"
        if preferred.is_file():
            r = _rel(preferred)
            extra_gifs.append(
                {
                    "label": label,
                    "name": preferred.name,
                    "rel_path": r,
                    "url": f"/api/artifacts/file?path={r}",
                    "slow_url": f"/api/artifacts/slow-gif?path={r}&factor=2.5",
                }
            )
        else:
            for gif in sorted(subdir.glob("*.gif")):
                r = _rel(gif)
                extra_gifs.append(
                    {
                        "label": label,
                        "name": gif.name,
                        "rel_path": r,
                        "url": f"/api/artifacts/file?path={r}",
                        "slow_url": f"/api/artifacts/slow-gif?path={r}&factor=2.5",
                    }
                )
                break  # one representative GIF per dir

    if run_dir_resolved.exists():
        try:
            level1_dirs = sorted(d for d in run_dir_resolved.iterdir() if d.is_dir())
        except OSError:
            level1_dirs = []
        for level1 in level1_dirs:
            if level1.name in _HANDLED_DIRS:
                continue
            has_gifs = any(level1.glob("*.gif"))
            if has_gifs:
                _add_gif_from_dir(level1, _humanize_path(level1.name))
            # One level deeper (e.g. mesh_gradients/gaot_loss, aero/cp_field)
            try:
                for level2 in sorted(d for d in level1.iterdir() if d.is_dir()):
                    if any(level2.glob("*.gif")):
                        _add_gif_from_dir(level2, _humanize_path(f"{level1.name}/{level2.name}"))
            except OSError:
                pass

    # --- Extra PLY groups: collect .ply files organized by subdirectory ---
    extra_ply_groups: list[dict[str, object]] = []

    def _collect_ply_group(subdir: Path, label: str) -> None:
        """Collect all .ply files from subdir, sorted by epoch."""
        ply_files = sorted(subdir.glob("*.ply"))
        if not ply_files:
            return
        plys = []
        for p in ply_files:
            r = _rel(p)
            epoch = extract_epoch(p.name)
            plys.append(
                {
                    "name": p.name,
                    "rel_path": r,
                    "url": f"/api/artifacts/file?path={r}",
                    "epoch": epoch,
                }
            )
        plys.sort(key=lambda m: m["epoch"] if m["epoch"] is not None else -1)
        if plys:
            extra_ply_groups.append(
                {
                    "label": label,
                    "ply_files": plys,
                }
            )

    if run_dir_resolved.exists():
        try:
            level1_dirs = sorted(d for d in run_dir_resolved.iterdir() if d.is_dir())
        except OSError:
            level1_dirs = []
        for level1 in level1_dirs:
            if level1.name in _HANDLED_DIRS:
                continue
            # Check for .ply files in level 1 (most subdirs)
            has_plys = any(level1.glob("*.ply"))
            if has_plys:
                _collect_ply_group(level1, _humanize_path(level1.name))
            # Check level 2 (e.g. mesh_gradients/gaot_loss)
            try:
                for level2 in sorted(d for d in level1.iterdir() if d.is_dir()):
                    if any(level2.glob("*.ply")):
                        _collect_ply_group(level2, _humanize_path(f"{level1.name}/{level2.name}"))
            except OSError:
                pass

    return {
        "run_dir": str(run_dir_resolved),
        "storage_root": str(storage_root_resolved),
        "mesh_assembly_gifs": gifs,
        "solar_loss_gifs": solar_gifs,
        "meshes": meshes,
        "meshes_glb": meshes_glb,
        "extra_gifs": extra_gifs,
        "extra_ply_groups": extra_ply_groups,
    }
