"""Enumerate artifacts inside a run directory.

Returns the same shape as v2's /run-local/artifacts so the JS components
can be shared between live launcher view and finished-run viewer.
"""

from __future__ import annotations

from pathlib import Path

from subprojects.wp7_gui.v3.shared.run_paths import extract_epoch, find_mesh_assembly_dir


def list_artifacts(run_dir: Path, storage_root: Path) -> dict[str, object]:
    """Return artifact lists for a given run directory.

    Returns:
        dict with keys: run_dir (str), storage_root (str), mesh_assembly_gifs,
        solar_loss_gifs, meshes — each a list of artifact dicts with name,
        rel_path, url (and slow_url for gifs).
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

    meshes = []
    if mesh_dir.exists():
        raw = sorted(list(mesh_dir.glob("*.ply")) + list(mesh_dir.glob("*.poly")))
        for p in raw:
            rel = _rel(p)
            meshes.append(
                {
                    "name": p.name,
                    "rel_path": rel,
                    "url": f"/api/artifacts/file?path={rel}",
                    "epoch": extract_epoch(p.name),
                }
            )
        meshes.sort(key=lambda m: m["epoch"] if m["epoch"] is not None else -1)

    return {
        "run_dir": str(run_dir_resolved),
        "storage_root": str(storage_root_resolved),
        "mesh_assembly_gifs": gifs,
        "solar_loss_gifs": solar_gifs,
        "meshes": meshes,
    }
