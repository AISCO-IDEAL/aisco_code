"""Stdlib filesystem poller for live artifact detection.

Scans a run directory's mtime-tracked artifact paths and returns newly
appeared files since the last call. Used by SSE endpoints to push
artifact-arrival events without the watchdog dependency.
"""
from __future__ import annotations

from pathlib import Path

# Subdirectories and glob patterns to watch inside a run dir.
_WATCH_SPECS: list[tuple[str, str]] = [
    ("meshes", "*.ply"),
    ("meshes", "*.poly"),
    ("mesh_assembly_mode_v4", "*.gif"),
    ("mesh_assembly_mode_v3", "*.gif"),
    ("mesh_assembly_mode_v2", "*.gif"),
    ("mesh_assembly_mode", "*.gif"),
    ("solar_loss_rendering", "*.gif"),
    ("gradient_conflict", "*.gif"),
]


def poll_once(
    run_dir: Path,
    seen_mtimes: dict[str, float],
) -> tuple[list[dict[str, str]], dict[str, float]]:
    """Check run_dir for new artifacts since seen_mtimes was last recorded.

    Args:
        run_dir: the `<storage_root>/<model_hash>/<wandb_run>` directory
        seen_mtimes: path→mtime mapping from the previous call (mutated in place)

    Returns:
        (new_events, updated_mtimes) where each event is
        {"kind": "mesh"|"gif", "name": filename, "path": absolute path str}
    """
    new_events: list[dict[str, str]] = []
    updated = dict(seen_mtimes)

    for subdir, pattern in _WATCH_SPECS:
        candidate = run_dir / subdir
        if not candidate.is_dir():
            continue
        kind = "mesh" if pattern.endswith(".ply") or pattern.endswith(".poly") else "gif"
        for p in candidate.glob(pattern):
            key = str(p)
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            if key not in updated or updated[key] != mtime:
                updated[key] = mtime
                if key not in seen_mtimes:
                    new_events.append({"kind": kind, "name": p.name, "path": key})

    return new_events, updated
