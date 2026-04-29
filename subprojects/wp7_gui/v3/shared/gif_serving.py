"""GIF file serving: direct serve and slow-playback with mtime-keyed cache."""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
from fastapi import HTTPException
from fastapi.responses import FileResponse


def _resolve_under_root(storage_root: Path, rel_path: str) -> Path | None:
    try:
        resolved = (storage_root / rel_path.lstrip("/")).resolve()
        resolved.relative_to(storage_root.resolve())
    except Exception:
        return None
    return resolved


def serve_file(storage_root: Path, rel_path: str) -> FileResponse:
    """Serve any artifact file under storage_root."""
    if not rel_path:
        raise HTTPException(status_code=400, detail={"message": "Missing path"})
    p = _resolve_under_root(storage_root, rel_path)
    if p is None:
        raise HTTPException(status_code=400, detail={"message": "Invalid path"})
    if not p.is_file():
        raise HTTPException(status_code=404, detail={"message": "File not found"})
    mime = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
    return FileResponse(str(p), media_type=mime, headers={"Cache-Control": "no-store"})


def serve_slow_gif(storage_root: Path, rel_path: str, factor: float, cache_dir: Path) -> FileResponse:
    """Serve a speed-reduced copy of a GIF, cached by (path, mtime, factor)."""
    if not rel_path:
        raise HTTPException(status_code=400, detail={"message": "Missing path"})
    factor = max(1.0, min(factor, 10.0))
    p = _resolve_under_root(storage_root, rel_path)
    if p is None:
        raise HTTPException(status_code=400, detail={"message": "Invalid path"})
    if not p.is_file():
        raise HTTPException(status_code=404, detail={"message": "File not found"})
    if p.suffix.lower() != ".gif":
        raise HTTPException(status_code=400, detail={"message": "Only GIF files supported for slow playback"})

    cache_key = hashlib.sha1(
        f"{p.resolve()}|{p.stat().st_mtime_ns}|{factor:.2f}".encode()
    ).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"slow_{cache_key}.gif"

    if not cached.exists():
        try:
            frames = imageio.mimread(p)
            meta = imageio.get_reader(p).get_meta_data()
            base_duration = float(meta.get("duration", 0.08))
            slow_duration = max(0.02, base_duration * factor)
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"message": f"Failed to build slowed GIF: {exc}"}) from exc
        if not frames:
            raise HTTPException(status_code=500, detail={"message": "Failed to decode GIF frames"})
        imageio.mimsave(cached, cast(list[Any], frames), duration=slow_duration)

    return FileResponse(str(cached), media_type="image/gif", headers={"Cache-Control": "no-store"})
