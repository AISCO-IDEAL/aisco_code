"""Pretraining GIF listing and serving — port of v2's PretrainingService."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse


class PretrainingService:
    def __init__(self, storage_root: Path) -> None:
        self._root = storage_root.resolve()

    def list_gifs(self) -> dict[str, Any]:
        if not self._root.is_dir():
            return {"status": "ok", "message": f"Pretraining folder not found: {self._root}", "source_dir": None, "gifs": []}
        selected = sorted(self._root.glob("*.gif"), key=lambda p: p.name.lower())[:4]
        items = [
            {
                "name": p.name,
                "rel_path": p.relative_to(self._root).as_posix(),
                "url": f"/viewer/api/pretraining/gif-file?path={p.relative_to(self._root).as_posix()}",
            }
            for p in selected
        ]
        return {"status": "ok", "message": f"Loaded {len(items)} GIF(s)", "source_dir": str(self._root), "gifs": items}

    def get_gif_response(self, rel_path: str) -> FileResponse:
        rel_path = rel_path.strip().lstrip("/")
        if not rel_path:
            raise HTTPException(status_code=400, detail={"message": "Missing path"})
        try:
            p = (self._root / rel_path).resolve()
            p.relative_to(self._root)
        except Exception as exc:
            raise HTTPException(status_code=400, detail={"message": "Invalid GIF path"}) from exc
        if not p.is_file():
            raise HTTPException(status_code=404, detail={"message": "GIF not found"})
        if p.suffix.lower() != ".gif":
            raise HTTPException(status_code=400, detail={"message": "Only GIF files supported"})
        return FileResponse(str(p), media_type="image/gif", headers={"Cache-Control": "no-store"})
