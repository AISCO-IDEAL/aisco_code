from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse


class PretrainingService:
    """Service layer for listing and serving pretraining GIFs."""

    def __init__(self, storage_root: Path) -> None:
        self._storage_root = storage_root.resolve()

    def list_gifs(self) -> dict[str, Any]:
        if not self._storage_root.exists() or not self._storage_root.is_dir():
            return {
                "status": "ok",
                "message": f"Pretraining GIF folder not found: {self._storage_root}",
                "source_dir": None,
                "gifs": [],
            }

        gif_paths = sorted(self._storage_root.glob("*.gif"), key=lambda path: path.name.lower())
        selected = gif_paths[:4]

        items: list[dict[str, str]] = []
        for gif_path in selected:
            rel_path = gif_path.relative_to(self._storage_root).as_posix()
            items.append(
                {
                    "name": gif_path.name,
                    "rel_path": rel_path,
                    "url": f"/pretraining/gif-file?path={rel_path}",
                }
            )

        return {
            "status": "ok",
            "message": f"Loaded {len(items)} GIF(s) from {self._storage_root}",
            "source_dir": str(self._storage_root),
            "gifs": items,
        }

    def get_gif_response(self, rel_path: str) -> FileResponse:
        rel_path = rel_path.strip().lstrip("/")
        if not rel_path:
            raise HTTPException(status_code=400, detail={"message": "Missing path query parameter"})

        gif_path = self._resolve_path_under_root(rel_path)
        if gif_path is None:
            raise HTTPException(status_code=400, detail={"message": "Invalid GIF path"})

        if not gif_path.exists() or not gif_path.is_file():
            raise HTTPException(status_code=404, detail={"message": "GIF file not found"})

        if gif_path.suffix.lower() != ".gif":
            raise HTTPException(status_code=400, detail={"message": "Only GIF files are supported"})

        return FileResponse(str(gif_path), media_type="image/gif", headers={"Cache-Control": "no-store"})

    def _resolve_path_under_root(self, rel_path: str) -> Path | None:
        try:
            resolved = (self._storage_root / rel_path).resolve()
            resolved.relative_to(self._storage_root)
        except Exception:
            return None

        return resolved
