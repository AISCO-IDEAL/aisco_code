"""WP2 assembly selection service — port of v2's AssemblySelectionService."""

from __future__ import annotations

import importlib.util
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException
from fastapi.responses import FileResponse


class AssemblySelectionService:
    def __init__(self, repo_root: Path) -> None:
        self._script = repo_root / "wp7_gui" / "v3" / "shared" / "GUI_assembly.py"
        # GUI_assembly.py writes all I/O to frontend/assembly/
        self._assembly_dir = repo_root / "wp7_gui" / "v3" / "frontend" / "assembly"
        self._params_path = self._assembly_dir / "parameters.yaml"
        self._assembly_ply = self._assembly_dir / "assembly.ply"
        self._lock = threading.Lock()
        self._module: Any | None = None

    def _read_params(self) -> dict[str, dict[str, float]]:
        if not self._params_path.exists():
            raise HTTPException(status_code=500, detail={"message": f"Missing parameters.yaml at {self._params_path}"})
        try:
            with self._params_path.open(encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        except OSError as exc:
            raise HTTPException(status_code=500, detail={"message": str(exc)}) from exc
        result: dict[str, dict[str, float]] = {}
        for section, values in raw.items():
            if not isinstance(section, str) or not isinstance(values, dict):
                continue
            axes = {ax: float(values[ax]) for ax in ("x", "y", "z") if isinstance(values.get(ax), (int, float))}
            if axes:
                result[section] = axes
        return result

    def _write_params(self, params: dict[str, dict[str, float]]) -> None:
        self._params_path.parent.mkdir(parents=True, exist_ok=True)
        with self._params_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(params, f, default_flow_style=False, sort_keys=True)

    def _rebuild(self) -> None:
        if self._module is None:
            if not self._script.exists():
                raise HTTPException(status_code=500, detail={"message": f"Missing GUI_assembly.py at {self._script}"})
            spec = importlib.util.spec_from_file_location("wp2_gui_assembly_runtime", str(self._script))
            if spec is None or spec.loader is None:
                raise HTTPException(status_code=500, detail={"message": "Failed to load GUI_assembly.py spec"})
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise HTTPException(status_code=500, detail={"message": f"Failed to import GUI_assembly.py: {exc}"}) from exc
            self._module = module

        fn = getattr(self._module, "call_assembly_from_params", None)
        if fn is None:
            raise HTTPException(status_code=500, detail={"message": "GUI_assembly.py missing call_assembly_from_params"})
        try:
            fn()
        except Exception as exc:
            raise HTTPException(status_code=500, detail={"message": f"Assembly rebuild failed: {exc}"}) from exc

    def get_params(self) -> dict[str, dict[str, float]]:
        return self._read_params()

    def update_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail={"message": "Payload must be a JSON object"})
        params: dict[str, dict[str, float]] = {}
        for section, values in payload.items():
            if not isinstance(section, str) or not isinstance(values, dict):
                continue
            axes = {ax: float(values[ax]) for ax in ("x", "y", "z") if isinstance(values.get(ax), (int, float))}
            if axes:
                params[section] = axes
        if not params:
            raise HTTPException(status_code=400, detail={"message": "No valid section values provided"})
        with self._lock:
            self._write_params(params)
            self._rebuild()
        return {
            "status": "ok",
            "message": f"[{datetime.now():%H:%M:%S}] Updated parameters and rebuilt mesh.",
            "mesh_url": "/launcher/api/assembly-selection/mesh",
            "params": params,
        }

    def add_wheel(self) -> dict[str, Any]:
        with self._lock:
            params = self._read_params()
            existing = [k for k in params if k.startswith("wheel_")]
            next_idx = max((int(k.split("_")[1]) for k in existing), default=-1) + 1
            params[f"wheel_{next_idx}"] = {"x": 0.0, "y": 0.0, "z": 0.0}
            self._write_params(params)
            self._rebuild()
        return {
            "status": "ok",
            "message": f"Added wheel_{next_idx}.",
            "params": params,
            "mesh_url": "/launcher/api/assembly-selection/mesh",
        }

    def remove_wheel(self) -> dict[str, Any]:
        with self._lock:
            params = self._read_params()
            existing = sorted(k for k in params if k.startswith("wheel_"))
            if not existing:
                return {
                    "status": "ok",
                    "message": "No wheels to remove.",
                    "params": params,
                    "mesh_url": "/launcher/api/assembly-selection/mesh",
                }
            removed = existing[-1]
            params.pop(removed, None)
            self._write_params(params)
            self._rebuild()
        return {
            "status": "ok",
            "message": f"Removed {removed}.",
            "params": params,
            "mesh_url": "/launcher/api/assembly-selection/mesh",
        }

    def get_mesh_response(self) -> FileResponse:
        if not self._assembly_ply.is_file():
            raise HTTPException(status_code=404, detail={"message": f"assembly.ply not found at {self._assembly_ply}"})
        return FileResponse(str(self._assembly_ply), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})

    def get_checks(self) -> dict:
        checks_path = self._assembly_dir / "checks.txt"
        if not checks_path.exists():
            return {"warnings": [], "ok": True}
        try:
            lines = checks_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return {"warnings": [], "ok": True}
        warnings = [warn.strip() for warn in lines if warn.strip().upper().startswith("[WARNING]")]
        return {"warnings": warnings, "ok": len(warnings) == 0}
