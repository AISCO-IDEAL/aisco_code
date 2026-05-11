import importlib.util
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException
from fastapi.responses import FileResponse


class AssemblySelectionService:
    """Backend service for WP2 assembly selection sliders and mesh refresh."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._project_root = repo_root.parent
        self._wp2_gui_helpers_dir = repo_root / "wp2_assembler" / "version4" / "GUI_helpers"
        self._parameters_path = self._wp2_gui_helpers_dir / "parameters.yaml"
        self._assembly_ply_path = self._wp2_gui_helpers_dir / "assembly.ply"
        self._gui_assembly_script = self._wp2_gui_helpers_dir / "GUI_assembly.py"

        self._lock = threading.Lock()
        self._gui_assembly_module: Any | None = None

    def _read_parameters(self) -> dict[str, dict[str, float]]:
        if not self._parameters_path.exists():
            raise HTTPException(
                status_code=500,
                detail={"message": f"Missing parameters file at {self._parameters_path}"},
            )

        try:
            with self._parameters_path.open(encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
        except OSError as error:
            raise HTTPException(status_code=500, detail={"message": str(error)}) from error

        if not isinstance(raw, dict):
            raise HTTPException(status_code=500, detail={"message": "parameters.yaml must contain a mapping at top level"})

        normalized: dict[str, dict[str, float]] = {}
        for section, values in raw.items():
            if not isinstance(section, str) or not isinstance(values, dict):
                continue

            axes: dict[str, float] = {}
            for axis in ("x", "y", "z"):
                value = values.get(axis)
                if isinstance(value, (int, float)):
                    axes[axis] = float(value)
            if axes:
                normalized[section] = axes

        return normalized

    def _write_parameters(self, params: dict[str, dict[str, float]]) -> None:
        self._parameters_path.parent.mkdir(parents=True, exist_ok=True)
        with self._parameters_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(params, handle, default_flow_style=False, sort_keys=True)

    @staticmethod
    def _coerce_payload(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail={"message": "Payload must be a JSON object"})

        result: dict[str, dict[str, float]] = {}
        for section, values in payload.items():
            if not isinstance(section, str) or not isinstance(values, dict):
                continue

            section_values: dict[str, float] = {}
            for axis in ("x", "y", "z"):
                value = values.get(axis)
                if isinstance(value, (int, float)):
                    section_values[axis] = float(value)
            if section_values:
                result[section] = section_values

        if not result:
            raise HTTPException(status_code=400, detail={"message": "No valid section values were provided"})

        return result

    def _load_gui_assembly_module(self) -> Any:
        if self._gui_assembly_module is not None:
            return self._gui_assembly_module

        # Ensure local repository modules are preferred over site-packages.
        project_root_str = str(self._project_root.resolve())
        if project_root_str in sys.path:
            sys.path.remove(project_root_str)
        sys.path.insert(0, project_root_str)

        if not self._gui_assembly_script.exists():
            raise HTTPException(
                status_code=500,
                detail={"message": f"Missing GUI assembly script at {self._gui_assembly_script}"},
            )

        spec = importlib.util.spec_from_file_location("wp2_gui_assembly_runtime", str(self._gui_assembly_script))
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail={"message": "Failed to create module spec for GUI_assembly.py"})

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as error:
            raise HTTPException(status_code=500, detail={"message": f"Failed to import GUI_assembly.py: {error}"}) from error

        self._gui_assembly_module = module
        return module

    def _rebuild_assembly(self) -> None:
        module = self._load_gui_assembly_module()
        call_assembly_from_params = getattr(module, "call_assembly_from_params", None)
        if call_assembly_from_params is None:
            raise HTTPException(status_code=500, detail={"message": "GUI_assembly.py does not expose call_assembly_from_params"})

        try:
            call_assembly_from_params()
        except Exception as error:
            raise HTTPException(status_code=500, detail={"message": f"Assembly rebuild failed: {error}"}) from error

    def get_params(self) -> dict[str, dict[str, float]]:
        return self._read_parameters()

    def update_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        params = self._coerce_payload(payload)
        with self._lock:
            self._write_parameters(params)
            self._rebuild_assembly()

        return {
            "status": "ok",
            "message": f"[{datetime.now().strftime('%H:%M:%S')}] Updated assembly parameters and rebuilt mesh.",
            "mesh_url": "/assembly-selection/mesh",
            "params": params,
        }

    def add_wheel(self) -> dict[str, Any]:
        with self._lock:
            params = self._read_parameters()
            existing = [key for key in params if key.startswith("wheel_")]
            next_idx = max((int(key.split("_")[1]) for key in existing), default=-1) + 1
            params[f"wheel_{next_idx}"] = {"x": 0.0, "y": 0.0, "z": 0.0}

            self._write_parameters(params)
            self._rebuild_assembly()

        return {
            "status": "ok",
            "message": f"Added wheel_{next_idx}.",
            "params": params,
            "mesh_url": "/assembly-selection/mesh",
        }

    def remove_wheel(self) -> dict[str, Any]:
        with self._lock:
            params = self._read_parameters()
            existing = sorted(key for key in params if key.startswith("wheel_"))
            if not existing:
                return {
                    "status": "ok",
                    "message": "No wheels to remove.",
                    "params": params,
                    "mesh_url": "/assembly-selection/mesh",
                }

            removed = existing[-1]
            params.pop(removed, None)
            self._write_parameters(params)
            self._rebuild_assembly()

        return {
            "status": "ok",
            "message": f"Removed {removed}.",
            "params": params,
            "mesh_url": "/assembly-selection/mesh",
        }

    def get_mesh_response(self) -> FileResponse:
        if not self._assembly_ply_path.exists() or not self._assembly_ply_path.is_file():
            raise HTTPException(
                status_code=404,
                detail={"message": f"assembly.ply not found at {self._assembly_ply_path}"},
            )

        return FileResponse(
            str(self._assembly_ply_path),
            media_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )
