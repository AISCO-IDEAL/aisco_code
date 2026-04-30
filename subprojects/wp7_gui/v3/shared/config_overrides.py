"""Config-override session management.

Port of v2's ConfigOverrideManager. Manages timestamped session folders under
config_overrides_root, validates edits against gui_schema.json, and persists
back to YAML. The WP4 interface (--cfg-folder contract) is unchanged.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

_OVERRIDE_PREFIX = "gui_override_"
_COPY_EXCLUDE = ("sweeps",)


def _fetch_wp4_config_paths(wp4_configs_dir: Path, exclude: tuple[str, ...] = _COPY_EXCLUDE) -> list[Path]:
    all_yamls = list(wp4_configs_dir.rglob("*.yaml"))
    if not exclude:
        return all_yamls
    excluded_dirs = {(wp4_configs_dir / sub).resolve() for sub in exclude}
    return [
        p for p in all_yamls
        if not any(
            str(p.resolve()).startswith(str(ex) + "/") or p.resolve() == ex
            for ex in excluded_dirs
        )
    ]


def _load_schema(schema_path: Path) -> dict[str, Any]:
    if not schema_path.exists():
        raise HTTPException(status_code=500, detail={"message": f"Schema missing: {schema_path}"})
    try:
        return json.loads(schema_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail={"message": f"Invalid gui_schema.json: {exc}"}) from exc


def _resolve_session(folder_str: str, root: Path) -> Path:
    try:
        candidate = Path(folder_str).resolve()
        candidate.relative_to(root.resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail={"message": f"Folder outside override root: {folder_str}"}) from exc
    if not candidate.name.startswith(_OVERRIDE_PREFIX):
        raise HTTPException(
            status_code=400,
            detail={"message": f"Folder must start with '{_OVERRIDE_PREFIX}': {candidate.name}"},
        )
    return candidate


def _read_dotted(cfg: dict, segments: list[str]) -> Any:
    d: Any = cfg
    for seg in segments:
        if not isinstance(d, dict) or seg not in d:
            return None
        d = d[seg]
    return d


def _write_dotted(cfg: dict, segments: list[str], value: Any) -> None:
    d = cfg
    for seg in segments[:-1]:
        if seg not in d or not isinstance(d[seg], dict):
            d[seg] = {}
        d = d[seg]
    d[segments[-1]] = value


def _validate_value(param: dict[str, Any], raw: Any, loc: str) -> Any:  # noqa: PLR0912
    ptype = param.get("type")
    coerced: int | float

    if ptype == "int":
        if isinstance(raw, bool) or not isinstance(raw, int):
            try:
                coerced = int(raw)
                if isinstance(raw, float) and not raw.is_integer():
                    raise ValueError  # noqa: TRY301
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail={"message": f"{loc}: expected int, got {raw!r}"}) from exc
        else:
            coerced = raw
        if "min" in param and coerced < param["min"]:
            raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} < min {param['min']}"})
        if "max" in param and coerced > param["max"]:
            raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} > max {param['max']}"})
        return coerced

    if ptype == "float":
        if isinstance(raw, bool):
            raise HTTPException(status_code=400, detail={"message": f"{loc}: expected float, got bool"})
        try:
            coerced = float(raw)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail={"message": f"{loc}: expected float, got {raw!r}"}) from exc
        if "min" in param and coerced < param["min"]:
            raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} < min {param['min']}"})
        if "max" in param and coerced > param["max"]:
            raise HTTPException(status_code=400, detail={"message": f"{loc}: {coerced} > max {param['max']}"})
        return coerced

    if ptype == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in {"true", "false"}:
            return raw.lower() == "true"
        raise HTTPException(status_code=400, detail={"message": f"{loc}: expected bool, got {raw!r}"})

    if ptype == "string":
        if not isinstance(raw, str):
            raise HTTPException(status_code=400, detail={"message": f"{loc}: expected string, got {type(raw).__name__}"})
        return raw

    if ptype == "enum":
        choices = param.get("choices", [])
        if raw not in choices:
            raise HTTPException(status_code=400, detail={"message": f"{loc}: {raw!r} not in {choices}"})
        return raw

    raise HTTPException(status_code=500, detail={"message": f"{loc}: unknown type {ptype!r}"})


class ConfigOverrideManager:
    def __init__(self, root: Path, schema_path: Path, wp4_configs_dir: Path) -> None:
        self._root = root
        self._schema_path = schema_path
        self._wp4_configs_dir = wp4_configs_dir

    def init_session(self) -> dict[str, Any]:
        self._root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self._root / f"{_OVERRIDE_PREFIX}{stamp}"
        if session_dir.exists():
            session_dir = self._root / f"{_OVERRIDE_PREFIX}{stamp}_{os.getpid()}"
        session_dir.mkdir(parents=True)

        sources = _fetch_wp4_config_paths(self._wp4_configs_dir)
        copied: list[str] = []
        for src in sources:
            rel = src.relative_to(self._wp4_configs_dir)
            dst = session_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(str(rel))

        return {
            "folder": str(session_dir),
            "name": session_dir.name,
            "copied_files": copied,
            "skipped_subfolders": list(_COPY_EXCLUDE),
        }

    def get_schema(self) -> dict[str, Any]:
        return _load_schema(self._schema_path)

    def get_values(self, folder_str: str) -> dict[str, Any]:
        session_dir = _resolve_session(folder_str, self._root)
        schema = _load_schema(self._schema_path)
        result: dict[str, dict[str, Any]] = {}
        for cfg_name, cfg_entry in schema.get("configs", {}).items():
            yaml_path = session_dir / cfg_name
            if not yaml_path.exists():
                continue
            with yaml_path.open(encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}
            result[cfg_name] = {
                p["path"]: _read_dotted(cfg_dict, p["path"].split("."))
                for p in cfg_entry.get("params", [])
            }
        return {"folder": str(session_dir), "values": result}

    def set_values(self, folder_str: str, updates: dict[str, dict[str, Any]]) -> dict[str, Any]:
        session_dir = _resolve_session(folder_str, self._root)
        schema = _load_schema(self._schema_path)
        schema_configs = schema.get("configs", {})

        validated: dict[str, list[tuple[list[str], Any]]] = {}
        for cfg_name, cfg_updates in updates.items():
            if cfg_name not in schema_configs:
                raise HTTPException(status_code=400, detail={"message": f"Config not editable: {cfg_name}"})
            param_lookup = {p["path"]: p for p in schema_configs[cfg_name].get("params", [])}
            entries = []
            for dotted_path, raw in cfg_updates.items():
                if dotted_path not in param_lookup:
                    raise HTTPException(
                        status_code=400,
                        detail={"message": f"Parameter not in schema: {cfg_name}::{dotted_path}"},
                    )
                coerced = _validate_value(param_lookup[dotted_path], raw, f"{cfg_name}::{dotted_path}")
                entries.append((dotted_path.split("."), coerced))
            validated[cfg_name] = entries

        written: list[str] = []
        for cfg_name, segments_and_values in validated.items():
            yaml_path = session_dir / cfg_name
            if not yaml_path.exists():
                raise HTTPException(status_code=404, detail={"message": f"YAML missing in session: {cfg_name}"})
            with yaml_path.open(encoding="utf-8") as f:
                cfg_dict = yaml.safe_load(f) or {}
            for segments, value in segments_and_values:
                _write_dotted(cfg_dict, segments, value)
            with yaml_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg_dict, f, sort_keys=False, default_flow_style=False)
            written.append(cfg_name)

        return {"folder": str(session_dir), "written": written}

    def list_sessions(self) -> dict[str, Any]:
        if not self._root.exists():
            return {"sessions": []}
        sessions = [
            {"name": e.name, "folder": str(e), "mtime": e.stat().st_mtime}
            for e in sorted(self._root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if e.is_dir() and e.name.startswith(_OVERRIDE_PREFIX)
        ]
        return {"sessions": sessions}

    def cleanup(self, folders: list[str] | None, all_sessions: bool) -> dict[str, Any]:
        removed: list[str] = []
        if all_sessions:
            if not self._root.exists():
                return {"removed": []}
            for e in self._root.iterdir():
                if e.is_dir() and e.name.startswith(_OVERRIDE_PREFIX):
                    shutil.rmtree(e)
                    removed.append(e.name)
            return {"removed": removed}
        for folder_str in folders or []:
            session_dir = _resolve_session(folder_str, self._root)
            if session_dir.exists():
                shutil.rmtree(session_dir)
                removed.append(session_dir.name)
        return {"removed": removed}
