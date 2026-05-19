"""Create 3D-printing artifacts from existing AISCO exports.

This is intentionally a post-processor. It reads meshes from a run folder,
reconstructs deterministic repair geometry from WP2 v4 configs, and writes a
report that distinguishes found geometry from repaired or missing data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
HELPER_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = HELPER_DIR / "print_config.yaml"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




def _status(message: str) -> None:
    print(f"[3d-print] {message}", flush=True)

def _require(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency '{module_name}'. Run this helper inside the AISCO project environment."
        ) from exc


@dataclass
class ExportReport:
    run_dir: str
    out_dir: str
    assembly_mesh: str | None
    source_mesh: str | None = None
    scale_factor: float | None = None
    solar_region: str = "missing: no real saved solar mask/selection artifact found"
    vision_region: str = "not attempted"
    assembly_overlap: str = "not attempted"
    wheels: str = "not attempted"
    seat_support: str = "not attempted"
    removable_shell: str = "not attempted"
    outputs: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class ConfigError(ValueError):
    pass


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "''", '""'}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml_mapping(config_path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ConfigError(f"Cannot parse config line: {raw_line}")
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _load_simple_yaml_mapping(config_path)
    with config_path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping: {config_path}")
    return data


def _as_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(str(value)).expanduser()


def _prompt_text(label: str, default: Any = None, required: bool = False) -> str | None:
    default_text = f" [{default}]" if default not in (None, "") else ""
    while True:
        value = input(f"{label}{default_text}: ").strip()
        if value:
            return value
        if default not in (None, ""):
            return str(default)
        if not required:
            return None
        print("This value is required.")


def _prompt_int(label: str, default: int | None = None, required: bool = False) -> int | None:
    while True:
        value = _prompt_text(label, default, required)
        if value in (None, ""):
            return None
        try:
            return int(value)
        except ValueError:
            print("Please enter an integer.")


def _prompt_float(label: str, default: float) -> float:
    while True:
        value = _prompt_text(label, default, required=True)
        try:
            return float(value)
        except (TypeError, ValueError):
            print("Please enter a number.")


def _prompt_bool(label: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        value = input(f"{label} [{suffix}]: ").strip().lower()
        if not value:
            return default
        if value in {"y", "yes", "true", "1"}:
            return True
        if value in {"n", "no", "false", "0"}:
            return False
        print("Please answer yes or no.")


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)

    run_dir = args.run_dir or _as_path(cfg.get("run_dir"))
    out_dir = args.out_dir or _as_path(cfg.get("out_dir"))
    assembly_mesh_dir = args.assembly_mesh_dir or _as_path(cfg.get("assembly_mesh_dir"))
    assembly_slot = args.assembly_slot if args.assembly_slot is not None else cfg.get("assembly_slot")
    shell_clearance_mm = args.shell_clearance_mm if args.shell_clearance_mm is not None else cfg.get("shell_clearance_mm", 0.35)
    scale_factor = args.scale_factor if args.scale_factor is not None else cfg.get("scale_factor", 150.0 / 5800.0)
    epoch = args.epoch if args.epoch is not None else cfg.get("epoch")
    solar_source = args.solar_source or cfg.get("solar_source", "selection")
    solar_dir = args.solar_dir or _as_path(cfg.get("solar_dir"))
    search_parent_for_solar = cfg.get("search_parent_for_solar", False)
    prompt_for_missing = bool(cfg.get("prompt_for_missing", True))
    confirm_before_export = bool(cfg.get("confirm_before_export", True))

    if assembly_slot in ("", "auto", None):
        assembly_slot = None
    else:
        assembly_slot = int(assembly_slot)
    if epoch in ("", "latest", "last", "auto", None):
        epoch = None
    else:
        epoch = int(epoch)

    if prompt_for_missing:
        if run_dir is None:
            value = _prompt_text("Run/export folder", cfg.get("run_dir"), required=True)
            run_dir = Path(value).expanduser() if value else None
        if out_dir is None:
            default_out = run_dir / "printables_3d" if run_dir is not None else None
            value = _prompt_text("Output folder", default_out, required=False)
            out_dir = Path(value).expanduser() if value else default_out
        assembly_mesh_dir_text = _prompt_text("Assembly mesh folder", assembly_mesh_dir, required=False)
        assembly_mesh_dir = Path(assembly_mesh_dir_text).expanduser() if assembly_mesh_dir_text else assembly_mesh_dir
        assembly_slot = _prompt_int("Assembly slot (blank = first available)", assembly_slot, required=False)
        epoch = _prompt_int("Epoch number (blank = latest)", epoch, required=False)
        shell_clearance_mm = _prompt_float("Shell socket clearance in mm", float(shell_clearance_mm))
        scale_factor = _prompt_float("Global print scale factor", float(scale_factor))
        solar_source = _prompt_text("Solar source folder", solar_source, required=True) or "selection"
        solar_dir_text = _prompt_text("Exact solar folder (blank = search inside run_dir)", solar_dir, required=False)
        solar_dir = Path(solar_dir_text).expanduser() if solar_dir_text else solar_dir
        search_parent_for_solar = _prompt_bool("Search parent folder for solar outputs", bool(search_parent_for_solar))

    if run_dir is None:
        raise SystemExit(f"No run_dir provided. Add it to {config_path} or enter it when prompted.")
    if out_dir is None:
        out_dir = run_dir / "printables_3d"

    settings = {
        "config_path": config_path,
        "run_dir": run_dir,
        "out_dir": out_dir,
        "assembly_mesh_dir": assembly_mesh_dir,
        "assembly_slot": assembly_slot,
        "epoch": epoch,
        "shell_clearance_mm": float(shell_clearance_mm),
        "scale_factor": float(scale_factor),
        "solar_source": str(solar_source),
        "solar_dir": solar_dir,
        "search_parent_for_solar": bool(search_parent_for_solar),
    }

    print("\n3D printable export settings")
    print(f"  config: {settings['config_path']}")
    print(f"  run_dir: {settings['run_dir']}")
    print(f"  out_dir: {settings['out_dir']}")
    print(f"  assembly_mesh_dir: {settings['assembly_mesh_dir'] if settings['assembly_mesh_dir'] is not None else 'run_dir/mesh_assembly_v4'}")
    print(f"  assembly_slot: {settings['assembly_slot'] if settings['assembly_slot'] is not None else 'first available'}")
    print(f"  epoch: {settings['epoch'] if settings['epoch'] is not None else 'latest'}")
    print(f"  shell_clearance_mm: {settings['shell_clearance_mm']}")
    print(f"  scale_factor: {settings['scale_factor']} (~{settings['scale_factor'] * 5800.0:.1f} mm for a 5800 mm car length)")
    print(f"  solar_source: solar/{settings['solar_source']}")
    print(f"  solar_dir: {settings['solar_dir'] if settings['solar_dir'] is not None else 'inside run_dir'}")
    print(f"  search_parent_for_solar: {settings['search_parent_for_solar']}")

    if confirm_before_export and not _prompt_bool("Run export with these settings?", True):
        raise SystemExit("Cancelled.")

    return settings


def _epoch_score(path: Path) -> tuple[int, int, str]:
    text = str(path)
    nums = [int(n) for n in re.findall(r"epoch[_-]?(\d+)", text, flags=re.I)]
    epoch = max(nums) if nums else -1
    preferred = 2 if "adaptive_mesh" in text else 1 if "mesh_shape" in text else 0
    return preferred, epoch, text


def _path_epoch(path: Path) -> int | None:
    nums = [int(n) for n in re.findall(r"epoch[_-]?(\d+)", str(path), flags=re.I)]
    return max(nums) if nums else None


def _path_slot(path: Path) -> int | None:
    nums = [int(n) for n in re.findall(r"slot[_-]?(\d+)", str(path), flags=re.I)]
    return max(nums) if nums else None


def find_final_mesh(run_dir: Path, epoch: int | None = None) -> Path | None:
    roots = [run_dir / "meshes", run_dir]
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        iterator = root.glob("*") if root == run_dir else root.rglob("*")
        candidates.extend(
            p
            for p in iterator
            if p.is_file()
            and p.suffix.lower() in {".ply", ".glb", ".stl", ".obj"}
            and "adaptive_mesh" in p.name.lower()
            and "printables_3d" not in str(p)
        )
    if epoch is not None:
        candidates = [p for p in candidates if _path_epoch(p) == epoch]
    if not candidates:
        return None
    return sorted(set(candidates), key=_epoch_score)[-1]


def _solar_search_roots(run_dir: Path, include_parent: bool, solar_dir: Path | None = None) -> list[Path]:
    roots = [solar_dir] if solar_dir is not None else [run_dir / "solar"]
    if include_parent and solar_dir is None:
        roots.append(run_dir.parent / "solar")
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root is None or not root.is_dir():
            continue
        resolved = root.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def find_solar_artifacts(
    run_dir: Path,
    solar_source: str,
    include_parent: bool = False,
    solar_dir: Path | None = None,
) -> list[Path]:
    roots = _solar_search_roots(run_dir, include_parent, solar_dir)
    source_roots: list[Path] = []
    for root in roots:
        source_root = root if root.name == solar_source else root / solar_source
        if source_root.is_dir():
            source_roots.append(source_root)
    results: list[Path] = []
    for root in source_roots:
        results.extend(p for p in root.glob("epoch_*.ply") if p.is_file())
        results.extend(p for p in root.glob("epoch_*/slot_*_mesh.ply") if p.is_file())
        if not results:
            results.extend(p for p in root.glob("epoch_*/slot_*_mesh.glb") if p.is_file())
    return sorted(set(results), key=_epoch_score)


def find_solar_mesh(
    run_dir: Path,
    solar_source: str,
    epoch: int | None,
    include_parent: bool = False,
    solar_dir: Path | None = None,
    slot: int | None = None,
) -> Path | None:
    roots = _solar_search_roots(run_dir, include_parent, solar_dir)
    source_roots: list[Path] = []
    for root in roots:
        source_root = root if root.name == solar_source else root / solar_source
        if source_root.is_dir():
            source_roots.append(source_root)
    candidates: list[Path] = []
    for root in source_roots:
        search_roots = [root]
        if epoch is not None:
            epoch_dir = root / f"epoch_{epoch:05d}"
            if epoch_dir.is_dir():
                search_roots = [epoch_dir]
        for search_root in search_roots:
            if search_root.is_dir() and search_root.name.startswith("epoch_"):
                if slot is not None:
                    candidates.extend(search_root.glob(f"slot_{slot:02d}_mesh.ply"))
                    candidates.extend(search_root.glob(f"slot_{slot:02d}_mesh.glb"))
                else:
                    candidates.extend(search_root.glob("slot_*_mesh.ply"))
                    if not candidates:
                        candidates.extend(search_root.glob("slot_*_mesh.glb"))
            else:
                candidates.extend(search_root.glob("epoch_*.ply"))
    candidates = [p for p in candidates if p.is_file()]
    if epoch is not None:
        candidates = [p for p in candidates if _path_epoch(p) == epoch]
    if slot is not None:
        slot_filtered = [p for p in candidates if _path_slot(p) == slot]
        candidates = slot_filtered or candidates
    if not candidates:
        return None
    return sorted(candidates, key=_epoch_score)[-1]


def find_assembly_mesh(
    assembly_mesh_dir: Path | None,
    run_dir: Path,
    epoch: int | None = None,
    slot: int | None = None,
) -> Path | None:
    roots = [assembly_mesh_dir] if assembly_mesh_dir is not None else [run_dir / "mesh_assembly_v4"]
    candidates: list[Path] = []
    for root in roots:
        if root is None or not root.exists():
            continue
        search_roots = [root]
        if epoch is not None:
            epoch_dir = root / f"epoch_{epoch:05d}"
            if epoch_dir.exists():
                search_roots = [epoch_dir]
        for search_root in search_roots:
            candidates.extend(
                p
                for p in search_root.glob("slot_*_mesh.ply")
                if p.is_file() and "printables_3d" not in str(p)
            )
            if not candidates:
                candidates.extend(
                    p
                    for p in search_root.glob("slot_*_mesh.glb")
                    if p.is_file() and "printables_3d" not in str(p)
                )
    if epoch is not None:
        candidates = [p for p in candidates if _path_epoch(p) == epoch]
    if slot is not None:
        candidates = [p for p in candidates if _path_slot(p) == slot]
    if not candidates:
        return None
    def score(path: Path) -> tuple[int, int, int, str]:
        name = str(path).lower()
        ext_score = 2 if path.suffix.lower() == ".ply" else 1
        vision_score = 0 if "with_vision_cone" in name else 1
        epoch_score = _path_epoch(path) if _path_epoch(path) is not None else -1
        slot_score = -(_path_slot(path) if _path_slot(path) is not None else 9999)
        return ext_score + vision_score, epoch_score, slot_score, str(path)
    return sorted(candidates, key=score)[-1]


def _component_bounds(mesh: Any) -> list[tuple[Any, Any, Any]]:
    """Return connected components with bounds, largest first."""
    components = []
    for part in mesh.split(only_watertight=False):
        if getattr(part, "is_empty", False):
            continue
        bounds = part.bounds
        extents = bounds[1] - bounds[0]
        components.append((part, bounds, extents))
    if not components:
        components.append((mesh, mesh.bounds, mesh.bounds[1] - mesh.bounds[0]))
    components.sort(key=lambda item: float(item[0].area), reverse=True)
    return components


def _infer_wheel_centers_from_mesh(assembly: Any) -> list[Any]:
    """Infer wheel centers from wheel-like connected components in an existing export."""
    np = _require("numpy")
    candidates: list[Any] = []
    for _part, bounds, extents in _component_bounds(assembly):
        x_extent, y_extent, z_extent = (float(v) for v in extents)
        # The exported wheels are bottom-cut and often split into open surfaces,
        # so their thin axis can be either the 96 mm width or a zero-width face.
        wheel_diameter_like = 350.0 <= x_extent <= 650.0 and 250.0 <= z_extent <= 650.0
        wheel_width_like = 0.0 <= y_extent <= 180.0
        near_floor = float(bounds[0][2]) <= 150.0
        if wheel_diameter_like and wheel_width_like and near_floor:
            center = (bounds[0] + bounds[1]) / 2.0
            candidates.append(np.array([center[0], center[1], 557.0 / 2.0], dtype=float))

    merged: list[Any] = []
    for center in candidates:
        if all(float(np.linalg.norm(center[:2] - existing[:2])) > 120.0 for existing in merged):
            merged.append(center)
    return merged


def _make_full_wheel_meshes(wheel_centers: list[Any]) -> Any | None:
    if not wheel_centers:
        return None
    np = _require("numpy")
    trimesh = _require("trimesh")
    wheels = []
    radius = 557.0 / 2.0
    height = 96.0
    rotate_to_y = trimesh.transformations.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])
    for center in wheel_centers:
        wheel = trimesh.creation.cylinder(radius=radius, height=height, sections=72)
        wheel.apply_transform(rotate_to_y)
        wheel.apply_translation(center)
        wheels.append(wheel)
    return trimesh.util.concatenate(wheels)


def _make_box_mesh(center: Any, extents: Any) -> Any:
    trimesh = _require("trimesh")
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def _mesh_based_supports(assembly: Any) -> tuple[Any, Any, Any]:
    """Build a base plate and a simple seat support from existing assembly bounds."""
    np = _require("numpy")
    trimesh = _require("trimesh")
    bounds = assembly.bounds
    margin_xy = np.array([140.0, 140.0])
    plate_thickness = 8.0
    plate_z_center = float(bounds[0][2]) - plate_thickness / 2.0 - 4.0
    plate_extents = np.array([
        max(float(bounds[1][0] - bounds[0][0]) + 2.0 * margin_xy[0], 220.0),
        max(float(bounds[1][1] - bounds[0][1]) + 2.0 * margin_xy[1], 220.0),
        plate_thickness,
    ])
    plate_center = np.array([
        float((bounds[0][0] + bounds[1][0]) / 2.0),
        float((bounds[0][1] + bounds[1][1]) / 2.0),
        plate_z_center,
    ])
    plate = _make_box_mesh(plate_center, plate_extents)

    components = _component_bounds(assembly)
    seat_like = None
    for part, comp_bounds, extents in components:
        ordered = sorted(float(v) for v in extents)
        wheel_like = 40.0 <= ordered[0] <= 180.0 and ordered[1] >= 250.0 and ordered[2] >= 350.0 and float(comp_bounds[0][2]) <= 80.0
        if not wheel_like:
            if seat_like is None or float(comp_bounds[1][2]) > float(seat_like[1][1][2]):
                seat_like = (part, comp_bounds, extents)
    if seat_like is None:
        seat_like = components[0]

    _part, seat_bounds, _extents = seat_like
    support_top = float(seat_bounds[0][2])
    support_bottom = float(bounds[0][2])
    support_height = max(support_top - support_bottom, 20.0)
    support_center = np.array([
        float((seat_bounds[0][0] + seat_bounds[1][0]) / 2.0),
        float((seat_bounds[0][1] + seat_bounds[1][1]) / 2.0),
        support_bottom + support_height / 2.0,
    ])
    support_extents = np.array([180.0, 180.0, support_height])
    support = _make_box_mesh(support_center, support_extents)
    plate_with_support = trimesh.util.concatenate([plate, support])
    return support, plate, plate_with_support


def _mesh_socket_cutters(assembly: Any, wheel_centers: list[Any], clearance_mm: float) -> Any:
    trimesh = _require("trimesh")
    cutters = []
    radius = (18.0 + 2.0 * float(clearance_mm)) / 2.0
    height = 220.0
    for center in wheel_centers:
        cutter = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
        cutter.apply_translation([float(center[0]), float(center[1]), float(assembly.bounds[0][2]) + height / 2.0])
        cutters.append(cutter)
    bounds = assembly.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    central = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    central.apply_translation([float(center[0]), float(center[1]), float(bounds[0][2]) + height / 2.0])
    cutters.append(central)
    return trimesh.util.concatenate(cutters)

def _load_mesh(path: Path) -> Any:
    trimesh = _require("trimesh")
    obj = trimesh.load(str(path), force="mesh", process=False)
    if getattr(obj, "is_empty", False):
        raise ValueError(f"Mesh is empty: {path}")
    return obj



def _scaled_mesh(mesh: Any, scale_factor: float) -> Any:
    scaled = mesh.copy()
    scaled.apply_scale(float(scale_factor))
    return scaled


def _export_mesh(mesh: Any, path: Path, scale_factor: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scaled = _scaled_mesh(mesh, scale_factor)
    scaled.export(str(path))

def _colored(mesh: Any, color: tuple[int, int, int, int]) -> Any:
    copy = mesh.copy()
    copy.visual.vertex_colors = color
    return copy


def _submesh_by_container(mesh: Any, container: Any) -> tuple[Any | None, str]:
    face_centers = mesh.triangles_center
    try:
        inside = container.contains(face_centers)
    except BaseException as exc:  # trimesh may need rtree/embree/manifold
        return None, f"skipped: containment query unavailable ({type(exc).__name__}: {exc})"
    face_ids = [int(i) for i, ok in enumerate(inside) if bool(ok)]
    if not face_ids:
        return None, "computed: no intersecting faces found"
    return mesh.submesh([face_ids], append=True, repair=False), f"computed: {len(face_ids)} faces"


def _try_boolean_difference(mesh: Any, cutters: Any) -> tuple[Any | None, str]:
    try:
        result = mesh.difference([cutters], check_volume=False)
    except BaseException as exc:
        return None, f"skipped: boolean difference unavailable ({type(exc).__name__}: {exc})"
    if result is None or getattr(result, "is_empty", False):
        return None, "skipped: boolean returned an empty result"
    return result, "computed"


def export_printables(
    run_dir: Path,
    out_dir: Path,
    assembly_mesh_dir: Path | None,
    assembly_slot: int | None,
    shell_clearance_mm: float,
    scale_factor: float,
    epoch: int | None = None,
    solar_source: str = "selection",
    search_parent_for_solar: bool = False,
    solar_dir: Path | None = None,
) -> ExportReport:
    _status("Starting printable export")
    trimesh = _require("trimesh")
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _status(f"Run dir: {run_dir}")
    _status(f"Output dir: {out_dir}")

    _status(f"Selecting source mesh for epoch: {epoch if epoch is not None else 'latest'}")
    final_mesh_path = find_final_mesh(run_dir, epoch=epoch)
    if final_mesh_path is None:
        wanted = "latest" if epoch is None else f"epoch {epoch}"
        raise SystemExit(f"No {wanted} mesh file found under {run_dir}")
    selected_epoch = epoch if epoch is not None else _path_epoch(final_mesh_path)
    _status(f"Selected epoch for matching exports: {selected_epoch if selected_epoch is not None else 'unknown/latest'}")

    _status("Finding existing assembly mesh export")
    assembly_mesh = find_assembly_mesh(assembly_mesh_dir, run_dir, epoch=selected_epoch, slot=assembly_slot)
    if assembly_mesh is None and selected_epoch is not None:
        _status("No assembly mesh matched the selected epoch; falling back to latest assembly mesh")
        assembly_mesh = find_assembly_mesh(assembly_mesh_dir, run_dir, epoch=None, slot=assembly_slot)
    if assembly_mesh is None:
        raise SystemExit("No assembly mesh found. Set assembly_mesh_dir and optionally assembly_slot in print_config.yaml.")
    report = ExportReport(str(run_dir), str(out_dir), str(assembly_mesh), scale_factor=scale_factor)
    report.source_mesh = str(final_mesh_path)
    _status(f"Source mesh: {final_mesh_path}")
    _status(f"Assembly mesh: {assembly_mesh}")

    _status(f"Looking for real solar/{solar_source} mesh for selected epoch/slot")

    _status("Loading source mesh")
    body = _load_mesh(final_mesh_path)
    body_colored = _colored(body, (185, 190, 188, 255))

    _status("Building/repairing assembly, wheels, seat support, plate, and shell socket cutters")
    outputs = {
        "inner_assembly_full_wheels": out_dir / "inner_assembly_full_wheels.stl",
        "wheel_repair_full_wheels": out_dir / "wheel_repair_full_wheels.stl",
        "seat_support": out_dir / "seat_support.stl",
        "support_plate": out_dir / "support_plate.stl",
        "plate_with_seat_support": out_dir / "plate_with_seat_support.stl",
        "shell_socket_cutters": out_dir / "shell_socket_cutters.stl",
    }

    _status("Using existing assembly mesh and inferring wheel positions from geometry")
    exported_assembly = _load_mesh(assembly_mesh)
    wheel_centers = _infer_wheel_centers_from_mesh(exported_assembly)
    full_wheels_mesh = _make_full_wheel_meshes(wheel_centers)
    seat_support_mesh, plate_mesh, plate_with_support_mesh = _mesh_based_supports(exported_assembly)
    cutters_mesh = _mesh_socket_cutters(exported_assembly, wheel_centers, shell_clearance_mm)
    inner_mesh = exported_assembly
    if full_wheels_mesh is not None:
        inner_mesh = trimesh.util.concatenate([exported_assembly, full_wheels_mesh])
        _export_mesh(full_wheels_mesh, outputs["wheel_repair_full_wheels"], scale_factor)
    else:
        report.warnings.append("No wheel-like components were inferred from assembly mesh; wheel repair was skipped.")
    _export_mesh(inner_mesh, outputs["inner_assembly_full_wheels"], scale_factor)
    _export_mesh(seat_support_mesh, outputs["seat_support"], scale_factor)
    _export_mesh(plate_mesh, outputs["support_plate"], scale_factor)
    _export_mesh(plate_with_support_mesh, outputs["plate_with_seat_support"], scale_factor)
    _export_mesh(cutters_mesh, outputs["shell_socket_cutters"], scale_factor)
    report.wheels = f"inferred {len(wheel_centers)} wheel center(s) from assembly mesh and added full regulation wheels"
    report.seat_support = "generated from existing assembly mesh bounds"

    _status("Skipping generated vision-cone extraction; assembly mesh is the source of truth")
    vision_part = None
    report.vision_region = "skipped: assembly mesh is the source of truth; use existing *_with_vision_cone mesh if needed"
    _status(f"Vision region: {report.vision_region}")
    _status("Attempting assembly-overlap face extraction")
    overlap_part, report.assembly_overlap = _submesh_by_container(body, inner_mesh)
    _status(f"Assembly overlap: {report.assembly_overlap}")
    solar_mesh_path = find_solar_mesh(run_dir, solar_source, selected_epoch, search_parent_for_solar, solar_dir, assembly_slot)
    solar_mesh = None
    if solar_mesh_path is not None:
        _status(f"Including real solar/{solar_source} mesh: {solar_mesh_path}")
        solar_mesh = _load_mesh(solar_mesh_path)
        report.solar_region = f"included real solar/{solar_source} mesh for selected epoch: {solar_mesh_path}"
    else:
        _status(f"No real solar/{solar_source} mesh found for selected epoch")

    _status("Creating multicolor GLB/PLY scene")
    scene = trimesh.Scene()
    scene.add_geometry(_scaled_mesh(body_colored, scale_factor), node_name="final_aeroshell")
    scene.add_geometry(_scaled_mesh(_colored(inner_mesh, (245, 145, 35, 255)), scale_factor), node_name="full_inner_assembly")
    if solar_mesh is not None:
        scene.add_geometry(_scaled_mesh(_colored(solar_mesh, (245, 205, 35, 255)), scale_factor), node_name="real_solar_artifact")
    if vision_part is not None:
        scene.add_geometry(_scaled_mesh(_colored(vision_part, (230, 40, 35, 255)), scale_factor), node_name="vision_cone_intersection")
    if overlap_part is not None:
        scene.add_geometry(_scaled_mesh(_colored(overlap_part, (130, 50, 220, 255)), scale_factor), node_name="assembly_overlap")

    glb_path = out_dir / "final_multicolor_real.glb"
    _status(f"Writing {glb_path.name}")
    scene.export(str(glb_path))
    report.outputs["final_multicolor_real_glb"] = str(glb_path)

    merged = trimesh.util.concatenate([body_colored, _colored(inner_mesh, (245, 145, 35, 255))])
    if solar_mesh is not None:
        merged = trimesh.util.concatenate([merged, _colored(solar_mesh, (245, 205, 35, 255))])
    if vision_part is not None:
        merged = trimesh.util.concatenate([merged, _colored(vision_part, (230, 40, 35, 255))])
        vision_path = out_dir / "vision_cone_intersection.stl"
        _export_mesh(vision_part, vision_path, scale_factor)
        report.outputs["vision_cone_intersection"] = str(vision_path)
    if overlap_part is not None:
        merged = trimesh.util.concatenate([merged, _colored(overlap_part, (130, 50, 220, 255))])
        overlap_path = out_dir / "assembly_overlap.stl"
        _export_mesh(overlap_part, overlap_path, scale_factor)
        report.outputs["assembly_overlap"] = str(overlap_path)
    ply_path = out_dir / "final_multicolor_real.ply"
    _status(f"Writing {ply_path.name}")
    _export_mesh(merged, ply_path, scale_factor)
    report.outputs["final_multicolor_real_ply"] = str(ply_path)

    shell_source = out_dir / "removable_shell_source.stl"
    _status(f"Writing {shell_source.name}")
    _export_mesh(body, shell_source, scale_factor)
    report.outputs["removable_shell_source"] = str(shell_source)

    _status("Attempting removable-shell socket boolean")
    socketed, shell_msg = _try_boolean_difference(body, cutters_mesh)
    _status(f"Socket boolean: {shell_msg}")
    report.removable_shell = shell_msg
    if socketed is not None:
        socketed_path = out_dir / "removable_shell_with_sockets.stl"
        _export_mesh(socketed, socketed_path, scale_factor)
        report.outputs["removable_shell_with_sockets"] = str(socketed_path)
    else:
        report.warnings.append("Socket cutters were exported separately because mesh boolean subtraction did not complete.")

    report.outputs.update({k: str(v) for k, v in outputs.items()})

    report_path = out_dir / "print_report.json"
    report.outputs["print_report"] = str(report_path)
    _status(f"Writing report: {report_path}")
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    _status("Printable export finished")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Override config run_dir.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override config out_dir.")
    parser.add_argument("--assembly-mesh-dir", type=Path, default=None, help="Folder containing mesh_assembly_v4 exports.")
    parser.add_argument("--assembly-slot", type=int, default=None, help="Assembly slot number to use, e.g. 0 for slot_00_mesh.ply.")
    parser.add_argument("--epoch", type=int, default=None, help="Override config epoch. Omit for latest.")
    parser.add_argument("--shell-clearance-mm", type=float, default=None, help="Override config shell_clearance_mm.")
    parser.add_argument("--scale-factor", type=float, default=None, help="Override config scale_factor.")
    parser.add_argument("--solar-source", type=str, default=None, help="Solar subfolder to include, e.g. selection, quality, penalty.")
    parser.add_argument("--solar-dir", type=Path, default=None, help="Exact solar folder to search, e.g. .../run/solar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    report = export_printables(
        settings["run_dir"],
        settings["out_dir"],
        settings["assembly_mesh_dir"],
        settings["assembly_slot"],
        settings["shell_clearance_mm"],
        settings["scale_factor"],
        settings["epoch"],
        settings["solar_source"],
        settings["search_parent_for_solar"],
        settings["solar_dir"],
    )
    print("\nExport complete")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()

