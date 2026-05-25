"""Create 3D-printing artifacts from existing AISCO exports.

This is intentionally a post-processor. It reads meshes from a run folder,
reconstructs deterministic repair geometry from exported WP2 v4 meshes, and
writes a report that distinguishes found geometry from repaired or missing data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Edit these paths when running this file directly.
INPUT_DIR = Path("/cluster/work/fuge/AISCO/beast_sync_wp2/1017_273570_desert-wave")
OUTPUT_DIR = INPUT_DIR / "printables_3d"

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


def _as_optional_int(value: Any) -> int | None:
    if value in ("", "auto", None):
        return None
    return int(value)


def _as_optional_epoch(value: Any) -> int | None:
    if value in ("", "latest", "last", "auto", None):
        return None
    return int(value)


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else run_dir / "printables_3d"
    assembly_mesh_dir = args.assembly_mesh_dir.expanduser() if args.assembly_mesh_dir is not None else None
    assembly_slot = _as_optional_int(args.assembly_slot)
    epoch = _as_optional_epoch(args.epoch)
    shell_clearance_mm = float(args.shell_clearance_mm)
    scale_factor = float(args.scale_factor)
    solar_source = str(args.solar_source)
    solar_dir = args.solar_dir.expanduser() if args.solar_dir is not None else None
    search_parent_for_solar = bool(args.search_parent_for_solar)
    keep_original_wheels = bool(args.keep_original_wheels)
    rollout_only = bool(args.rollout_only)

    settings = {
        "run_dir": run_dir,
        "out_dir": out_dir,
        "assembly_mesh_dir": assembly_mesh_dir,
        "assembly_slot": assembly_slot,
        "epoch": epoch,
        "shell_clearance_mm": shell_clearance_mm,
        "scale_factor": scale_factor,
        "solar_source": solar_source,
        "solar_dir": solar_dir,
        "search_parent_for_solar": search_parent_for_solar,
        "keep_original_wheels": keep_original_wheels,
        "rollout_only": rollout_only,
    }

    print("\n3D printable export settings")
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
    print(f"  keep_original_wheels: {settings['keep_original_wheels']}")
    print(f"  rollout_only: {settings['rollout_only']}")

    return settings


def _epoch_score(path: Path) -> tuple[int, int, str]:
    text = str(path)
    nums = [int(n) for n in re.findall(r"epoch[_-]?(\d+)", text, flags=re.IGNORECASE)]
    epoch = max(nums) if nums else -1
    preferred = 3 if "mesh_shape" in text else 2 if "adaptive_mesh" in text else 1
    return preferred, epoch, text


def _path_epoch(path: Path) -> int | None:
    nums = [int(n) for n in re.findall(r"epoch[_-]?(\d+)", str(path), flags=re.IGNORECASE)]
    return max(nums) if nums else None


def _path_slot(path: Path) -> int | None:
    nums = [int(n) for n in re.findall(r"slot[_-]?(\d+)", str(path), flags=re.IGNORECASE)]
    return max(nums) if nums else None


def find_final_mesh(run_dir: Path, epoch: int | None = None) -> Path | None:
    roots = [run_dir / "mesh_shape", run_dir / "meshes", run_dir]
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
            and (
                "adaptive_mesh" in p.name.lower()
                or root.name == "mesh_shape"
            )
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


def find_solar_mesh(  # noqa: PLR0912
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
                for p in search_root.glob("epoch_*.ply")
                if p.is_file() and "printables_3d" not in str(p) and "with_vision_cone" not in p.name.lower()
            )
            candidates.extend(
                p
                for p in search_root.glob("epoch_*.glb")
                if p.is_file() and "printables_3d" not in str(p) and "with_vision_cone" not in p.name.lower()
            )
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
        path_epoch = _path_epoch(path)
        path_slot = _path_slot(path)
        epoch_score = path_epoch if path_epoch is not None else -1
        slot_score = -(path_slot if path_slot is not None else 9999)
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




def _inner_assembly_components_only(assembly: Any) -> Any:
    """Drop shell-sized and bounding-box-sized components from an assembly export."""
    trimesh = _require("trimesh")
    kept = []
    for part, _bounds, extents in _component_bounds(assembly):
        max_extent = max(float(v) for v in extents)
        nonzero_extents = [float(v) for v in extents if float(v) > 1.0]
        broad_flat_sheet = len(nonzero_extents) <= 2 and max_extent > 1800.0
        shell_sized = max_extent > 1800.0
        if shell_sized or broad_flat_sheet:
            continue
        kept.append(part)
    if not kept:
        return assembly.copy()
    return trimesh.util.concatenate(kept)


def _open3d_occupancy(mesh: Any, points: Any, chunk_size: int = 100_000) -> tuple[Any | None, str | None]:
    np = _require("numpy")
    try:
        o3d = __import__("open3d")
    except ImportError as exc:
        return None, f"open3d unavailable ({type(exc).__name__}: {exc})"

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    points = np.asarray(points, dtype=np.float32)
    if len(vertices) == 0 or len(faces) == 0 or len(points) == 0:
        return None, "empty mesh or query points"

    o3d_mesh = o3d.t.geometry.TriangleMesh()
    o3d_mesh.vertex.positions = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
    o3d_mesh.triangle.indices = o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d_mesh)

    inside = np.empty(len(points), dtype=bool)
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        query = o3d.core.Tensor(points[start:end], dtype=o3d.core.Dtype.Float32)
        inside[start:end] = scene.compute_occupancy(query).numpy().astype(bool)
    return inside, None


def _filter_inner_assembly_outside_body(inner_mesh: Any, body: Any) -> tuple[Any, str]:
    """Keep only whole inner-assembly components with no sampled points inside the body."""
    np = _require("numpy")
    trimesh = _require("trimesh")
    kept = []
    removed_components = 0
    removed_faces = 0
    total_components = 0

    for part, _bounds, _extents in _component_bounds(inner_mesh):
        total_components += 1
        vertices = np.asarray(part.vertices, dtype=float)
        centers = np.asarray(part.triangles_center, dtype=float)
        query_points = np.vstack([vertices, centers]) if len(centers) else vertices
        if len(query_points) > 5000:
            step = max(1, len(query_points) // 5000)
            query_points = query_points[::step]

        inside, error = _open3d_occupancy(body, query_points)
        if inside is None:
            return inner_mesh, f"skipped: body occupancy unavailable ({error}); kept original inner assembly"

        if bool(np.any(inside)):
            removed_components += 1
            removed_faces += len(part.faces)
            continue
        kept.append(part)

    if removed_components == 0:
        return inner_mesh, f"computed: no inner components intersected the body across {total_components} component(s)"
    if not kept:
        return inner_mesh, f"skipped: all {total_components} inner components tested inside; kept original inner assembly"

    filtered = trimesh.util.concatenate(kept)
    return filtered, (
        f"computed: removed {removed_components}/{total_components} inner component(s) with any sampled point inside the body "
        f"({removed_faces} face(s)); kept only fully-outside components whole"
    )


def _boolean_inner_assembly_outside_body(inner_mesh: Any, body: Any) -> tuple[Any, str]:
    """Boolean subtract the main body from the inner assembly, with coarse fallback."""
    boolean_inner, inner_repair_msg = _repair_mesh_for_boolean(inner_mesh)
    boolean_body, body_repair_msg = _repair_mesh_for_boolean(body)
    cut_inner, boolean_msg = _try_boolean_difference(boolean_inner, boolean_body)
    if cut_inner is not None:
        return cut_inner, f"boolean inner minus body: {inner_repair_msg}; body {body_repair_msg}; {boolean_msg}"

    fallback_inner, fallback_msg = _filter_inner_assembly_outside_body(inner_mesh, body)
    return fallback_inner, f"boolean inner minus body failed ({boolean_msg}); fallback: {fallback_msg}"


def _cleanup_multicolor_dir(multicolor_dir: Path) -> None:
    for stale_name in (
        "final_multicolor_real.glb",
        "final_multicolor_real.ply",
        "multicolor.3mf",
        "assembly_overlap.stl",
        "mesh.stl",
        "inner_assembly.stl",
        "solar_placement.stl",
        "vision_cone_intersection.stl",
    ):
        stale_path = multicolor_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


def _cleanup_single_color_dir(single_color_dir: Path) -> None:
    for stale_name in (
        "plate_with_support.stl",
        "inner_assembly.stl",
        "inner_assembly_only.stl",
        "mesh_with_inner_assembly.stl",
        "mesh_with_inner_assembly.3mf",
        "unmodified_mesh_flat_bottom.stl",
        "single_color_unmodified_flat_bottom.3mf",
        "mesh_with_cutouts.stl",
        "unmodified_mesh.stl",
        "unmodified_inner_assembly.stl",
        "single_print.3mf",
        "single_print_unmodified.3mf",
        "removable_shell_source.stl",
        "removable_shell_with_sockets.stl",
        "inner_assembly_full_wheels.stl",
        "wheel_repair_full_wheels.stl",
        "seat_support.stl",
        "support_plate.stl",
        "plate_with_seat_support.stl",
        "shell_socket_cutters.stl",
    ):
        stale_path = single_color_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

def _is_cut_wheel_component(bounds: Any, extents: Any) -> bool:
    x_extent, y_extent, z_extent = (float(v) for v in extents)
    wheel_diameter_like = 350.0 <= x_extent <= 650.0 and 250.0 <= z_extent <= 650.0
    wheel_width_like = 0.0 <= y_extent <= 180.0
    near_floor = float(bounds[0][2]) <= 150.0
    return wheel_diameter_like and wheel_width_like and near_floor


def _infer_wheel_centers_from_mesh(assembly: Any) -> list[Any]:
    """Infer wheel centers from wheel-like connected components in an existing export."""
    np = _require("numpy")
    radius = 557.0 / 2.0
    groups: list[list[Any]] = []
    for _part, bounds, extents in _component_bounds(assembly):
        if not _is_cut_wheel_component(bounds, extents):
            continue
        center = (bounds[0] + bounds[1]) / 2.0
        candidate = np.array([center[0], center[1], radius], dtype=float)
        for group in groups:
            group_center = np.mean(group, axis=0)
            if float(np.linalg.norm(candidate[:2] - group_center[:2])) <= 160.0:
                group.append(candidate)
                break
        else:
            groups.append([candidate])
    return [np.array([float(np.mean(group, axis=0)[0]), float(np.mean(group, axis=0)[1]), radius], dtype=float) for group in groups]


def _is_component_in_wheel_footprint(bounds: Any, wheel_centers: list[Any]) -> bool:
    if not wheel_centers:
        return False
    np = _require("numpy")
    center = (bounds[0] + bounds[1]) / 2.0
    near_floor = float(bounds[0][2]) <= 160.0 and float(bounds[1][2]) <= 650.0
    if not near_floor:
        return False
    return any(float(np.linalg.norm(center[:2] - wheel_center[:2])) <= 310.0 for wheel_center in wheel_centers)


def _remove_cut_wheel_components(mesh: Any, wheel_centers: list[Any]) -> Any:
    trimesh = _require("trimesh")
    kept = [
        part
        for part, bounds, extents in _component_bounds(mesh)
        if not (_is_cut_wheel_component(bounds, extents) or _is_component_in_wheel_footprint(bounds, wheel_centers))
    ]
    if not kept:
        return mesh.copy()
    return trimesh.util.concatenate(kept)


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


def _make_cylinder_between(start: Any, end: Any, radius: float) -> Any:
    np = _require("numpy")
    trimesh = _require("trimesh")
    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    vector = end_point - start_point
    height = float(np.linalg.norm(vector))
    if height <= 1e-9:
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=1)
        sphere.apply_translation(start_point)
        return sphere
    cylinder = trimesh.creation.cylinder(radius=float(radius), height=height, sections=18)
    transform = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], vector / height)
    cylinder.apply_transform(transform)
    cylinder.apply_translation((start_point + end_point) / 2.0)
    return cylinder


def _union_plate_with_support(plate: Any, support: Any) -> Any:
    trimesh = _require("trimesh")
    fallback = trimesh.util.concatenate([plate, support])
    try:
        union_parts = [plate, *support.split(only_watertight=False)]
        unioned = trimesh.boolean.union(union_parts, check_volume=False)
    except BaseException:
        return fallback
    if unioned is None or getattr(unioned, "is_empty", False):
        return fallback
    return unioned


def _mesh_based_supports(assembly: Any, wheel_centers: list[Any]) -> tuple[Any, Any, Any]:
    """Build a wheel-ground plate and a light support frame below the seat."""
    np = _require("numpy")
    trimesh = _require("trimesh")
    bounds = assembly.bounds
    wheel_radius = 557.0 / 2.0
    ground_z = min((float(center[2]) - wheel_radius for center in wheel_centers), default=float(bounds[0][2]))
    plate_thickness = 90.0
    plate_margin = np.array([220.0, 180.0])

    if wheel_centers:
        wheel_xy = np.asarray([[float(center[0]), float(center[1])] for center in wheel_centers], dtype=float)
        min_xy = wheel_xy.min(axis=0) - np.array([wheel_radius, 96.0 / 2.0]) - plate_margin
        max_xy = wheel_xy.max(axis=0) + np.array([wheel_radius, 96.0 / 2.0]) + plate_margin
        assembly_min_xy = np.asarray(bounds[0][:2], dtype=float) - np.array([80.0, 80.0])
        assembly_max_xy = np.asarray(bounds[1][:2], dtype=float) + np.array([80.0, 80.0])
        min_xy = np.minimum(min_xy, assembly_min_xy)
        max_xy = np.maximum(max_xy, assembly_max_xy)
    else:
        min_xy = np.asarray(bounds[0][:2], dtype=float) - plate_margin
        max_xy = np.asarray(bounds[1][:2], dtype=float) + plate_margin

    plate_center = np.array([
        float((min_xy[0] + max_xy[0]) / 2.0),
        float((min_xy[1] + max_xy[1]) / 2.0),
        ground_z - plate_thickness / 2.0,
    ])
    plate_extents = np.array([
        max(float(max_xy[0] - min_xy[0]), 600.0),
        max(float(max_xy[1] - min_xy[1]), 420.0),
        plate_thickness,
    ])
    plate = _make_box_mesh(plate_center, plate_extents)

    components = _component_bounds(assembly)
    seat_like = None
    for part, comp_bounds, extents in components:
        ordered = sorted(float(v) for v in extents)
        wheel_like = 40.0 <= ordered[0] <= 180.0 and ordered[1] >= 250.0 and ordered[2] >= 350.0 and float(comp_bounds[0][2]) <= 160.0
        if (
            not wheel_like
            and float(comp_bounds[0][2]) > ground_z + 260.0
            and (seat_like is None or float(comp_bounds[1][2]) > float(seat_like[1][1][2]))
        ):
            seat_like = (part, comp_bounds, extents)
    if seat_like is None:
        seat_like = max(components, key=lambda item: float(item[1][1][2]))

    _part, seat_bounds, _extents = seat_like
    seat_min = np.asarray(seat_bounds[0], dtype=float)
    seat_max = np.asarray(seat_bounds[1], dtype=float)
    frame_radius = 42.0
    seat_overlap = frame_radius * 0.35
    support_bottom = ground_z - seat_overlap
    support_top = max(float(seat_min[2]) + seat_overlap, ground_z + 80.0)
    top_points = [
        np.array([seat_min[0], seat_min[1], support_top]),
        np.array([seat_max[0], seat_min[1], support_top]),
        np.array([seat_max[0], seat_max[1], support_top]),
        np.array([seat_min[0], seat_max[1], support_top]),
    ]
    bottom_points = [point.copy() for point in top_points]
    for point in bottom_points:
        point[2] = support_bottom

    beams = []
    for bottom, top in zip(bottom_points, top_points, strict=True):
        beams.append(_make_cylinder_between(bottom, top, frame_radius))
    for points in (top_points, bottom_points):
        for start, end in zip(points, points[1:] + points[:1], strict=True):
            beams.append(_make_cylinder_between(start, end, frame_radius * 0.75))
    beams.extend([
        _make_cylinder_between(bottom_points[0], top_points[2], frame_radius * 0.65),
        _make_cylinder_between(bottom_points[1], top_points[3], frame_radius * 0.65),
        _make_cylinder_between(bottom_points[2], top_points[0], frame_radius * 0.65),
        _make_cylinder_between(bottom_points[3], top_points[1], frame_radius * 0.65),
    ])
    support = trimesh.util.concatenate(beams)
    plate_with_support = _union_plate_with_support(plate, support)
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


def _box_from_bounds(bounds: Any, clearance_xy_mm: float, clearance_z_mm: float) -> Any:
    np = _require("numpy")
    min_corner = np.array(bounds[0], dtype=float) - np.array([clearance_xy_mm, clearance_xy_mm, clearance_z_mm])
    max_corner = np.array(bounds[1], dtype=float) + np.array([clearance_xy_mm, clearance_xy_mm, clearance_z_mm])
    center = (min_corner + max_corner) / 2.0
    extents = np.maximum(max_corner - min_corner, np.array([1.0, 1.0, 1.0]))
    return _make_box_mesh(center, extents)


def _single_print_shell_cutters(body: Any, inner_assembly: Any, plate_with_support: Any, clearance_mm: float) -> Any:
    trimesh = _require("trimesh")
    cutters = []
    clearance_xy = float(clearance_mm)
    clearance_z = float(clearance_mm)

    for mesh in (inner_assembly, plate_with_support):
        for _part, bounds, _extents in _component_bounds(mesh):
            cutters.append(_box_from_bounds(bounds, clearance_xy, clearance_z))

    plate_like_bounds = None
    plate_like_area = -1.0
    for _part, bounds, extents in _component_bounds(plate_with_support):
        height = float(extents[2])
        area = float(extents[0]) * float(extents[1])
        if height <= 160.0 and area > plate_like_area:
            plate_like_bounds = bounds
            plate_like_area = area
    plate_bounds = plate_like_bounds if plate_like_bounds is not None else plate_with_support.bounds
    body_bounds = body.bounds
    bottom_bounds = [
        [float(plate_bounds[0][0]), float(plate_bounds[0][1]), float(body_bounds[0][2]) - clearance_z],
        [float(plate_bounds[1][0]), float(plate_bounds[1][1]), float(plate_bounds[1][2]) + clearance_z],
    ]
    cutters.append(_box_from_bounds(bottom_bounds, clearance_xy, 0.0))
    return trimesh.util.concatenate(cutters)


def _flatten_mesh_below(mesh: Any, cut_z: float) -> tuple[Any, str]:
    np = _require("numpy")
    trimesh = _require("trimesh")
    vertices = np.array(mesh.vertices, dtype=float)
    below = vertices[:, 2] < float(cut_z)
    if not bool(np.any(below)):
        return mesh.copy(), f"skipped: no vertices below z={float(cut_z):.3f}"

    try:
        sliced = mesh.slice_plane(
            plane_origin=[0.0, 0.0, float(cut_z)],
            plane_normal=[0.0, 0.0, 1.0],
            cap=True,
        )
        if sliced is not None and not getattr(sliced, "is_empty", False):
            sliced.remove_degenerate_faces()
            sliced.remove_duplicate_faces()
            sliced.remove_unreferenced_vertices()
            trimesh.repair.fix_normals(sliced)
            return sliced, f"computed: capped plane cut at z={float(cut_z):.3f}"
    except BaseException as exc:
        slice_error = f"{type(exc).__name__}: {exc}"
    else:
        slice_error = "slice returned empty mesh"

    flattened = mesh.copy()
    vertices[below, 2] = float(cut_z)
    flattened.vertices = vertices
    flattened.remove_degenerate_faces()
    flattened.remove_duplicate_faces()
    flattened.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(flattened)
    return (
        flattened,
        f"fallback: capped plane cut unavailable ({slice_error}); clamped {int(below.sum())} vertex/vertices below z={float(cut_z):.3f}",
    )


def _cut_mesh_faces_inside_boxes(mesh: Any, cutters: Any) -> tuple[Any | None, str]:
    np = _require("numpy")
    face_centers = np.asarray(mesh.triangles_center, dtype=float)
    remove = np.zeros(len(face_centers), dtype=bool)
    box_count = 0
    for _part, bounds, _extents in _component_bounds(cutters):
        box_count += 1
        lower = np.asarray(bounds[0], dtype=float)
        upper = np.asarray(bounds[1], dtype=float)
        inside = np.all((face_centers >= lower) & (face_centers <= upper), axis=1)
        remove |= inside
    kept_ids = np.flatnonzero(~remove)
    removed_faces = int(remove.sum())
    if removed_faces == 0:
        return None, f"skipped: no shell faces inside {box_count} cutout boxes"
    cut_mesh = mesh.submesh([kept_ids.tolist()], append=True, repair=False)
    if cut_mesh is None or getattr(cut_mesh, "is_empty", False):
        return None, f"skipped: face cut removed all shell faces using {box_count} cutout boxes"
    return cut_mesh, f"computed by face-cut fallback: removed {removed_faces} faces using {box_count} cutout boxes"

def _edge_count_by_pair(faces: Any) -> tuple[Any, Any]:
    np = _require("numpy")
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)


def _boundary_loops(boundary_edges: Any) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {}
    for start, end in boundary_edges:
        start_i = int(start)
        end_i = int(end)
        adjacency.setdefault(start_i, []).append(end_i)
        adjacency.setdefault(end_i, []).append(start_i)

    loops: list[list[int]] = []
    used_edges: set[tuple[int, int]] = set()
    for raw_start, raw_end in boundary_edges:
        start = int(raw_start)
        end = int(raw_end)
        first_edge = (start, end) if start < end else (end, start)
        if first_edge in used_edges:
            continue

        previous = start
        current = end
        loop = [start, current]
        used_edges.add(first_edge)
        for _unused in range(len(boundary_edges) + 5):
            if current == start:
                break
            next_vertex = None
            for candidate in adjacency[current]:
                if candidate == previous:
                    continue
                candidate_edge = (current, candidate) if current < candidate else (candidate, current)
                if candidate_edge not in used_edges or candidate == start:
                    next_vertex = candidate
                    break
            if next_vertex is None:
                break
            next_edge = (current, next_vertex) if current < next_vertex else (next_vertex, current)
            if next_edge in used_edges:
                break
            used_edges.add(next_edge)
            previous = current
            current = next_vertex
            if current != start:
                loop.append(current)
        loops.append(loop)
    return loops


def _repair_mesh_for_boolean(mesh: Any) -> tuple[Any, str]:
    np = _require("numpy")
    trimesh = _require("trimesh")
    repaired = mesh.copy()
    repaired.merge_vertices()
    trimesh.repair.fix_normals(repaired)

    faces = np.asarray(repaired.faces, dtype=np.int64)
    vertices = np.asarray(repaired.vertices, dtype=float)
    unique_edges, edge_counts = _edge_count_by_pair(faces)
    nonmanifold_edges = {tuple(edge) for edge in unique_edges[edge_counts > 2]}

    removed_faces = 0
    if nonmanifold_edges:
        remove = np.zeros(len(faces), dtype=bool)
        for face_index, (a, b, c) in enumerate(faces):
            for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
                edge = (start, end) if start < end else (end, start)
                if edge in nonmanifold_edges:
                    remove[face_index] = True
                    break
        removed_faces = int(remove.sum())
        faces = faces[~remove]

    unique_edges, edge_counts = _edge_count_by_pair(faces)
    boundary_edges = unique_edges[edge_counts == 1]
    loops = _boundary_loops(boundary_edges) if len(boundary_edges) else []

    out_vertices = vertices.tolist()
    out_faces = faces.tolist()
    capped_loops = 0
    for loop in loops:
        if len(loop) < 3:
            continue
        center_index = len(out_vertices)
        out_vertices.append(vertices[loop].mean(axis=0).tolist())
        for index, start in enumerate(loop):
            out_faces.append([int(start), int(loop[(index + 1) % len(loop)]), center_index])
        capped_loops += 1

    repaired = trimesh.Trimesh(vertices=np.asarray(out_vertices), faces=np.asarray(out_faces), process=True)
    repaired.merge_vertices()
    trimesh.repair.fix_normals(repaired)
    status = (
        f"surface repair removed {removed_faces} nonmanifold face(s) and capped {capped_loops} boundary loop(s); "
        f"watertight={bool(repaired.is_watertight)}"
    )
    return repaired, status



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


def _face_mask_from_vertex_colors(mesh: Any, mode: str) -> Any | None:
    np = _require("numpy")
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is None or len(colors) == 0:
        return None
    face_rgb = np.asarray(colors, dtype=float)[mesh.faces, :3].mean(axis=1)
    if mode == "green":
        mask = face_rgb[:, 1] > face_rgb[:, 0]
    elif mode == "red":
        mask = (face_rgb[:, 0] > face_rgb[:, 1]) & (face_rgb[:, 0] > face_rgb[:, 2])
    elif mode == "dark":
        mask = face_rgb.mean(axis=1) < 96.0
    else:
        raise ValueError(f"Unknown color mask mode: {mode}")
    return mask


def _submesh_by_face_mask(mesh: Any, mask: Any) -> Any | None:
    face_ids = [int(i) for i, ok in enumerate(mask) if bool(ok)]
    if not face_ids:
        return None
    return mesh.submesh([face_ids], append=True, repair=False)


def _raised_surface_region(surface: Any, height_mm: float) -> Any:
    np = _require("numpy")
    trimesh = _require("trimesh")
    vertices = np.asarray(surface.vertices, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    normals = np.array(surface.vertex_normals, dtype=float)
    normal_lengths = np.linalg.norm(normals, axis=1)
    bad_normals = normal_lengths <= 1e-12
    if bool(np.any(bad_normals)):
        center = vertices.mean(axis=0)
        fallback = vertices - center
        fallback_lengths = np.linalg.norm(fallback, axis=1)
        fallback_lengths[fallback_lengths <= 1e-12] = 1.0
        fallback = fallback / fallback_lengths[:, None]
        normals[bad_normals] = fallback[bad_normals]
        normal_lengths = np.linalg.norm(normals, axis=1)
    normals = normals / normal_lengths[:, None]

    n_vertices = len(vertices)
    bottom = vertices.copy()
    top = vertices + normals * float(height_mm)
    out_vertices = np.vstack([bottom, top])
    out_faces: list[list[int]] = []

    for a, b, c in faces:
        out_faces.append([int(a + n_vertices), int(b + n_vertices), int(c + n_vertices)])
        out_faces.append([int(c), int(b), int(a)])

    edge_counts: dict[tuple[int, int], int] = {}
    edge_dirs: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, c in faces:
        for start, end in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            key = (start, end) if start < end else (end, start)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            edge_dirs[key] = (start, end)

    for key, count in edge_counts.items():
        if count != 1:
            continue
        start, end = edge_dirs[key]
        start_top = start + n_vertices
        end_top = end + n_vertices
        out_faces.append([start, end, end_top])
        out_faces.append([start, end_top, start_top])

    return trimesh.Trimesh(vertices=out_vertices, faces=np.asarray(out_faces, dtype=np.int64), process=False)


def _extract_raised_color_region(mesh: Any, mode: str, height_mm: float) -> tuple[Any | None, str]:
    mask = _face_mask_from_vertex_colors(mesh, mode)
    if mask is None:
        return None, "skipped: source mesh has no vertex colors to decode"
    selected_faces = int(mask.sum())
    if selected_faces == 0:
        return None, "computed: no selected faces found"
    surface = _submesh_by_face_mask(mesh, mask)
    if surface is None:
        return None, "computed: no selected faces found"
    raised = _raised_surface_region(surface, height_mm)
    return raised, f"computed: {selected_faces} selected faces embossed by {height_mm:g} model mm"


def _mesh_to_3mf_object(mesh: Any, object_id: int, material_index: int, name: str) -> str:
    vertices = _scaled_mesh(mesh, 1.0).vertices
    faces = _scaled_mesh(mesh, 1.0).faces
    lines = [f'    <object id="{object_id}" type="model" name="{escape(name)}" pid="1" pindex="{material_index}">']
    lines.append("      <mesh>")
    lines.append("        <vertices>")
    for x, y, z in vertices:
        lines.append(f'          <vertex x="{float(x):.9g}" y="{float(y):.9g}" z="{float(z):.9g}" />')
    lines.append("        </vertices>")
    lines.append("        <triangles>")
    for v1, v2, v3 in faces:
        lines.append(f'          <triangle v1="{int(v1)}" v2="{int(v2)}" v3="{int(v3)}" />')
    lines.append("        </triangles>")
    lines.append("      </mesh>")
    lines.append("    </object>")
    return "\n".join(lines)


def _export_multicolor_3mf(parts: list[tuple[str, Any, str]], path: Path, scale_factor: float) -> None:
    scaled_parts = [(name, _scaled_mesh(mesh, scale_factor), color) for name, mesh, color in parts]
    material_lines = [
        f'      <base name="{escape(name)}" displaycolor="{color}" />'
        for name, _mesh, color in scaled_parts
    ]
    object_lines = [
        _mesh_to_3mf_object(mesh, object_id=index + 1, material_index=index, name=name)
        for index, (name, mesh, _color) in enumerate(scaled_parts)
    ]
    build_lines = [f'    <item objectid="{index + 1}" />' for index in range(len(scaled_parts))]
    model = "\n".join([
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
        '  <metadata name="Title">AISCO multicolor printable</metadata>',
        '  <resources>',
        '    <basematerials id="1">',
        *material_lines,
        '    </basematerials>',
        *object_lines,
        '  </resources>',
        '  <build>',
        *build_lines,
        '  </build>',
        '</model>',
        "",
    ])
    content_types = '''<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>
'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("3D/3dmodel.model", model)

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
        cutter_parts = [part for part in cutters.split(only_watertight=False) if not getattr(part, "is_empty", False)]
    except BaseException:
        cutter_parts = [cutters]
    if not cutter_parts:
        return None, "skipped: no boolean cutters were available"
    try:
        result = mesh.difference(cutter_parts, check_volume=False)
    except BaseException as exc:
        return None, f"skipped: boolean difference unavailable ({type(exc).__name__}: {exc})"
    if result is None or getattr(result, "is_empty", False):
        return None, "skipped: boolean returned an empty result"
    return result, f"computed with {len(cutter_parts)} cutter solid(s); watertight={bool(result.is_watertight)}"


def export_printables(  # noqa: PLR0912, PLR0915
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
    keep_original_wheels: bool = False,
    rollout_only: bool = False,
) -> ExportReport:
    _status("Starting printable export")
    trimesh = _require("trimesh")
    run_dir = run_dir.resolve()
    out_dir = out_dir.resolve()
    multicolor_dir = out_dir / "multicolor"
    single_color_dir = out_dir / "single_color"
    rollout_live_prints_dir = out_dir / "rollout_live_prints"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rollout_only:
        multicolor_dir.mkdir(parents=True, exist_ok=True)
        single_color_dir.mkdir(parents=True, exist_ok=True)
    rollout_live_prints_dir.mkdir(parents=True, exist_ok=True)
    _status(f"Run dir: {run_dir}")
    _status(f"Output dir: {out_dir}")
    if not rollout_only:
        _status(f"Multicolor dir: {multicolor_dir}")
        _status(f"Single-color dir: {single_color_dir}")
    _status(f"Rollout live prints dir: {rollout_live_prints_dir}")
    if not rollout_only:
        _cleanup_multicolor_dir(multicolor_dir)
        _cleanup_single_color_dir(single_color_dir)
    _cleanup_single_color_dir(rollout_live_prints_dir)

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
    selected_slot = assembly_slot if assembly_slot is not None else _path_slot(assembly_mesh)
    _status(f"Source mesh: {final_mesh_path}")
    _status(f"Assembly mesh: {assembly_mesh}")
    _status(f"Selected slot for matching masks: {selected_slot if selected_slot is not None else 'first available'}")

    if not rollout_only:
        _status(f"Looking for real solar/{solar_source} mesh and solar/vision_cone_mask for selected epoch/slot")

    _status("Loading source mesh")
    body = _load_mesh(final_mesh_path)

    if rollout_only:
        _status("Building rollout live print only")
    else:
        _status("Building/repairing assembly, wheels, seat support, plate, and shell cutouts")
    output_name_prefix = run_dir.name
    single_outputs = {
        "single_plate_with_support": single_color_dir / "plate_with_support.stl",
        "single_inner_assembly": single_color_dir / "inner_assembly.stl",
        "single_mesh_with_inner_assembly": single_color_dir / "mesh_with_inner_assembly.stl",
        "single_mesh_with_inner_assembly_3mf": single_color_dir / "mesh_with_inner_assembly.3mf",
        "single_mesh_with_cutouts": single_color_dir / "mesh_with_cutouts.stl",
        "single_unmodified_mesh": single_color_dir / "unmodified_mesh.stl",
        "rollout_unmodified_mesh_flat_bottom": rollout_live_prints_dir / "unmodified_mesh_flat_bottom.stl",
        "single_unmodified_inner_assembly": single_color_dir / "unmodified_inner_assembly.stl",
        "single_print_3mf": single_color_dir / f"single_color_{output_name_prefix}.3mf",
        "single_print_unmodified_3mf": single_color_dir / f"single_color_unmodified_{output_name_prefix}.3mf",
        "rollout_unmodified_flat_bottom_3mf": rollout_live_prints_dir / "single_color_unmodified_flat_bottom.3mf",
    }

    _status("Using existing assembly mesh and inferring wheel positions from geometry")
    exported_assembly = _load_mesh(assembly_mesh)
    assembly_core = _inner_assembly_components_only(exported_assembly)
    wheel_centers = _infer_wheel_centers_from_mesh(exported_assembly)
    if keep_original_wheels:
        _status("Wheel test mode: keeping original cut wheels and adding full wheels")
        wheel_base_mesh = assembly_core
    else:
        _status("Replacing original cut wheels with full wheels")
        wheel_base_mesh = _remove_cut_wheel_components(assembly_core, wheel_centers)
    full_wheels_mesh = _make_full_wheel_meshes(wheel_centers)
    inner_mesh = wheel_base_mesh
    if full_wheels_mesh is not None:
        inner_mesh = trimesh.util.concatenate([wheel_base_mesh, full_wheels_mesh])
    else:
        report.warnings.append("No wheel-like components were inferred from assembly mesh; wheel repair was skipped.")
    unmodified_inner_assembly = inner_mesh.copy()
    rollout_scale_factor = float(scale_factor) * 0.4
    body_height = float(body.bounds[1][2] - body.bounds[0][2])
    flat_bottom_z = float(body.bounds[0][2] + 0.1 * body_height)
    flat_bottom_body, flat_bottom_msg = _flatten_mesh_below(body, flat_bottom_z)
    _status(f"Flat-bottom unmodified mesh: {flat_bottom_msg}")
    report.outputs["single_unmodified_flat_bottom_cut_z_model_mm"] = f"{flat_bottom_z:.6g}"
    report.outputs["rollout_live_prints_scale_factor"] = f"{rollout_scale_factor:.6g}"

    if rollout_only:
        report.wheels = f"inferred {len(wheel_centers)} wheel center(s) from assembly mesh and added full regulation wheels"
        report.seat_support = "skipped: rollout-only export"
        report.removable_shell = "skipped: rollout-only export"
        report.assembly_overlap = "skipped: rollout-only export"
        _status(f"Writing {single_outputs['rollout_unmodified_mesh_flat_bottom'].name}")
        _export_mesh(flat_bottom_body, single_outputs["rollout_unmodified_mesh_flat_bottom"], rollout_scale_factor)
        _status(f"Writing {single_outputs['rollout_unmodified_flat_bottom_3mf'].name}")
        _export_multicolor_3mf(
            [
                ("unmodified_mesh_flat_bottom", flat_bottom_body, "#D8D8D8"),
                ("unmodified_inner_assembly", unmodified_inner_assembly, "#D8D8D8"),
            ],
            single_outputs["rollout_unmodified_flat_bottom_3mf"],
            rollout_scale_factor,
        )
        report.outputs["rollout_unmodified_mesh_flat_bottom"] = str(single_outputs["rollout_unmodified_mesh_flat_bottom"])
        report.outputs["rollout_unmodified_flat_bottom_3mf"] = str(single_outputs["rollout_unmodified_flat_bottom_3mf"])
        report_path = out_dir / "print_report.json"
        report.outputs["print_report"] = str(report_path)
        _status(f"Writing report: {report_path}")
        report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
        _status("Rollout live print export finished")
        return report

    _seat_support_mesh, _plate_mesh, plate_with_support_mesh = _mesh_based_supports(wheel_base_mesh, wheel_centers)
    _status("Boolean-cutting inner assembly with the source mesh")
    inner_mesh, inner_filter_msg = _boolean_inner_assembly_outside_body(inner_mesh, body)
    _status(f"Inner assembly boolean: {inner_filter_msg}")
    report.wheels = f"inferred {len(wheel_centers)} wheel center(s) from assembly mesh and added full regulation wheels"
    report.seat_support = "generated as a wheel-ground plate with a beam frame under the raised seat"

    vision_part = None
    report.assembly_overlap = inner_filter_msg
    if "failed" in inner_filter_msg or inner_filter_msg.startswith("skipped"):
        report.warnings.append(f"Inner assembly boolean cut did not complete cleanly: {inner_filter_msg}")
    color_region_height_mm = 24.0
    solar_mesh_path = find_solar_mesh(run_dir, solar_source, selected_epoch, search_parent_for_solar, solar_dir, selected_slot)
    solar_mesh = None
    if solar_mesh_path is not None:
        _status(f"Decoding real solar/{solar_source} color mask: {solar_mesh_path}")
        solar_source_mesh = _load_mesh(solar_mesh_path)
        solar_mask_mode = "dark" if solar_source == "binary_mask" else "red"
        solar_mesh, solar_msg = _extract_raised_color_region(solar_source_mesh, solar_mask_mode, color_region_height_mm)
        _status(f"Solar color region: {solar_msg}")
        report.solar_region = (
            f"included real solar/{solar_source} selected faces for selected epoch/slot: {solar_mesh_path}; {solar_msg}"
        )
        if solar_mesh is None:
            report.warnings.append(f"Solar region was not exported because {solar_msg}")
    else:
        _status(f"No real solar/{solar_source} mesh found for selected epoch/slot")

    vision_mesh_path = find_solar_mesh(run_dir, "vision_cone_mask", selected_epoch, search_parent_for_solar, solar_dir, selected_slot)
    if vision_mesh_path is not None:
        _status(f"Decoding real solar/vision_cone_mask color mask: {vision_mesh_path}")
        vision_source_mesh = _load_mesh(vision_mesh_path)
        vision_part, vision_msg = _extract_raised_color_region(vision_source_mesh, "green", color_region_height_mm)
        _status(f"Vision color region: {vision_msg}")
        report.vision_region = (
            f"included real solar/vision_cone_mask selected faces for selected epoch/slot: {vision_mesh_path}; {vision_msg}"
        )
        if vision_part is None:
            report.warnings.append(f"Vision region was not exported because {vision_msg}")
    else:
        _status("No real solar/vision_cone_mask mesh found for selected epoch/slot")
        report.vision_region = "missing: no real solar/vision_cone_mask mesh found for selected epoch/slot"

    _status("Writing multicolor STL set, one file per material")
    mesh_path = multicolor_dir / "mesh.stl"
    _status(f"Writing {mesh_path.name}")
    _export_mesh(body, mesh_path, scale_factor)
    report.outputs["multicolor_mesh"] = str(mesh_path)

    assembly_path = multicolor_dir / "inner_assembly.stl"
    _status(f"Writing {assembly_path.name}")
    _export_mesh(inner_mesh, assembly_path, scale_factor)
    report.outputs["multicolor_inner_assembly"] = str(assembly_path)

    if solar_mesh is not None:
        solar_path = multicolor_dir / "solar_placement.stl"
        _status(f"Writing {solar_path.name}")
        _export_mesh(solar_mesh, solar_path, scale_factor)
        report.outputs["multicolor_solar_placement"] = str(solar_path)

    if vision_part is not None:
        vision_path = multicolor_dir / "vision_cone_intersection.stl"
        _status(f"Writing {vision_path.name}")
        _export_mesh(vision_part, vision_path, scale_factor)
        report.outputs["multicolor_vision_cone_intersection"] = str(vision_path)

    multicolor_parts = [
        ("body", body, "#D8D8D8"),
        ("inner_assembly", inner_mesh, "#303030"),
    ]
    if solar_mesh is not None:
        multicolor_parts.append(("solar_placement", solar_mesh, "#E53935"))
    if vision_part is not None:
        multicolor_parts.append(("vision_cone_intersection", vision_part, "#00A060"))
    multicolor_3mf_path = multicolor_dir / f"multicolor_{output_name_prefix}.3mf"
    _status(f"Writing {multicolor_3mf_path.name}")
    _export_multicolor_3mf(multicolor_parts, multicolor_3mf_path, scale_factor)
    report.outputs["multicolor_3mf"] = str(multicolor_3mf_path)

    _status("Writing single-print STL set")
    _status(f"Writing {single_outputs['single_plate_with_support'].name}")
    _export_mesh(plate_with_support_mesh, single_outputs["single_plate_with_support"], scale_factor)
    _status(f"Writing {single_outputs['single_inner_assembly'].name}")
    _export_mesh(inner_mesh, single_outputs["single_inner_assembly"], scale_factor)
    mesh_with_inner_assembly = trimesh.util.concatenate([body, inner_mesh])
    _status(f"Writing {single_outputs['single_mesh_with_inner_assembly'].name}")
    _export_mesh(mesh_with_inner_assembly, single_outputs["single_mesh_with_inner_assembly"], scale_factor)
    _status(f"Writing {single_outputs['single_mesh_with_inner_assembly_3mf'].name}")
    _export_multicolor_3mf(
        [("mesh", body, "#D8D8D8"), ("inner_assembly", inner_mesh, "#D8D8D8")],
        single_outputs["single_mesh_with_inner_assembly_3mf"],
        scale_factor,
    )
    _status(f"Writing {single_outputs['single_unmodified_mesh'].name}")
    _export_mesh(body, single_outputs["single_unmodified_mesh"], scale_factor)
    _status(f"Writing {single_outputs['rollout_unmodified_mesh_flat_bottom'].name}")
    _export_mesh(flat_bottom_body, single_outputs["rollout_unmodified_mesh_flat_bottom"], rollout_scale_factor)
    _status(f"Writing {single_outputs['single_unmodified_inner_assembly'].name}")
    _export_mesh(unmodified_inner_assembly, single_outputs["single_unmodified_inner_assembly"], scale_factor)

    print_clearance_mm = 0.6
    shell_cutout_clearance_mm = max(float(shell_clearance_mm), print_clearance_mm / float(scale_factor))
    _status(f"Building shell cutouts with {shell_cutout_clearance_mm:.3f} model mm clearance")
    support_cutout_mesh = trimesh.util.concatenate([_plate_mesh, _seat_support_mesh])
    shell_cutters = _single_print_shell_cutters(body, inner_mesh, support_cutout_mesh, shell_cutout_clearance_mm)
    boolean_body, repair_msg = _repair_mesh_for_boolean(body)
    _status(f"Shell source repair: {repair_msg}")
    _status("Attempting slide-over shell cutout boolean")
    cut_shell, shell_msg = _try_boolean_difference(boolean_body, shell_cutters)
    _status(f"Shell cutout boolean: {shell_msg}")
    report.removable_shell = f"{repair_msg}; {shell_msg}"
    if cut_shell is None:
        cut_shell, fallback_msg = _cut_mesh_faces_inside_boxes(body, shell_cutters)
        _status(f"Shell cutout fallback: {fallback_msg}")
        report.removable_shell = f"{shell_msg}; {fallback_msg}"
    if cut_shell is None:
        cut_shell = body
        report.warnings.append("Shell cutouts were not applied because boolean and face-cut fallback did not complete.")
    elif not bool(cut_shell.is_watertight):
        report.warnings.append("Shell mesh is not watertight; boolean result should be inspected before printing.")

    _status(f"Writing {single_outputs['single_mesh_with_cutouts'].name}")
    _export_mesh(cut_shell, single_outputs["single_mesh_with_cutouts"], scale_factor)

    _status(f"Writing {single_outputs['single_print_3mf'].name}")
    _export_multicolor_3mf(
        [
            ("plate_with_support", plate_with_support_mesh, "#8A8A8A"),
            ("inner_assembly", inner_mesh, "#303030"),
            ("mesh_with_cutouts", cut_shell, "#D8D8D8"),
        ],
        single_outputs["single_print_3mf"],
        scale_factor,
    )

    _status(f"Writing {single_outputs['single_print_unmodified_3mf'].name}")
    _export_multicolor_3mf(
        [
            ("unmodified_mesh", body, "#D8D8D8"),
            ("unmodified_inner_assembly", unmodified_inner_assembly, "#303030"),
        ],
        single_outputs["single_print_unmodified_3mf"],
        scale_factor,
    )

    _status(f"Writing {single_outputs['rollout_unmodified_flat_bottom_3mf'].name}")
    _export_multicolor_3mf(
        [
            ("unmodified_mesh_flat_bottom", flat_bottom_body, "#D8D8D8"),
            ("unmodified_inner_assembly", unmodified_inner_assembly, "#D8D8D8"),
        ],
        single_outputs["rollout_unmodified_flat_bottom_3mf"],
        rollout_scale_factor,
    )
    report.outputs.update({k: str(v) for k, v in single_outputs.items()})

    report_path = out_dir / "print_report.json"
    report.outputs["print_report"] = str(report_path)
    _status(f"Writing report: {report_path}")
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n")
    _status("Printable export finished")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=INPUT_DIR, help="Run/export folder to post-process. Default: INPUT_DIR.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR, help="Output folder. Default: OUTPUT_DIR.")
    parser.add_argument("--assembly-mesh-dir", type=Path, default=None, help="Folder containing mesh_assembly_v4 exports.")
    parser.add_argument("--assembly-slot", type=int, default=None, help="Assembly slot number to use, e.g. 0 for slot_00_mesh.ply.")
    parser.add_argument("--epoch", type=int, default=None, help="Mesh epoch to export. Omit for latest.")
    parser.add_argument("--shell-clearance-mm", type=float, default=0.35, help="Shell socket clearance in model millimeters.")
    parser.add_argument("--scale-factor", type=float, default=0.9 * 150.0 / 5800.0, help="Global print scale factor.")
    parser.add_argument("--solar-source", type=str, default="binary_mask", help="Solar subfolder to include, e.g. binary_mask, selection, quality, penalty.")
    parser.add_argument("--solar-dir", type=Path, default=None, help="Exact solar folder to search, e.g. .../run/solar.")
    parser.add_argument(
        "--search-parent-for-solar",
        action="store_true",
        help="Also search the parent folder for solar outputs.",
    )
    parser.add_argument(
        "--keep-original-wheels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Test mode: keep the original cut wheel pieces and add full wheels for alignment comparison.",
    )
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="Only export the scaled rollout live print STL/3MF; skip multicolor, supports, masks, and shell cutouts.",
    )
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
        settings["keep_original_wheels"],
        settings["rollout_only"],
    )
    print("\nExport complete")
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
