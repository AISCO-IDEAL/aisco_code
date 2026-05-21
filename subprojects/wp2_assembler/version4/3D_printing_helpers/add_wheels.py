"""Add full wheels to an existing exported assembly mesh.

The input is a .ply or .glb mesh whose wheels are already present but cut at the
bottom. The script infers the wheel positions from that existing geometry, adds
full wheel cylinders at those positions, and saves the result next to the input
file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _status(message: str) -> None:
    print(f"[add-wheels] {message}", flush=True)


def _require(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise SystemExit(f"Missing dependency '{module_name}'. Run this helper inside the AISCO project environment.") from exc


def _load_mesh(path: Path) -> Any:
    trimesh = _require("trimesh")
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if getattr(mesh, "is_empty", False):
        raise SystemExit(f"Mesh is empty: {path}")
    return mesh


def _component_bounds(mesh: Any) -> list[tuple[Any, Any, Any]]:
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


def _is_cut_wheel_component(bounds: Any, extents: Any) -> bool:
    x_extent, y_extent, z_extent = (float(v) for v in extents)
    wheel_diameter_like = 350.0 <= x_extent <= 650.0 and 250.0 <= z_extent <= 650.0
    wheel_width_like = 0.0 <= y_extent <= 180.0
    near_floor = float(bounds[0][2]) <= 150.0
    return wheel_diameter_like and wheel_width_like and near_floor


def infer_wheel_centers(mesh: Any) -> list[Any]:
    """Infer wheel centers from bottom-cut wheel surfaces in an exported mesh."""
    np = _require("numpy")
    radius = 557.0 / 2.0
    groups: list[list[Any]] = []
    for _part, bounds, extents in _component_bounds(mesh):
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
    return [
        np.array([float(np.mean(group, axis=0)[0]), float(np.mean(group, axis=0)[1]), radius], dtype=float) for group in groups
    ]


def _is_component_in_wheel_footprint(bounds: Any, wheel_centers: list[Any]) -> bool:
    if not wheel_centers:
        return False
    np = _require("numpy")
    center = (bounds[0] + bounds[1]) / 2.0
    near_floor = float(bounds[0][2]) <= 160.0 and float(bounds[1][2]) <= 650.0
    if not near_floor:
        return False
    return any(float(np.linalg.norm(center[:2] - wheel_center[:2])) <= 310.0 for wheel_center in wheel_centers)


def remove_cut_wheel_components(mesh: Any, wheel_centers: list[Any]) -> Any:
    trimesh = _require("trimesh")
    kept = [
        part
        for part, bounds, extents in _component_bounds(mesh)
        if not (_is_cut_wheel_component(bounds, extents) or _is_component_in_wheel_footprint(bounds, wheel_centers))
    ]
    if not kept:
        return mesh.copy()
    return trimesh.util.concatenate(kept)


def make_full_wheels(wheel_centers: list[Any]) -> Any | None:
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


def default_output_path(input_path: Path, keep_original_wheels: bool) -> Path:
    suffix = "_with_original_and_full_wheels" if keep_original_wheels else "_full_wheels"
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def add_wheels(input_path: Path, output_path: Path | None = None, keep_original_wheels: bool = False) -> Path:
    trimesh = _require("trimesh")
    input_path = input_path.expanduser().resolve()
    if input_path.suffix.lower() not in {".ply", ".glb"}:
        raise SystemExit("Input must be a .ply or .glb mesh.")
    if not input_path.exists():
        raise SystemExit(f"Input mesh not found: {input_path}")
    output_path = output_path.expanduser().resolve() if output_path else default_output_path(input_path, keep_original_wheels)

    _status(f"Input: {input_path}")
    mesh = _load_mesh(input_path)
    wheel_centers = infer_wheel_centers(mesh)
    _status(f"Inferred {len(wheel_centers)} wheel center(s)")
    if not wheel_centers:
        raise SystemExit("No wheel-like components found. Output was not written.")

    wheels = make_full_wheels(wheel_centers)
    if keep_original_wheels:
        _status("Test mode: keeping original cut wheels and adding full wheels")
        base_mesh = mesh
    else:
        _status("Replacing original cut wheels with full wheels")
        base_mesh = remove_cut_wheel_components(mesh, wheel_centers)
    repaired = trimesh.util.concatenate([base_mesh, wheels])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(str(output_path))
    _status(f"Wrote: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Input .ply or .glb assembly mesh.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path. Default: beside input with a wheel suffix.")
    parser.add_argument(
        "--keep-original-wheels",
        action="store_true",
        help="Test mode: keep the original cut wheel pieces and add full wheels for alignment comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if input_path is None:
        value = input("Input .ply/.glb mesh: ").strip()
        if not value:
            raise SystemExit("No input mesh provided.")
        input_path = Path(value)
    add_wheels(input_path, args.out, args.keep_original_wheels)


if __name__ == "__main__":
    main()
