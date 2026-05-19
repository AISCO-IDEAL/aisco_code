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
        raise SystemExit(
            f"Missing dependency '{module_name}'. Run this helper inside the AISCO project environment."
        ) from exc


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


def infer_wheel_centers(mesh: Any) -> list[Any]:
    """Infer wheel centers from bottom-cut wheel surfaces in an exported mesh."""
    np = _require("numpy")
    candidates: list[Any] = []
    for _part, bounds, extents in _component_bounds(mesh):
        x_extent, y_extent, z_extent = (float(v) for v in extents)
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


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_full_wheels{input_path.suffix}")


def add_wheels(input_path: Path, output_path: Path | None = None) -> Path:
    trimesh = _require("trimesh")
    input_path = input_path.expanduser().resolve()
    if input_path.suffix.lower() not in {".ply", ".glb"}:
        raise SystemExit("Input must be a .ply or .glb mesh.")
    if not input_path.exists():
        raise SystemExit(f"Input mesh not found: {input_path}")
    output_path = output_path.expanduser().resolve() if output_path else default_output_path(input_path)

    _status(f"Input: {input_path}")
    mesh = _load_mesh(input_path)
    wheel_centers = infer_wheel_centers(mesh)
    _status(f"Inferred {len(wheel_centers)} wheel center(s)")
    if not wheel_centers:
        raise SystemExit("No wheel-like components found. Output was not written.")

    wheels = make_full_wheels(wheel_centers)
    repaired = trimesh.util.concatenate([mesh, wheels])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.export(str(output_path))
    _status(f"Wrote: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="Input .ply or .glb assembly mesh.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output path. Default: beside input with _full_wheels suffix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input
    if input_path is None:
        value = input("Input .ply/.glb mesh: ").strip()
        if not value:
            raise SystemExit("No input mesh provided.")
        input_path = Path(value)
    add_wheels(input_path, args.out)


if __name__ == "__main__":
    main()
