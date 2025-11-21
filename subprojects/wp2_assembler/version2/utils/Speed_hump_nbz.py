from __future__ import annotations

import contextlib
import importlib.util
import inspect
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import cadquery as cq
from cadquery import exporters

try:  # pragma: no cover - optional viewer dependency
    from ocp_vscode import Animation, show
except Exception:  # pragma: no cover - viewer missing during CI
    Animation = None
    show = None
    _ANIMATION_SUPPORTS_LEGACY_FRAMES = False
else:  # pragma: no cover - animation available
    try:
        _ANIMATION_SUPPORTS_LEGACY_FRAMES = "frames" in inspect.signature(
            Animation
        ).parameters
    except (ValueError, TypeError):
        _ANIMATION_SUPPORTS_LEGACY_FRAMES = False

try:  # pragma: no cover - optional physics dependency during CI
    import pybullet as p
    import pybullet_data
except Exception as exc:  # pragma: no cover - pybullet missing
    p = None
    pybullet_data = None
    _PYBULLET_IMPORT_ERROR = exc
else:  # pragma: no cover - pybullet present
    _PYBULLET_IMPORT_ERROR = None

_HERE = Path(__file__).resolve().parent


def _load_attr_from_local(module_filename: str, attr_name: str):
    module_path = _HERE / module_filename
    if not module_path.exists():
        raise ImportError(f"{module_filename} not found next to {__file__}")
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create loader for {module_filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attr_name)


try:  # prefer dedicated helper if available
    from .inner_nbz import build_car_geometry as _inner_nbz_builder
except Exception:  # pragma: no cover - fallback to deterministic builder
    try:
        _inner_nbz_builder = _load_attr_from_local("inner_nbz.py", "build_car_geometry")
    except Exception:
        _inner_nbz_builder = None

try:  # pragma: no cover - casing differs between environments
    from .speed_hump import create_speed_hump as _speed_hump_builder
except Exception:  # pragma: no cover - fallback to uppercase file
    _speed_hump_builder = None
    for candidate in ("speed_hump.py", "Speed_hump.py"):
        try:
            _speed_hump_builder = _load_attr_from_local(candidate, "create_speed_hump")
            break
        except Exception:
            continue
    if _speed_hump_builder is None:
        raise

MM_TO_M = 0.001
CAR_CLEARANCE_M = 0.02
DEFAULT_TOTAL_TRAVEL_M = 0.5
DEFAULT_STEP_MM = 1.0
SETTLE_STEPS = 240
CAR_VIEWER_COLOR = cq.Color(0.9, 0.2, 0.2)
HUMP_VIEWER_COLOR = cq.Color(0.2, 0.4, 0.9, 0.5)
HUMP_VIEWER_Z_OFFSET_MM = -150.0
HUMP_VIEWER_FRONT_GAP_MM = 50.0


@dataclass
class SimulationResult:
    frames: List[List[cq.Shape]]
    car_shape: cq.Shape
    hump_shape: cq.Shape
    car_positions_mm: List[Tuple[float, float, float]]
    car_orientations_quat: List[Tuple[float, float, float, float]]
    hump_positions_mm: List[Tuple[float, float, float]]

    def build_viewer_payload(
        self, time_step: float = 0.01
    ) -> Animation | tuple[cq.Assembly, Animation]:
        if Animation is None:
            raise RuntimeError("ocp_vscode Animation is not available")

        if _ANIMATION_SUPPORTS_LEGACY_FRAMES:
            return Animation(self.frames, time_step=time_step)

        if not self.frames:
            raise RuntimeError("No frames available to build viewer payload")

        initial_car, initial_hump = self.frames[0]
        car_front_mm = initial_car.BoundingBox().xmax
        hump_front_mm = initial_hump.BoundingBox().xmax
        hump_x_offset_mm = car_front_mm + HUMP_VIEWER_FRONT_GAP_MM - hump_front_mm

        initial_hump = initial_hump.translate(
            (hump_x_offset_mm, 0.0, HUMP_VIEWER_Z_OFFSET_MM)
        )
        assembly = cq.Assembly(name="SpeedHumpSim")
        assembly.add(initial_car, name="car", color=CAR_VIEWER_COLOR)
        assembly.add(initial_hump, name="hump", color=HUMP_VIEWER_COLOR)
        assembly.mates = []  # Attribute required for ocp_vscode Animation detection

        animation = Animation(assembly)
        total_frames = len(self.frames)
        times = [idx * time_step for idx in range(total_frames)]

        if not (
            len(self.car_positions_mm)
            == len(self.car_orientations_quat)
            == len(self.hump_positions_mm)
            == total_frames
        ):
            raise RuntimeError("Simulation data is incomplete for viewer animation")

        car_path = f"/{assembly.name}/car"
        hump_path = f"/{assembly.name}/hump"
        animation.add_track(car_path, "t", times, self.car_positions_mm)
        animation.add_track(car_path, "q", times, self.car_orientations_quat)
        hump_positions_visual = [
            (x + hump_x_offset_mm, y, z + HUMP_VIEWER_Z_OFFSET_MM)
            for x, y, z in self.hump_positions_mm
        ]
        animation.add_track(hump_path, "t", times, hump_positions_visual)
        return assembly, animation


def _resolve_car_builder() -> Callable[[], cq.Shape]:
    if _inner_nbz_builder is not None:
        return _inner_nbz_builder
    try:
        from subprojects.wp2_assembler.version2.components.nbz import inner_nbz2_sym
    except Exception as exc:  # pragma: no cover - module missing during CI
        raise RuntimeError(
            "Unable to import an inner NBZ builder. Ensure 'inner_nbz' is available."
        ) from exc
    return inner_nbz2_sym


def _extract_shape(obj: cq.Shape | cq.Workplane | cq.Assembly) -> cq.Shape:
    if isinstance(obj, cq.Assembly):
        return obj.toCompound()
    if isinstance(obj, cq.Workplane):
        val = obj.val()
        if isinstance(val, cq.Shape):
            return val
    if isinstance(obj, cq.Shape):
        return obj
    raise TypeError(f"Unsupported CadQuery object type: {type(obj)!r}")


def cq_solid_to_bullet_collision_shape(solid: cq.Shape, scale: float = MM_TO_M) -> int:
    if p is None:  # pragma: no cover - pybullet missing during CI
        raise RuntimeError("pybullet is not available") from _PYBULLET_IMPORT_ERROR
    mesh_solid = solid if math.isclose(scale, 1.0) else solid.scale(scale)
    tmp = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        exporters.export(mesh_solid, tmp_path, exporters.ExportTypes.STL)
        return p.createCollisionShape(p.GEOM_MESH, fileName=tmp_path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)


def _build_car_shape() -> cq.Shape:
    builder = _resolve_car_builder()
    return _extract_shape(builder())


def _build_speed_hump_shape() -> cq.Shape:
    return _extract_shape(_speed_hump_builder())


def _axis_angle_from_quaternion(quat: Sequence[float]) -> Tuple[Tuple[float, float, float], float]:
    x, y, z, w = quat
    angle = 2.0 * math.acos(max(min(w, 1.0), -1.0))
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return (1.0, 0.0, 0.0), 0.0
    return (x / s, y / s, z / s), angle


def transform_cq_solid_with_bullet_pose(
    solid: cq.Shape, pos_m: Sequence[float], orn: Sequence[float]
) -> cq.Shape:
    pos_mm = tuple(coord / MM_TO_M for coord in pos_m)
    axis, angle = _axis_angle_from_quaternion(orn)
    transformed = solid
    if angle > 1e-6:
        transformed = transformed.rotate((0.0, 0.0, 0.0), axis, math.degrees(angle))
    return transformed.translate(pos_mm)


def _setup_world(car_shape: cq.Shape, hump_shape: cq.Shape):
    if p is None:  # pragma: no cover - pybullet missing during CI
        raise RuntimeError("pybullet is not available") from _PYBULLET_IMPORT_ERROR

    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0.0, 0.0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=120)

    plane_shape = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0.0, plane_shape)

    car_bbox = car_shape.BoundingBox()
    hump_bbox = hump_shape.BoundingBox()

    car_collision = cq_solid_to_bullet_collision_shape(car_shape)
    hump_collision = cq_solid_to_bullet_collision_shape(hump_shape)

    volume_m3 = (
        max(car_bbox.xlen, 1.0)
        * max(car_bbox.ylen, 1.0)
        * max(car_bbox.zlen, 1.0)
        * MM_TO_M
        * MM_TO_M
        * MM_TO_M
    )
    car_mass = max(80.0, volume_m3 * 400.0)

    car_bottom_offset_m = -car_bbox.zmin * MM_TO_M
    hump_bottom_offset_m = -hump_bbox.zmin * MM_TO_M
    car_initial_pos = [0.0, 0.0, car_bottom_offset_m + CAR_CLEARANCE_M]
    car_initial_orn = (0.0, 0.0, 0.0, 1.0)
    car_body = p.createMultiBody(
        baseMass=car_mass,
        baseCollisionShapeIndex=car_collision,
        basePosition=car_initial_pos,
        baseOrientation=car_initial_orn,
    )

    hump_initial_x = (car_bbox.xlen * 0.5 + hump_bbox.xlen) * MM_TO_M
    hump_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=hump_collision,
        basePosition=[hump_initial_x, 0.0, hump_bottom_offset_m],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
    )

    return {
        "client": client,
        "car_body": car_body,
        "hump_body": hump_body,
        "car_initial_pose": (car_initial_pos, car_initial_orn),
        "hump_bottom_offset_m": hump_bottom_offset_m,
        "hump_initial_x": hump_initial_x,
        "car_bbox": car_bbox,
        "hump_bbox": hump_bbox,
    }


def _simulate_frames(
    car_shape: cq.Shape,
    hump_shape: cq.Shape,
    world: dict,
    step_mm: float,
    total_travel_m: float,
) -> Tuple[
    List[List[cq.Shape]],
    List[Tuple[float, float, float]],
    List[Tuple[float, float, float, float]],
    List[Tuple[float, float, float]],
]:
    step_m = step_mm * MM_TO_M
    num_steps = max(1, int(total_travel_m / step_m))
    frames: List[List[cq.Shape]] = []
    car_pose = world["car_initial_pose"]
    hump_body = world["hump_body"]
    car_body = world["car_body"]
    hump_bottom_offset_m = world["hump_bottom_offset_m"]
    hump_initial_x = world["hump_initial_x"]
    hump_bottom_offset_mm = -world["hump_bbox"].zmin
    car_positions_mm: List[Tuple[float, float, float]] = []
    car_orientations_quat: List[Tuple[float, float, float, float]] = []
    hump_positions_mm: List[Tuple[float, float, float]] = []

    for step_idx in range(num_steps):
        x_hump = hump_initial_x - step_idx * step_m
        p.resetBasePositionAndOrientation(
            hump_body, [x_hump, 0.0, hump_bottom_offset_m], [0.0, 0.0, 0.0, 1.0]
        )
        p.resetBasePositionAndOrientation(car_body, car_pose[0], car_pose[1])
        for _ in range(SETTLE_STEPS):
            p.stepSimulation()
        car_pose = p.getBasePositionAndOrientation(car_body)
        car_pos_mm = tuple(coord / MM_TO_M for coord in car_pose[0])
        car_positions_mm.append(tuple(float(val) for val in car_pos_mm))
        car_orientations_quat.append(tuple(float(val) for val in car_pose[1]))

        frame_car = transform_cq_solid_with_bullet_pose(car_shape, car_pose[0], car_pose[1])
        frame_hump = hump_shape.translate(
            (x_hump / MM_TO_M, 0.0, hump_bottom_offset_mm)
        )
        hump_positions_mm.append((x_hump / MM_TO_M, 0.0, hump_bottom_offset_mm))
        frames.append([frame_car, frame_hump])

    return frames, car_positions_mm, car_orientations_quat, hump_positions_mm


def run_speed_hump_simulation(
    step_mm: float = DEFAULT_STEP_MM, total_travel_m: float = DEFAULT_TOTAL_TRAVEL_M
) -> SimulationResult:
    car_shape = _build_car_shape()
    hump_shape = _build_speed_hump_shape()
    world = _setup_world(car_shape, hump_shape)
    try:
        frames, car_positions_mm, car_orientations, hump_positions_mm = _simulate_frames(
            car_shape, hump_shape, world, step_mm, total_travel_m
        )
    finally:
        if p is not None:
            p.disconnect(world["client"])

    return SimulationResult(
        frames=frames,
        car_shape=car_shape,
        hump_shape=hump_shape,
        car_positions_mm=car_positions_mm,
        car_orientations_quat=car_orientations,
        hump_positions_mm=hump_positions_mm,
    )


if __name__ == "__main__":  # pragma: no cover - manual visualization entrypoint
    simulation = run_speed_hump_simulation()
    if show is not None and Animation is not None:
        try:
            viewer_payload = simulation.build_viewer_payload()
        except Exception as exc:  # pragma: no cover - viewer runtime issue
            print("Failed to prepare viewer animation:", exc)
            print("Animation frames generated (viewer unavailable).")
        else:
            if isinstance(viewer_payload, tuple):
                assembly, animation = viewer_payload
                try:
                    show(assembly)
                except Exception as show_exc:  # pragma: no cover - viewer runtime issue
                    print("Viewer display failed:", show_exc)
                else:
                    try:
                        animation.animate(1.0)
                    except Exception as animate_exc:
                        print("Viewer animation playback failed:", animate_exc)
            else:
                show(viewer_payload)
    else:
        print("Animation frames generated (viewer unavailable).")
