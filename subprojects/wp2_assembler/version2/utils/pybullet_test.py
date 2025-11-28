"""
Drop the inner NBZ (assembly 30) onto the speed hump + floor in PyBullet.
Meshes are written next to this script under helper_files_pybullet_test/.
"""

import math
import sys
from pathlib import Path
from time import sleep
from typing import Any, List, Optional, Tuple

# Ensure local package (with config files) is on sys.path ahead of any installed copy
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cadquery as cq
import numpy as np
import pybullet as p
from cadquery import exporters

from subprojects.wp2_assembler.version2.components.nbz import call_assembly
from subprojects.wp2_assembler.version2.utils.Speed_hump import (
    create_speed_hump_and_floor,
)


def export_shape_to_stl(shape: cq.Shape, name: str) -> str:
    """
    Save a CadQuery shape to STL so PyBullet can load it.
    Returns the filesystem path as a string.
    """
    folder = Path(__file__).parent / "helper_files_pybullet_test"

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.stl"
    exporters.export(shape, str(path))  # exporter expects a string path
    return str(path)


def assembly_obj_to_shape(asm_obj: Any) -> cq.Shape:
    """Extract a world-space CadQuery shape from an AssemblyObject."""
    base_obj = asm_obj.obj
    if isinstance(base_obj, cq.Assembly):
        shape = base_obj.toCompound()
    elif isinstance(base_obj, cq.Workplane):
        shape = base_obj.val()
    elif isinstance(base_obj, cq.Shape):
        shape = base_obj
    else:
        raise TypeError(f"Cannot export object type {type(base_obj)} for {asm_obj.name}")

    return shape.located(asm_obj.loc) if getattr(asm_obj, "loc", None) else shape


def export_inner_nbz_parts(
    assembly: cq.Assembly,
) -> tuple[Optional[str], list[tuple[str, str]]]:
    """
    Export the inner NBZ as one body STL and individual wheel STLs.
    Wheels are detected by name prefix 'wheel' / 'Wheel'.
    Returns (body_path, [(wheel_name, wheel_path), ...]).
    """
    wheel_paths: list[tuple[str, str]] = []
    body_shapes: list[cq.Shape] = []

    for name, asm_obj in assembly.objects.items():
        # Skip self-reference to avoid duplicating the full assembly
        if name == assembly.name:
            continue
        if asm_obj is None or getattr(asm_obj, "obj", None) is None:
            continue
        part_name = name or "part"
        shape = assembly_obj_to_shape(asm_obj)
        if part_name.lower().startswith("wheel"):
            wheel_paths.append((part_name, export_shape_to_stl(shape, part_name)))
        else:
            body_shapes.append(shape)

    body_path: Optional[str] = None
    if body_shapes:
        body_compound = cq.Compound.makeCompound(body_shapes)
        body_path = export_shape_to_stl(body_compound, "inner_nbz_body")

    return body_path, wheel_paths


def is_stable(lin_vel, ang_vel, tol: float = 1e-3) -> bool:
    """Check if linear and angular speeds are below tol."""
    return np.linalg.norm(lin_vel) < tol and np.linalg.norm(ang_vel) < tol


# --- Build and export meshes -------------------------------------------------

inner_asm, _ = call_assembly(30)
ground_shape = create_speed_hump_and_floor()

ground_stl = export_shape_to_stl(ground_shape, "ground")
body_stl, wheel_parts = export_inner_nbz_parts(inner_asm)

if body_stl is None and not wheel_parts:
    body_stl = export_shape_to_stl(inner_asm.toCompound(), "inner_nbz")

car_part_paths: list[tuple[str, str]] = []
if body_stl:
    car_part_paths.append(("body", body_stl))
car_part_paths.extend(wheel_parts)

if not car_part_paths:
    raise RuntimeError("No car parts were exported for PyBullet.")

print("Exported parts:")
for name, path in car_part_paths:
    print(f" - {name}: {path}")

# --- PyBullet setup ----------------------------------------------------------

cid = p.connect(p.DIRECT)  # use p.GUI if you want a window
p.setGravity(0, 0, -9.81)

ground_id = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_MESH,
        fileName=ground_stl,
        meshScale=[0.001, 0.001, 0.001],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,  # floor + hump can be concave
    ),  # mm → meters
    basePosition=[0, 0, 0],
)

p.changeDynamics(ground_id, -1,
                 lateralFriction=0.0,
                 rollingFriction=0.0,
                 spinningFriction=0.0)

# --- Drop and wait for stabilization ----------------------------------------

x_lock = 0.0
y_lock = 0.0
yaw_lock = 0.0      # Radians!

# Build the multibody from the exported STL parts (body + wheels)
car_parts: list[tuple[str, int]] = []
total_mass = 50.0
mass_each = total_mass / len(car_part_paths)
start_pos = [1.0, 0, 1.0]  # start 1 m above
start_orn = p.getQuaternionFromEuler([0, 0, 0])

for name, path in car_part_paths:
    col = p.createCollisionShape(
        p.GEOM_MESH, fileName=path, meshScale=[0.001, 0.001, 0.001]
    )
    vis = p.createVisualShape(
        p.GEOM_MESH, fileName=path, meshScale=[0.001, 0.001, 0.001]
    )
    body_id = p.createMultiBody(
        baseMass=mass_each,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=start_pos,
        baseOrientation=start_orn,
    )
    p.changeDynamics(
        body_id, -1, lateralFriction=0.0, rollingFriction=0.0, spinningFriction=0.0
    )
    car_parts.append((name, body_id))

# Fix all parts together (root = first entry)
root = car_parts[0][1]
for _, child in car_parts[1:]:
    p.createConstraint(
        parentBodyUniqueId=root,
        parentLinkIndex=-1,
        childBodyUniqueId=child,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
    )

stable_counter = 0
max_steps = 10_000
sim_dt = p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0)
required_stable_steps = 100

time_arr = []
rolls = []
pitchs = []
yaws = []
offset = []

pos_array = []
orn_array = []

offset_ground =[]

for step in range(max_steps):

    #sleep(1/4800)
    pos, orn = p.getBasePositionAndOrientation(root)
    x, y, z = pos
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)    # Radians!

    #p.resetBasePositionAndOrientation(root, [x, y, z], p.getQuaternionFromEuler([roll, pitch, yaw]))

    #p.resetBasePositionAndOrientation(root, [x, y, z], orn)

    time_arr.append(step/1000)

    rolls.append(roll)
    pitchs.append(pitch)
    yaws.append(yaw)

    offset.append((x*1000, y*1000, z*1000))

    pos_array.append([i * 1000 for i in pos])
    orn_array.append(orn)

    p.stepSimulation()

    lin_vel, ang_vel = p.getBaseVelocity(root)
    stable_counter = stable_counter + 1 if is_stable(lin_vel, ang_vel) else 0
    if stable_counter >= required_stable_steps:
        print(f"Settled after {step} steps ({step * sim_dt:.2f} s)")
        break
else:
    print("Did not settle within the max step budget.")


# --- Report final pose and exit ---------------------------------------------

final_pos, final_orn = p.getBasePositionAndOrientation(root)
final_euler = [math.degrees(a) for a in p.getEulerFromQuaternion(final_orn)]
print("Final position (m):", final_pos)
print("Final orientation (deg):", final_euler)

#sleep(1)

p.disconnect()

print(len(time_arr),len(pos_array),len(orn_array))

def sim():
    return time_arr, pos_array, orn_array
