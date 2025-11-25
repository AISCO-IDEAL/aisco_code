"""
Drop the inner NBZ (assembly 30) onto the speed hump + floor in PyBullet.
Meshes are written next to this script under helper_files_pybullet_test2/.
"""

import math
from pathlib import Path
from time import sleep

import cadquery as cq
import numpy as np
import pybullet as p
from cadquery import exporters

from subprojects.wp2_assembler.version2.components.nbz import call_assembly
from subprojects.wp2_assembler.version2.utils.Speed_hump import (
    create_speed_hump_and_floor,
)


def export_shape_to_stl(shape: cq.Shape, path: Path) -> str:
    """Export a CadQuery shape to STL and return the file path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exporters.export(shape, str(path))
    return str(path)


def assembly_obj_to_shape(asm_obj) -> cq.Shape:
    """Get a Shape located in world coordinates from an AssemblyObject."""
    base_obj = asm_obj.obj
    if isinstance(base_obj, cq.Assembly):
        shape = base_obj.toCompound()
    elif isinstance(base_obj, cq.Workplane):
        shape = base_obj.val()
    elif isinstance(base_obj, cq.Shape):
        shape = base_obj
    else:
        raise TypeError(f"Unsupported assembly object type: {type(base_obj)!r}")
    return shape.located(asm_obj.loc) if getattr(asm_obj, "loc", None) else shape


def export_inner_parts(inner_assembly: cq.Assembly, folder: Path):
    """
    Export wheels individually and group the remaining geometry as the canopy.
    Returns (canopy_path, [(part_name, path), ...] for wheels).
    """
    wheel_paths: list[tuple[str, str]] = []
    canopy_shapes: list[cq.Shape] = []

    for name, asm_obj in inner_assembly.objects.items():
        if name == inner_assembly.name:
            continue
        if asm_obj is None or getattr(asm_obj, "obj", None) is None:
            continue
        shape = assembly_obj_to_shape(asm_obj)
        part_name = name or "part"
        if part_name.lower().startswith("wheel"):
            wheel_path = export_shape_to_stl(shape, folder / f"{part_name}.stl")
            wheel_paths.append((part_name, wheel_path))
        else:
            canopy_shapes.append(shape)

    canopy_path = None
    if canopy_shapes:
        canopy_shape = cq.Compound.makeCompound(canopy_shapes)
        canopy_path = export_shape_to_stl(canopy_shape, folder / "canopy.stl")

    return canopy_path, wheel_paths


def is_stable(lin_vel, ang_vel, tol: float = 1e-3) -> bool:
    return np.linalg.norm(lin_vel) < tol and np.linalg.norm(ang_vel) < tol


# ============================================================
#  Build & export pieces from CadQuery assemblies
# ============================================================

EXPORT_DIR = Path(__file__).parent / "helper_files_pybullet_test2"
inner_asm, _ = call_assembly(30)

ground_shape = create_speed_hump_and_floor()
ground_stl = export_shape_to_stl(ground_shape, EXPORT_DIR / "ground.stl")

canopy_path, wheel_paths = export_inner_parts(inner_asm, EXPORT_DIR)

car_part_paths: list[tuple[str, str]] = []
if canopy_path:
    car_part_paths.append(("canopy", canopy_path))
car_part_paths.extend(wheel_paths)

if not car_part_paths:
    raise RuntimeError("No parts exported for PyBullet simulation.")


# ============================================================
# PyBullet setup
# ============================================================

cid = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# --- Ground (static concave mesh) ---
ground_id = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=p.createCollisionShape(
        p.GEOM_MESH,
        fileName=ground_stl,
        meshScale=[0.001]*3,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH
    ),
    basePosition=[0, 0, 0],
)

p.changeDynamics(ground_id, -1,
                 lateralFriction=0.0,
                 rollingFriction=0.0,
                 spinningFriction=0.0)


# ============================================================
# Load decomposed car as multiple convex bodies
# ============================================================

car_parts = []
num_parts = len(car_part_paths)
mass_each = 50.0 / num_parts

start_pos = [1.0, 0, 1.0]
start_orn = p.getQuaternionFromEuler([0, 0, 0])

for part_name, path in car_part_paths:
    col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=path,
        meshScale=[0.001]*3   # mm → m
    )
    vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=path,
        meshScale=[0.001]*3
    )

    bid = p.createMultiBody(
        baseMass=mass_each,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=start_pos,
        baseOrientation=start_orn
    )

    p.changeDynamics(bid, -1,
                     lateralFriction=0.0,
                     rollingFriction=0.0,
                     spinningFriction=0.0)

    car_parts.append(bid)

# Fix all parts together (root = first)
root = car_parts[0]

for child in car_parts[1:]:
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


# ============================================================
# Simulation loop (your original loop)
# ============================================================

x_lock = 0.0
y_lock = 0.0
yaw_lock = 0.0

stable_counter = 0
max_steps = 5_000
sim_dt = p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0)
required_stable_steps = 100

time_arr = []
rolls = []
pitchs = []
yaws = []
offset = []
offset_ground = []

for step in range(max_steps):

    sleep(1/4800)
    if step % 100 == 0:
        print(step)

    pos, orn = p.getBasePositionAndOrientation(root)
    x, y, z = pos
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    yaw = yaw_lock

    time_arr.append(step/1000)
    rolls.append(roll)
    pitchs.append(pitch)
    yaws.append(yaw)
    offset.append((x*1000, y*1000, z*1000))  # meters → mm

    pos2, orn2 = p.getBasePositionAndOrientation(ground_id)
    offset_ground.append(pos2)

    p.stepSimulation()

    lin_vel, ang_vel = p.getBaseVelocity(root)
    stable_counter = stable_counter + 1 if is_stable(lin_vel, ang_vel) else 0

    if stable_counter >= required_stable_steps:
        print(f"Settled after {step} steps ({step * sim_dt:.2f} s)")
        break

else:
    print("Did not settle within max steps.")


# ============================================================
# Final pose
# ============================================================

final_pos, final_orn = p.getBasePositionAndOrientation(root)
final_euler = [math.degrees(a) for a in p.getEulerFromQuaternion(final_orn)]
print("Final position (m):", final_pos)
print("Final orientation (deg):", final_euler)

p.disconnect()


def sim():
    return time_arr, offset, rolls, pitchs, yaws, offset_ground
