"""
Drop the inner NBZ (assembly 30) onto the speed hump + floor in PyBullet.
Meshes are written next to this script under helper_files_pybullet_test/.
"""

from __future__ import annotations

import math
from pathlib import Path

import cadquery as cq
import numpy as np
import pybullet as p
import pybullet_data
from cadquery import exporters

from subprojects.wp2_assembler.version2.components.nbz import call_assembly
from subprojects.wp2_assembler.version2.utils.Speed_hump import create_speed_hump_and_floor


def export_shape_to_stl(shape: cq.Shape, name: str, folder: Path) -> str:
    """
    Save a CadQuery shape to STL so PyBullet can load it.
    Returns the filesystem path as a string.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.stl"
    exporters.export(shape, str(path))  # exporter expects a string path
    return str(path)


def is_stable(lin_vel, ang_vel, tol: float = 1e-3) -> bool:
    """Check if linear and angular speeds are below tol."""
    return np.linalg.norm(lin_vel) < tol and np.linalg.norm(ang_vel) < tol


# --- Build and export meshes -------------------------------------------------

inner_asm, _ = call_assembly(30)
inner_shape = inner_asm.toCompound()
ground_shape = create_speed_hump_and_floor()

mesh_dir = Path(__file__).parent / "helper_files_pybullet_test"
inner_stl = export_shape_to_stl(inner_shape, "inner_nbz", mesh_dir)
ground_stl = export_shape_to_stl(ground_shape, "ground", mesh_dir)


# --- PyBullet setup ----------------------------------------------------------

cid = p.connect(p.DIRECT)  # use p.GUI if you want a window
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

ground_col = p.createCollisionShape(p.GEOM_MESH, fileName=ground_stl)
ground_vis = p.createVisualShape(p.GEOM_MESH, fileName=ground_stl)
ground_id = p.createMultiBody(
    baseMass=0.0,
    baseCollisionShapeIndex=ground_col,
    baseVisualShapeIndex=ground_vis,
    basePosition=[0, 0, 0],
)

car_col = p.createCollisionShape(p.GEOM_MESH, fileName=inner_stl)
car_vis = p.createVisualShape(p.GEOM_MESH, fileName=inner_stl, rgbaColor=[0.8, 0.2, 0.2, 0.8])
car_id = p.createMultiBody(
    baseMass=50.0,
    baseCollisionShapeIndex=car_col,
    baseVisualShapeIndex=car_vis,
    basePosition=[0, 0, 1.0],  # start 1 m above
    baseOrientation=[0, 0, 0, 1],
)


# --- Drop and wait for stabilization ----------------------------------------

stable_counter = 0
xy_anchor = (0.0, 0.0)  # lock X/Y by resetting each step
max_steps = 10_000
sim_dt = p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0)
required_stable_steps = 500

for step in range(max_steps):
    pos, orn = p.getBasePositionAndOrientation(car_id)
    x, y, z = pos
    p.resetBasePositionAndOrientation(car_id, [xy_anchor[0], xy_anchor[1], z], orn)

    p.stepSimulation()

    lin_vel, ang_vel = p.getBaseVelocity(car_id)
    stable_counter = stable_counter + 1 if is_stable(lin_vel, ang_vel) else 0
    if stable_counter >= required_stable_steps:
        print(f"Settled after {step} steps ({step * sim_dt:.2f} s)")
        break
else:
    print("Did not settle within the max step budget.")


# --- Report final pose and exit ---------------------------------------------

final_pos, final_orn = p.getBasePositionAndOrientation(car_id)
final_euler = [math.degrees(a) for a in p.getEulerFromQuaternion(final_orn)]
print("Final position (m):", final_pos)
print("Final orientation (deg):", final_euler)

p.disconnect()

print(f"Meshes saved under: {mesh_dir}")
