"""
Drop the inner NBZ (assembly 30) onto the speed hump + floor in PyBullet.
Meshes are written next to this script under helper_files_pybullet_test/.
"""

import math
import os
from time import sleep

import numpy as np
import pybullet as p
import trimesh


def is_stable(lin_vel, ang_vel, tol: float = 1e-3) -> bool:
    return np.linalg.norm(lin_vel) < tol and np.linalg.norm(ang_vel) < tol


# ============================================================
#  Build & export convex pieces with trimesh
# ============================================================

base_dir = "/Users/federicosanchez/Documents/ETH/Aisco"
ground_stl = os.path.join(base_dir, "ground.stl")
inner_stl = os.path.join(base_dir, "inner_nbz.stl")

print("Loading STL...")
mesh = trimesh.load(inner_stl)

# Split into connected components
parts = mesh.split(only_watertight=False)
if len(parts) == 1:
    print("Only one shell found. Running convex decomposition...")
    parts = mesh.convex_decomposition()

print(f"Found {len(parts)} parts")

# Export each part
part_paths = []
for i, pm in enumerate(parts):
    out_path = os.path.join(base_dir, f"inner_part_{i}.stl")
    pm.export(out_path)
    part_paths.append(out_path)

print("Exported decomposed parts.")


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
num_parts = len(part_paths)
mass_each = 50.0 / num_parts

start_pos = [1.0, 0, 1.0]
start_orn = p.getQuaternionFromEuler([0, 0, 0])

for path in part_paths:
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
max_steps = 1_000
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
