"""
Drop the inner NBZ (assembly 30) onto the speed hump + floor in PyBullet.
Meshes are written next to this script under helper_files_pybullet_test/.
"""
import math
import time  # TEMP timing/ETA
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cadquery as cq
import numpy as np
import pybullet as p
from cadquery import exporters
from ocp_vscode import show

from subprojects.wp2_assembler.version2.components.nbz import call_assembly
from subprojects.wp2_assembler.version2.utils import Speed_hump  # Added helper import
#from subprojects.wp2_assembler.version2.utils.animation_speed_hump import animate
from subprojects.wp2_assembler.version2.utils.Speed_hump import (
    create_speed_hump_and_floor,
)


def get_speed_hump_total_length() -> float:
    """Return the total speed hump length from regulations (mm).  # Added helper"""
    _, _, _, total_length, _ = Speed_hump._collect_dimensions()
    return total_length

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

def _as_shape(obj: cq.Shape | cq.Workplane | cq.Assembly) -> cq.Shape:
    """Return a CadQuery shape regardless of whether a Workplane or Assembly was provided."""
    if isinstance(obj, cq.Assembly):
        return obj.toCompound()
    if isinstance(obj, cq.Workplane):
        return obj.val()
    if isinstance(obj, cq.Shape):
        return obj
    raise TypeError(f"Unsupported CadQuery object type: {type(obj)!r}")

def trim_outer_nbz2(assembly_number: int, pos_array, orn_array) -> cq.Shape:

    inner, outer = call_assembly(assembly_number)
    ground_shape = _as_shape(create_speed_hump_and_floor())

    if len(pos_array) != len(orn_array):
        raise ValueError("Error in trim_outer_nbz: pos_array and orn_array must have the same length")

    
    trimmed_outer = outer
    knife = [ground_shape]
    for i in range(200):
        loc = cq.Location(x=i*10-200, y=0, z=0, rx= 0, ry= 0, rz= 0)

        trimmed_outer = trimmed_outer.located(loc)
        knife.append(trimmed_outer.intersect(ground_shape).located(loc))
        if(i%10==0):
            print(i)

    trimmed_outer = outer
    for i in knife:
        trimmed_outer=trimmed_outer.cut(i)
    
    '''
    i=0
    knife = []
    moved_outer = outer
    l = len(pos_array)

    for pos, orn in zip(pos_array, orn_array):
        i+=1
        if(i%10==0):
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            roll_deg, pitch_deg, yaw_deg = (
                math.degrees(roll),
                math.degrees(pitch),
                math.degrees(yaw),
            )
            r = 1
            location = cq.Location(x=round(pos[0], r), y=round(pos[1], r), z=round(pos[2], r), rx= round(roll_deg, r), ry= round(pitch_deg, r), rz= round(yaw_deg, r))

            moved_outer=moved_outer.located(location)

            knife.append(moved_outer.intersect(ground_shape).located(location))

            if(i%100==0):
                print(i, l)

    '''

    show(knife)
    input("To compund")
    combined_compound = cq.Compound.makeCompound(knife)
    show(combined_compound)

    input("simple? ")

    simple_tunnel = combined_compound.clean()
    show(simple_tunnel)
    input("fuse? ")

    
    fused_tunnel = knife[0]
    
    # Loop through the rest and add them one by one
    a = 0
    for shape in knife[1:]:
        a+=1
        fused_tunnel = fused_tunnel.fuse(shape, tol=0.1)
        if(a%10 == 0):
            print(a,len(knife))
            show(fused_tunnel)
    
    show(fused_tunnel)

    input("Continue? ")

    final_outer = trimmed_outer.cut(combined_compound)
    show(final_outer)
    
    input("Continue? ")


    for i in knife:
        a+=1
        try:
            trimmed_outer=trimmed_outer.cut(i)
        except:
            print("error at a:", a)
        if(a%10==0):
            print(a, len(knife))
            show(trimmed_outer, i)

    input("Continue? ")
    show(trimmed_outer)

    return trimmed_outer

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

def is_stable(body_id: int, tol: float = 1e-3) -> bool:
    """Return True if linear and angular speeds are below tol."""
    lin_vel, ang_vel = p.getBaseVelocity(body_id)
    return np.linalg.norm(lin_vel) < tol and np.linalg.norm(ang_vel) < tol

def sim(assembly_number, only_final: bool = False):

    #returns position in meters and orientation in quaterion radians of the settled car

    # --- Build and export meshes -------------------------------------------------

    inner_asm, outer = call_assembly(assembly_number)
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

    '''
    print("Exported parts:")
    for name, path in car_part_paths:
        print(f" - {name}: {path}")
    '''

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

    # Build the multibody from the exported STL parts (body + wheels)
    car_parts: list[tuple[str, int]] = []
    total_mass = 50.0
    mass_each = total_mass / len(car_part_paths)

    bbox = outer.BoundingBox()
    outer_height = int((bbox.xmax - bbox.xmin)/1000)

    start_pos = [0, 0, outer_height]  # start
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


    pos_array = []
    orn_array = []

    bbox = outer.BoundingBox()
    outer_length = int(bbox.xmax - bbox.xmin)  # in mm
    move = int(outer_length/2 + get_speed_hump_total_length()/2 + 100)

    stable_steps = 0
    needed_stable_steps = 100
    max_steps = 10_000

    for i in range(max_steps):

        pos, orn = p.getBasePositionAndOrientation(root)
        x, y, z = pos
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)    # Radians!

        p.resetBasePositionAndOrientation(root, [-move/1000, 0, z], p.getQuaternionFromEuler([roll, pitch, 0])) #fix x,y and yaw
        #pos_array.append([i * 1000 for i in pos])
        #orn_array.append(orn)

        stable_steps = (stable_steps + 1)*is_stable(body_id)

        if (stable_steps >= needed_stable_steps):
            break
        if (i == max_steps-1):
            raise Exception("Error in sim: car didnt settle in the falling stage")

        p.stepSimulation()

    for step in range(-move,move):
        step_in_meters = step / 1000

        pos, orn = p.getBasePositionAndOrientation(root)
        x, y, z = pos
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)    # Radians!

        p.resetBasePositionAndOrientation(root, [step_in_meters, 0, z], p.getQuaternionFromEuler([roll, pitch, 0])) #fix x,y and yaw

        pos_array.append([i * 1000 for i in pos])
        orn_array.append(orn)

        p.stepSimulation()


    # --- Report final pose and exit ---------------------------------------------
    final_pos, final_orn = p.getBasePositionAndOrientation(root)
    p.disconnect()
    if (only_final):
        return final_pos, final_orn # pos in meters, orn in quaternions and radians!
    else:
        return pos_array, orn_array


def trim_outer_nbz(assembly_number: int, pos_array, orn_array) -> cq.Shape:

    inner, outer = call_assembly(assembly_number)
    ground_shape = create_speed_hump_and_floor()
    inv_pos_array, inv_orn_array = invert_transform(pos_array, orn_array)

    i=0
    knife = []
    ground_shape_solid = ground_shape.val()
    l = len(inv_pos_array)

    print("Start")
    start = time.time()

    inv_pos_array = np.round(inv_pos_array, 1)
    inv_orn_array = np.round(inv_orn_array, 2)

    euler_array = np.array([p.getEulerFromQuaternion(q) for q in inv_orn_array])
    euler_deg = np.degrees(euler_array)

    for pos, orn in zip(inv_pos_array, euler_deg):
        #i+=1
        #if(i%1==0):
        '''
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        roll_deg, pitch_deg, yaw_deg = (
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),
        )
        '''
        roll_deg, pitch_deg, yaw_deg = orn
        
        location = cq.Location(x=pos[0], y=pos[1], z=pos[2], rx= roll_deg, ry= pitch_deg, rz= yaw_deg)
        
        knife.append(outer.intersect(ground_shape_solid.located(location)))

    end = time.time()
    print(end-start)
    #show(knife)

    start = time.time()
    #combined_compound = cq.Compound.makeCompound(knife).fuse().clean()
    combined_compound = fuse_compound_tree(cq.Compound.makeCompound(knife))
    combined_compound = combined_compound.clean()
    end = time.time()
    print(end-start)

    show(combined_compound)
    return

def invert_transform(pos_list, quat_list):
    inv_pos_list = []
    inv_quat_list = []

    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return np.array([
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
            aw*bw - ax*bx - ay*by - az*bz
        ])

    def qrot(q, v):
        v_q = np.array([v[0], v[1], v[2], 0.0])
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        return qmul(qmul(q, v_q), q_conj)[:3]

    for pos, quat in zip(pos_list, quat_list):
        px, py, pz = pos
        qx, qy, qz, qw = quat

        quat = np.array([qx, qy, qz, qw])
        quat_inv = np.array([-qx, -qy, -qz,  qw])

        p = np.array([px, py, pz])
        pos_inv = -qrot(quat_inv, p)

        inv_pos_list.append(pos_inv.tolist())
        inv_quat_list.append(quat_inv.tolist())

    return inv_pos_list, inv_quat_list

def fuse_compound_tree(comp: cq.Compound):
    # Extract shapes inside the compound
    work = comp.Solids()
    # In most real cases you only expect Solids, but this is safe

    if len(work) == 0:
        return None
    # Turn every element into a Solid/Shape for fusing

    # Tree fuse loop
    while len(work) > 1:
        new_work = []
        for i in range(0, len(work), 2):
            if i + 1 < len(work):
                fused = work[i].fuse(work[i + 1])
                # Clean after each fuse for stability
                fused = fused.clean()
                new_work.append(fused)
            else:
                # Odd element, move to next round unchanged
                new_work.append(work[i])
        work = new_work

    # Return the single fused result
    return work[0]

if __name__ == "__main__":
    # Visualization: show and trim the intersection between ground + speed hump and outer NBZ.
    # Adjust the ID to any assembly present in your assembly_map.yaml.
    demo_id = 10
    print(f"Showing outer NBZ vs speed hump for assembly {demo_id}...")

    # Uncomment to run the PyBullet drop + slide simulation.

    pos_array2, orn_array2 = sim(demo_id)

    #animate(demo_id, pos_array2, orn_array2)

    trim_outer_nbz(demo_id, pos_array2, orn_array2)