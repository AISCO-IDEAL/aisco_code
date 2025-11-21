import time
start = time.time()

import math
import cadquery as cq
from ocp_vscode import Animation, show

from subprojects.wp2_assembler.version2.components.nbz import call_assembly
from subprojects.wp2_assembler.version2.utils.Speed_hump import create_speed_hump_and_floor
from subprojects.wp2_assembler.version2.utils.reg_imp import get_regulation
from subprojects.wp2_assembler.version2.utils import Speed_hump

def _as_shape(obj: cq.Shape | cq.Workplane | cq.Assembly) -> cq.Shape:
    if isinstance(obj, cq.Assembly):
        return obj.toCompound()
    if isinstance(obj, cq.Workplane):
        value = obj.val()
        if isinstance(value, cq.Shape):
            return value
    if isinstance(obj, cq.Shape):
        return obj
    raise TypeError(f"Unsupported CadQuery object type: {type(obj)!r}")

def import_speed_bump() -> cq.Shape:
    """Return the regulation-based speed bump as a CadQuery Shape."""
    return _as_shape(create_speed_hump_and_floor())

if __name__ == "__main__":  # pragma: no cover - manual visualization helper
    print("Elapsed:", time.time() - start, "seconds")
    start = time.time()
    
    # Build the three pieces we want to see in the viewer.
    assembly_id = 30
    inner, outer = call_assembly(assembly_id)
    bump = import_speed_bump()
    _,_,_,Speed_hump_lenght,_ =  Speed_hump._collect_dimensions()
    shift_vec = cq.Vector(-(outer.BoundingBox().xmax + Speed_hump_lenght/2), 0, 0)

    # Set up a CadQuery assembly; ocp_vscode reads this to show and animate parts.
    assembly = cq.Assembly(name="SpeedHumpPreview")
    assembly.add(
        bump,
        name="speed_bump",
        loc=cq.Location((0,0,0)),
        color=cq.Color(0, 1, 0, 0.4),
    )
    assembly.add(
        outer,
        name="outer_nbz",
        loc=cq.Location((shift_vec)),
        color=cq.Color(0.7, 0.7, 0.7, 0.15),  # light tint so inner parts stay visible
    )
    assembly.add(
        inner,
        name="inner_nbz",
        loc=cq.Location((shift_vec)),
        color=cq.Color(0.9, 0.2, 0.2, 0.8),  # solid car geometry for clarity
    )
    assembly.mates = []  # Minimal mate list lets ocp_vscode detect this as animatable

    # Animation parameters: number of frames and spacing between them.
    frame_count = 100
    time_step = 0.02  # seconds between frames (controls playback speed)
    offsets = [(10.0 * i, 0.0, 0.0) for i in range(frame_count)]  # slide along +X
    # Build quaternions for outer_nbz: rotate about Z by 0→360 deg across frames.
    rotations = []
    for i in range(frame_count):
        rotations.append((5*i))
    times = [i * time_step for i in range(frame_count)]  # timestamps per frame
    

    # offsets = [(0,0,0),(outer.BoundingBox().xlen + 400,0,0)]

    # Build an ocp_vscode Animation and register a translation track for inner_nbz.
    animation = Animation(assembly)
    animation.add_track(f"/{assembly.name}/inner_nbz", "t", times, offsets)
    animation.add_track(f"/{assembly.name}/outer_nbz", "t", times, offsets)


    # Play the animation if the viewer exposes animate(); otherwise just show frames.
    animate = getattr(animation, "animate", None)
    if callable(animate):
        show(assembly)  # opens the viewer with the initial frame
        animate(1.0)  # play once; change the argument for repeats (e.g., animate(3))
    else:
        show(animation)  # fallback display if animate is absent


    print("Elapsed:", time.time() - start, "seconds")
