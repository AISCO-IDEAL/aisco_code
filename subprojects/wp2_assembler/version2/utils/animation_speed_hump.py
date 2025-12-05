import cadquery as cq
from ocp_vscode import Animation, show

from subprojects.wp2_assembler.version2.components.nbz import call_assembly
from subprojects.wp2_assembler.version2.utils import Speed_hump
from subprojects.wp2_assembler.version2.utils.Speed_hump import create_speed_hump_and_floor
from subprojects.wp2_assembler.version2.utils.positioning_program import sim
from subprojects.wp2_assembler.version2.utils.positioning_program import invert_transform


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

def animate(assembly_id, pos_array, orn_array):

    if(len(pos_array) != len(orn_array)):
        raise Exception("Error in animation: len(pos_array) != len(orn_array)")
    else:
        times = [(i + 1) * 0.001 for i in range(len(pos_array))]

    # Build the three pieces we want to see in the viewer.
    inner, outer = call_assembly(assembly_id)
    bump = import_speed_bump()
    _,_,_,Speed_hump_lenght,_ =  Speed_hump._collect_dimensions()

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
        loc=cq.Location((0,0,0)),
        color=cq.Color(0.7, 0.7, 0.7, 0.15),  # light tint so inner parts stay visible
    )
    assembly.add(
        inner,
        name="inner_nbz",
        loc=cq.Location((0,0,0)),
        color=cq.Color(0.9, 0.2, 0.2, 0.8),  # solid car geometry for clarity
    )
    assembly.mates = []  # Minimal mate list lets ocp_vscode detect this as animatable

    # Build an ocp_vscode Animation and register a translation track for inner_nbz.
    animation = Animation(assembly)

    '''
    animation.add_track(f"/{assembly.name}/inner_nbz", "t", times, pos_array)
    animation.add_track(
        f"/{assembly.name}/inner_nbz",
        "q",
        times,
        [_quat_dict(q) for q in orn_array],
    )

    animation.add_track(f"/{assembly.name}/outer_nbz", "t", times, pos_array)
    animation.add_track(
        f"/{assembly.name}/outer_nbz",
        "q",
        times,
        [_quat_dict(q) for q in orn_array],
    )
    '''

    inv_pos, inv_quat = invert_transform( )

    animation.add_track(f"/{assembly.name}/speed_bump", "t", times, coords)
    animation.add_track(
        f"/{assembly.name}/speed_bump",
        "q",
        times,
        [_quat_dict(q) for q in orn_array],
    )

    show(assembly)  # opens the viewer with the initial frame
    animation.animate(1.0)

if __name__ == "__main__":

    demo_id = 10
    pos_array2, orn_array2 = sim(demo_id)
    animate(demo_id, pos_array2, orn_array2)
