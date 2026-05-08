from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    solid_to_kaolin_mesh,
)
import torch
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
import cadquery as cq
from subprojects.wp2_assembler.version4.components.nbz import vision_nbz


def get_vision_cone_mesh(id_number) -> tuple[torch.Tensor, torch.Tensor]:
    vision_cone: cq.Assembly = vision_nbz(id_number)
    vison_cone_solid = vision_cone.toCompound().Solids()[0]
    print(f"vision cone: {vison_cone_solid}")
    dev = load_config(v="v2", cfg_name="general_config.yaml")["general_info"]["device"]
    if dev not in ["cpu", "cuda"]:
        raise ValueError(
            f"Invalid device specified in config: {dev}. Code needs modifying if you'd like to use a different device."
        )
    verts, faces = solid_to_kaolin_mesh(vison_cone_solid, device=dev)
    return verts, faces


def render_vis_cone(id_number):
    verts, faces = get_vision_cone_mesh(id_number)
    print(f"Vision cone vertices:\n{verts}\nVision cone faces:\n{faces}\n")


if __name__ == "__main__":
    id_number = load_bb_assembly_id()
    print(f"Rendering vision cone for assembly id: {id_number}")
    render_vis_cone(id_number)
    print("Done.")