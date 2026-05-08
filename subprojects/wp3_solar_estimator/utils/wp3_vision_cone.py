import cadquery as cq
import torch
from subprojects.wp2_assembler.version4.components.nbz import vision_nbz
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils import save_scene_glb, solid_rgb
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (
    load_bb_assembly_id,
    load_inner_outer_solids,
    merge_meshes,
    solid_to_kaolin_mesh,
)


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

def mesh_intersection(
    verts1: torch.Tensor, faces1: torch.Tensor, vertices2: torch.Tensor, faces2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the intersection of two meshes defined by their vertices and faces.

    :param verts1: Vertices of the first mesh (N1 x 3)
    :param faces1: Faces of the first mesh (M1 x 3)
    :param vertices2: Vertices of the second mesh (N2 x 3)
    :param faces2: Faces of the second mesh (M2 x 3)
    :return: A tuple containing the vertices and faces of the intersected mesh.
    """
    # This is a placeholder implementation. Computing the intersection of two meshes is non-trivial and typically requires a robust geometry processing library.
    # You might want to use libraries like PyMesh, Trimesh, or CGAL for this purpose.
    raise NotImplementedError(
        "Mesh intersection is not implemented. Consider using a geometry processing library for this functionality."
    )


def render_vis_cone(id_number, out_path: str = "/cluster/home/aalthaus/vision_cone.glb"):
    verts, faces = get_vision_cone_mesh(id_number)
    print(f"Vision cone vertices:\n{verts}\nVision cone faces:\n{faces}\n")
    verts_np = verts.cpu().numpy()
    save_scene_glb(out_path, [("vision_cone", verts_np, faces.cpu().numpy(), solid_rgb(len(verts_np), (0.9, 0.1, 0.9)))])
    print(f"Saved to {out_path} — open with: python -m http.server 8080 (then viewer/?model={out_path})")


if __name__ == "__main__":
    id_number = load_bb_assembly_id()
    print(f"Saving GLB scene for vision cone: {id_number}")
    render_vis_cone(id_number)
    print("Done.")
