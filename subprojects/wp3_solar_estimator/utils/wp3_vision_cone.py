import torch


def get_vision_cone_mesh(id_number) -> tuple[torch.Tensor, torch.Tensor]:
    from subprojects.wp2_assembler.version4.components.nbz import vision_nbz
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
    from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (
        solid_to_kaolin_mesh,
    )

    vision_cone = vision_nbz(id_number)
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
    import trimesh

    device = verts1.device
    dtype = verts1.dtype

    mesh1 = trimesh.Trimesh(vertices=verts1.detach().cpu().numpy(), faces=faces1.detach().cpu().numpy(), process=False)
    mesh2 = trimesh.Trimesh(
        vertices=vertices2.detach().cpu().numpy(), faces=faces2.detach().cpu().numpy(), process=False
    )

    result = trimesh.boolean.intersection([mesh1, mesh2])
    if result is None or result.is_empty:
        empty_verts = torch.empty((0, 3), device=device, dtype=dtype)
        empty_faces = torch.empty((0, 3), device=device, dtype=faces1.dtype)
        return empty_verts, empty_faces

    if isinstance(result, trimesh.Scene):
        result = trimesh.util.concatenate(tuple(result.geometry.values()))

    verts = torch.as_tensor(result.vertices, device=device, dtype=dtype)
    faces = torch.as_tensor(result.faces, device=device, dtype=faces1.dtype)
    return verts, faces


def render_vis_cone(id_number, out_path: str = "/cluster/home/aalthaus/vision_cone.glb"):
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils import save_scene_glb, solid_rgb

    verts, faces = get_vision_cone_mesh(id_number)
    print(f"Vision cone vertices:\n{verts}\nVision cone faces:\n{faces}\n")
    verts_np = verts.cpu().numpy()
    save_scene_glb(out_path, [("vision_cone", verts_np, faces.cpu().numpy(), solid_rgb(len(verts_np), (0.9, 0.1, 0.9)))])
    print(f"Saved to {out_path} — open with: python -m http.server 8080 (then viewer/?model={out_path})")


def _box_mesh(center: tuple[float, float, float], size: float) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = center
    h = size / 2
    verts = torch.tensor(
        [
            [x - h, y - h, z - h],
            [x + h, y - h, z - h],
            [x + h, y + h, z - h],
            [x - h, y + h, z - h],
            [x - h, y - h, z + h],
            [x + h, y - h, z + h],
            [x + h, y + h, z + h],
            [x - h, y + h, z + h],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=torch.long,
    )
    return verts, faces


def _save_test_scene(out_path: str, meshes: list[tuple[str, torch.Tensor, torch.Tensor, tuple[float, float, float]]]):
    import trimesh

    scene = trimesh.Scene()
    for name, verts, faces, rgb in meshes:
        mesh = trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy(), process=False)
        color = [*rgb, 1.0]
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=color, doubleSided=True)
        )
        scene.add_geometry(mesh, geom_name=name)
    scene.export(out_path)


if __name__ == "__main__":
    out_path = "vision_cone_intersection_test.glb"
    verts1, faces1 = _box_mesh(center=(0.0, 0.0, 0.0), size=2.0)
    verts2, faces2 = _box_mesh(center=(0.7, 0.7, 0.7), size=2.0)
    verts_i, faces_i = mesh_intersection(verts1, faces1, verts2, faces2)

    _save_test_scene(
        out_path,
        [
            ("box_1", verts1, faces1, (0.1, 0.4, 0.9)),
            ("box_2", verts2, faces2, (0.9, 0.5, 0.1)),
            ("intersection", verts_i, faces_i, (0.1, 0.9, 0.3)),
        ],
    )

    print(f"Saved visual intersection test to {out_path}")
    print("Blue/orange are the input boxes. Green is their intersection.")