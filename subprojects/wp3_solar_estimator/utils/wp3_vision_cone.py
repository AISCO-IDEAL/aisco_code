from pathlib import Path

import torch


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
    else:
        from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config  # noqa: PLC0415

        configured_device = load_config(v="v2", cfg_name="general_config.yaml")["general_info"]["device"]
        resolved = torch.device(configured_device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"Invalid device specified: {resolved}. Code needs modifying if you'd like to use a different device.")
    return resolved


def get_vision_cone_mesh(id_number: int, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    from subprojects.wp2_assembler.version4.components.nbz import vision_nbz  # noqa: PLC0415
    from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (  # noqa: PLC0415
        solid_to_kaolin_mesh,
    )

    vision_cone = vision_nbz(id_number)
    vision_cone_solid = vision_cone.toCompound().Solids()[0]
    dev = _resolve_device(device)
    verts, faces = solid_to_kaolin_mesh(vision_cone_solid, device=str(dev))
    return verts, faces


def mesh_intersection(
    verts1: torch.Tensor, faces1: torch.Tensor, vertices2: torch.Tensor, faces2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return the faces of mesh2 whose vertices all lie inside mesh1's volume.

    :param verts1: Vertices of the containing mesh / vision cone (N1 x 3)
    :param faces1: Faces of the containing mesh / vision cone (M1 x 3)
    :param vertices2: Vertices of the mesh to clip (N2 x 3)
    :param faces2: Faces of the mesh to clip (M2 x 3)
    :return: Vertices and faces of mesh2 that are fully inside mesh1.
    """
    import numpy as np  # noqa: PLC0415

    device = verts1.device
    dtype = verts1.dtype
    verts2_np = vertices2.detach().cpu().numpy()
    faces2_np = faces2.detach().cpu().numpy()

    face_mask = _face_inside_mask(verts1, faces1, vertices2, faces2)
    selected_faces = faces2_np[face_mask]

    if len(selected_faces) == 0:
        return torch.empty((0, 3), device=device, dtype=dtype), torch.empty((0, 3), device=device, dtype=faces1.dtype)

    used_idx = np.unique(selected_faces)
    remap = np.zeros(len(verts2_np), dtype=np.int64)
    remap[used_idx] = np.arange(len(used_idx))

    new_verts = torch.as_tensor(verts2_np[used_idx], device=device, dtype=dtype)
    new_faces = torch.as_tensor(remap[selected_faces], device=device, dtype=faces1.dtype)
    return new_verts, new_faces


def _face_inside_mask(cone_verts: torch.Tensor, cone_faces: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor):
    """Return a bool mask of shape (num_faces,) — True where all 3 vertices are inside the cone."""
    import numpy as np  # noqa: PLC0415
    import open3d as o3d  # noqa: PLC0415

    cone_o3d = o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(cone_verts.detach().cpu().numpy().astype(np.float32)),
        triangle_indices=o3d.core.Tensor(cone_faces.detach().cpu().numpy().astype(np.int32)),
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(cone_o3d)
    inside = scene.compute_occupancy(o3d.core.Tensor(verts.detach().cpu().numpy().astype(np.float32))).numpy().astype(bool)
    return inside[faces.detach().cpu().numpy()].all(axis=1)


def get_vision_cone_face_mask(
    cone_verts: torch.Tensor,
    cone_faces: torch.Tensor,
    mesh_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
) -> torch.Tensor:
    """Return a bool tensor of shape (num_faces,) — True for faces fully inside the vision cone."""
    mask_np = _face_inside_mask(cone_verts, cone_faces, mesh_verts, mesh_faces)
    return torch.as_tensor(mask_np, device=mesh_faces.device, dtype=torch.bool)


def render_vis_cone(
    id_number: int,
    out_path: str | Path = "vision_cone.glb",
    device: str | torch.device | None = None,
) -> Path:
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils import save_scene_glb, solid_rgb  # noqa: PLC0415

    verts, faces = get_vision_cone_mesh(id_number, device=device)
    print(f"Vision cone mesh: {len(verts)} vertices, {len(faces)} faces on {verts.device}")
    verts_np = verts.cpu().numpy()
    saved_path = save_scene_glb(
        out_path, [("vision_cone", verts_np, faces.cpu().numpy(), solid_rgb(len(verts_np), (0.9, 0.1, 0.9)))]
    )
    print(f"Saved to {saved_path}")
    return saved_path


def render_intersection(
    cone_verts: torch.Tensor,
    cone_faces: torch.Tensor,
    mesh_verts: torch.Tensor,
    mesh_faces: torch.Tensor,
    out_dir: str | Path,
    timestamp: str,
) -> tuple[Path, Path]:
    """Split mesh into selected (inside cone) and non-selected faces; save as two GLBs.

    Returns (selected_path, not_selected_path).
    """
    import numpy as np  # noqa: PLC0415
    import trimesh  # noqa: PLC0415

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verts_np = mesh_verts.detach().cpu().numpy()
    faces_np = mesh_faces.detach().cpu().numpy()

    mask = _face_inside_mask(cone_verts, cone_faces, mesh_verts, mesh_faces)

    def _save(path: Path, face_indices: "np.ndarray", rgb: tuple[int, int, int]) -> Path:
        used_idx = np.unique(face_indices)
        remap = np.zeros(len(verts_np), dtype=np.int64)
        remap[used_idx] = np.arange(len(used_idx))
        sub_verts = verts_np[used_idx]
        sub_faces = remap[face_indices]
        mesh = trimesh.Trimesh(
            vertices=np.asarray(sub_verts, dtype=np.float64), faces=np.asarray(sub_faces, dtype=np.int32), process=False
        )
        rgba_uint8 = np.tile(np.array([*rgb, 255], dtype=np.uint8), (len(sub_verts), 1))
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=rgba_uint8)
        scene = trimesh.Scene()
        scene.add_geometry(mesh, geom_name="mesh")
        scene.export(str(path))
        return path

    selected_faces = faces_np[mask]
    not_selected_faces = faces_np[~mask]

    sel_path = _save(out_dir / f"selected_{timestamp}.glb", selected_faces, (26, 230, 77))
    not_sel_path = _save(out_dir / f"not_selected_{timestamp}.glb", not_selected_faces, (26, 102, 230))

    print(
        f"Saved to {out_dir}\n"
        f"  selected (inside cone):  {mask.sum()} faces → {sel_path.name}\n"
        f"  not selected (outside):  {(~mask).sum()} faces → {not_sel_path.name}"
    )
    return sel_path, not_sel_path


if __name__ == "__main__":
    from datetime import datetime
    from typing import cast

    import trimesh

    _PLY = "/cluster/work/fuge/AISCO/temp_/6166b4e953/26_66948009_deep-thunder/mesh_shape/epoch_01400/slot_07_mesh.ply"
    mesh = cast(trimesh.Trimesh, trimesh.load(_PLY, force="mesh"))
    aero_verts = torch.as_tensor(mesh.vertices, dtype=torch.float32)
    aero_faces = torch.as_tensor(mesh.faces, dtype=torch.long)

    cone_verts, cone_faces = get_vision_cone_mesh(0, device="cpu")
    render_intersection(
        cone_verts,
        cone_faces,
        aero_verts,
        aero_faces,
        out_dir="/cluster/home/aalthaus/",
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
