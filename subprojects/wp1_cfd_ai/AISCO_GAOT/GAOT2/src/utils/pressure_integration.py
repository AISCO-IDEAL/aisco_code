import torch
from torch import Tensor


def compute_vertex_areas(pos: Tensor, face: Tensor) -> Tensor:
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    face_areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)

    vertex_areas = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    third = face_areas / 3.0
    vertex_areas.scatter_add_(0, face[0], third)
    vertex_areas.scatter_add_(0, face[1], third)
    vertex_areas.scatter_add_(0, face[2], third)
    return vertex_areas


def compute_face_normals(pos: Tensor, face: Tensor) -> Tensor:
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return cross / (cross.norm(dim=-1, keepdim=True) + 1e-12)


def pressure_to_drag(
    pred_pressure: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int = 0,
) -> Tensor:
    p = pred_pressure.squeeze(-1) if pred_pressure.dim() > 1 else pred_pressure

    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    # 1. Compute unnormalized face normal vectors
    area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F, 3]

    # 2. ROBUST ORIENTATION: Match the dataset's outward normals
    outward_n = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3.0
    dot_prod = (area_normals * outward_n).sum(dim=-1, keepdim=True)

    # If dot product is negative, our normal is pointing inward. Flip it.
    area_normals = torch.where(dot_prod < 0, -area_normals, area_normals)

    # 3. Integrate: Pressure pushes INWARD, so we use -p_face
    p_face = (p[face[0]] + p[face[1]] + p[face[2]]) / 3.0
    n_dir_area = area_normals[:, drag_direction]

    drag = (-p_face * n_dir_area).sum()

    return drag


def pressure_shear_to_drag(
    pred_pressure: Tensor,
    pred_shear: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int = 0,
) -> Tensor:
    p = pred_pressure.squeeze(-1) if pred_pressure.dim() > 1 else pred_pressure

    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    # 1. Compute unnormalized face normal vectors
    area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F, 3]

    # 2. ROBUST ORIENTATION: Match the dataset's outward normals
    outward_n = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3.0
    dot_prod = (area_normals * outward_n).sum(dim=-1, keepdim=True)
    area_normals = torch.where(dot_prod < 0, -area_normals, area_normals)

    # 3. --- PRESSURE DRAG ---
    p_face = (p[face[0]] + p[face[1]] + p[face[2]]) / 3.0
    n_dir_area = area_normals[:, drag_direction]

    # Pressure pushes INWARD, so we use -p_face
    pressure_drag = (-p_face * n_dir_area).sum()

    # 4. --- FRICTION DRAG ---
    # OpenFOAM's wallShearStress is already the exact force vector (kinematic).
    # We just average it to the face and multiply by the exact area!
    tau_face = (pred_shear[face[0]] + pred_shear[face[1]] + pred_shear[face[2]]) / 3.0
    face_areas = area_normals.norm(dim=-1)  # [F]

    tau_dir = tau_face[:, drag_direction]
    friction_drag = (-tau_dir * face_areas).sum()

    return pressure_drag + friction_drag


def drag_to_cd(
    drag: Tensor,
    rho: float = 1.225,
    u_inf: float = 38.889,
    A_ref: float = 2.16,
) -> Tensor:
    q = 0.5 * rho * u_inf**2 * A_ref
    return drag / q
