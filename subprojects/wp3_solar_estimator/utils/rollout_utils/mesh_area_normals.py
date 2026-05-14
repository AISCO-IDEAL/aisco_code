import torch


def compute_face_areas_and_normals(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute per-face areas and unit normals from a triangle mesh.

    Args:
        vertices: (N, 3) vertex positions in mm
        faces: (F, 3) face vertex indices
        epsilon: numerical stability clamp for normal normalisation

    Returns:
        face_cross: (F, 3) cross products (unnormalised normals, magnitude = 2 * area)
        face_area_m2: (F,) face areas in m²
        face_normals_unit: (F, 3) unit outward normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    edge_a = v1 - v0
    edge_b = v2 - v0
    face_cross = torch.cross(edge_a, edge_b, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
    face_area_mm2 = 0.5 * face_cross_norm
    face_area_m2 = face_area_mm2 * 1e-6
    face_normals_unit = face_cross / face_cross_norm.unsqueeze(-1).clamp(min=epsilon)
    return face_cross, face_area_m2, face_normals_unit
