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
        faces: (M, 3) triangle face indices
        epsilon: numerical stability floor for normal normalisation

    Returns:
        face_area_mm2: (M,) face areas in mm²
        face_area_m2: (M,) face areas in m²
        face_normals_unit: (M, 3) unit outward normals
    """
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
    face_area_mm2 = 0.5 * face_cross_norm
    face_area_m2 = face_area_mm2 * 1e-6
    face_normals_unit = face_cross / face_cross_norm.clamp_min(epsilon).unsqueeze(-1)
    return face_area_mm2, face_area_m2, face_normals_unit
