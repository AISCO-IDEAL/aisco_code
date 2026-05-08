from __future__ import annotations

import numpy as np


def face_scalars_to_vertex(
    face_values: np.ndarray,
    faces: np.ndarray,
    verts: np.ndarray,
) -> np.ndarray:
    """Area-weighted average of per-face scalar values at each vertex.

    Args:
        face_values: Per-face scalar values, shape (F,).
        faces: Face index array, shape (F, 3).
        verts: Vertex position array, shape (V, 3). Used only for area computation.

    Returns:
        Per-vertex scalar values, shape (V,) as float32.

    Raises:
        ValueError: If len(face_values) != len(faces).
    """
    face_values = np.asarray(face_values, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    verts = np.asarray(verts, dtype=np.float64)

    if len(face_values) != len(faces):
        raise ValueError(
            f"face_values length ({len(face_values)}) does not match faces length ({len(faces)})."
        )

    n_verts = len(verts)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).astype(np.float32)

    weighted_sum = np.zeros(n_verts, dtype=np.float64)
    area_sum = np.zeros(n_verts, dtype=np.float64)
    weighted_vals = face_values * face_areas
    for i in range(3):
        np.add.at(weighted_sum, faces[:, i], weighted_vals)
        np.add.at(area_sum, faces[:, i], face_areas)

    safe_area = np.where(area_sum > 0.0, area_sum, 1.0)
    return (weighted_sum / safe_area).astype(np.float32)
