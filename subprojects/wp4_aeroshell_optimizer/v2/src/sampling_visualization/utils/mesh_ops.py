from __future__ import annotations

import numpy as np


def face_scalars_to_vertex(
    face_values: np.ndarray,
    faces: np.ndarray,
    verts: np.ndarray,
) -> np.ndarray:
    """
    Area-weighted average of per-face scalar values at each vertex.
    Very useful for smoothly interpolating face-based metrics like error or confidence for visualization.
    For hard per-face colors, use face_rgb_to_flat_mesh instead.

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


def face_rgb_to_flat_mesh(
    face_rgb: np.ndarray,
    faces: np.ndarray,
    verts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Duplicate vertices so each face gets its own constant (flat) color.

    GPUs interpolate per-vertex colors across faces, so true flat shading
    requires each face to own 3 private vertex copies all set to the same color.
    The returned arrays drop directly into save_mesh_ply / render_mesh.

    Args:
        face_rgb: Per-face RGB colors, shape (F, 3), float32 in [0, 1].
        faces: Face index array, shape (F, 3).
        verts: Vertex position array, shape (V, 3).

    Returns:
        (new_verts (F*3, 3), new_faces (F, 3), vertex_rgb (F*3, 3)) — each
        group of 3 consecutive vertices belongs to one face and shares its color.
    """
    face_rgb = np.asarray(face_rgb, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    verts = np.asarray(verts, dtype=np.float64)

    new_verts = verts[faces.reshape(-1)]  # (F*3, 3)
    new_faces = np.arange(len(faces) * 3, dtype=np.int32).reshape(-1, 3)  # (F, 3)
    vertex_rgb = np.repeat(face_rgb, 3, axis=0)  # (F*3, 3)

    return new_verts, new_faces, vertex_rgb
