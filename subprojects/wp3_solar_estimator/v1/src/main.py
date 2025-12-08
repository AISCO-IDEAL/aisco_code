import numpy as np

from subprojects.wp3_solar_estimator.v1.src.kaolin_utils import get_diff_mesh


def _to_numpy(x) -> np.ndarray:
    """Convert torch Tensor or array-like to NumPy array without importing torch."""
    # Detach from graph if tensor-like
    if hasattr(x, "detach"):
        x = x.detach()
    # Move to CPU if tensor-like
    if hasattr(x, "cpu"):
        x = x.cpu()
    # Convert to numpy if available
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def compute_face_normals(vertices, faces, eps: float = 1e-12) -> np.ndarray:
    """
    Compute unit normals for each triangular face using NumPy.

    Args:
        vertices: (V, 3) float array of vertex positions.
        faces: (F, 3) int array of vertex indices per face.
        eps: small value to avoid division by zero.

    Returns:
        (F, 3) float array of normalized face normals.
    """
    v = _to_numpy(vertices)
    f = _to_numpy(faces).astype(np.int64, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    n = np.cross(e1, e2)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.clip(norm, eps, None)
    n_unit = n / norm
    return n_unit


def compute_triangle_areas(vertices, faces) -> np.ndarray:
    """Compute areas for each triangular face using NumPy.

    Args:
        vertices: (V, 3) float array of vertex positions.
        faces: (F, 3) int array of vertex indices per face.

    Returns:
        (F,) float array of triangle areas.
    """
    v = _to_numpy(vertices)
    f = _to_numpy(faces).astype(np.int64, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross_prod = np.cross(e1, e2)
    return 0.5 * np.linalg.norm(cross_prod, axis=1)


def main_loop(model, area_budget: float = 6e6, sun_ray_vector=None) -> float:
    """Compute total received power under an area budget by greedily selecting faces (NumPy).

    Args:
        model: Source object passed to get_diff_mesh to obtain (vertices, faces).
        area_budget: Maximum total area to select (e.g., in mm^2).
        sun_ray_vector: Optional (3,) array-like representing sun direction. If None, a default is used.

    Returns:
        Scalar float (or NumPy scalar) of total power from selected faces.
    """
    vertices, faces = get_diff_mesh(model)
    v = _to_numpy(vertices)
    f = _to_numpy(faces).astype(np.int64, copy=False)

    # calculate normal vectors for each face
    face_normals = compute_face_normals(v, f)

    # Prepare and normalize sun ray vector
    if sun_ray_vector is None:
        s = np.array([-1.0, 0.0, 1.0], dtype=v.dtype if getattr(v.dtype, "kind", "f") == "f" else float)
    else:
        s = _to_numpy(sun_ray_vector).astype(float, copy=False)
    s_norm = np.linalg.norm(s)
    if s_norm > 0:
        s = s / s_norm

    # calculate dot product between face normals and sun ray vector (clamp back-facing to zero)
    dot_products = np.clip(face_normals @ s, 0.0, None)

    # calculate area of each triangle face
    triangle_areas = compute_triangle_areas(v, f)

    # Greedy selection by highest dot product subject to area budget (vectorized)
    sorted_indices = np.argsort(dot_products)[::-1]
    sorted_areas = triangle_areas[sorted_indices]
    cum_area = np.cumsum(sorted_areas)
    select_mask = cum_area <= area_budget
    selected_indices = sorted_indices[select_mask]

    if selected_indices.size == 0:
        return 0.0

    # calculate the total power received by the selected faces
    total_power = np.sum(dot_products[selected_indices] * triangle_areas[selected_indices])

    return float(total_power)
