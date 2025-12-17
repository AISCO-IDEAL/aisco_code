from collections.abc import Callable

import numpy as np
import torch
from skimage import measure
import trimesh
from pathlib import Path

from subprojects.wp3_solar_estimator.v1.src.kaolin_utils import get_diff_mesh
from subprojects.wp3_solar_estimator.v1.src.geometric_utils import compute_face_normals, compute_triangle_areas


def _to_tensor(x, device=None, dtype=None) -> torch.Tensor:
    """Convert input to a torch.Tensor on the desired device without unnecessary transfers.
    Accepts torch tensors, NumPy arrays, or array-like.
    """
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        if device is not None and t.device.type != device:
            t = t.to(device)
        return t
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if dtype is not None:
            t = t.to(dtype)
        if device is not None:
            t = t.to(device)
        return t
    # Fallback for array-like
    t = torch.as_tensor(x, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def n_diff_solar_loss(model, area_budget: float = 6e6, sun_ray_vector=None, use_amp: bool = True) -> float:
    """Compute total received power under an area budget by greedily selecting faces (PyTorch, GPU-accelerated).

    Args:
        model: Source object passed to get_diff_mesh to obtain (vertices, faces).
        area_budget: Maximum total area to select (e.g., in mm^2).
        sun_ray_vector: Optional (3,) array-like representing sun direction. If None, a default is used.
        use_amp: Use mixed precision on CUDA for higher throughput.

    Returns:
        Scalar float of total power from selected faces.
    """
    vertices, faces = get_diff_mesh(model)

    # Ensure tensors and keep device/dtype from vertices
    device = (
        vertices.device
        if isinstance(vertices, torch.Tensor)
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )

    # Convert to tensors on device
    vt = _to_tensor(vertices, device=device)
    ft = _to_tensor(faces, device=device, dtype=torch.long)

    # Autocast only when on CUDA
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else torch.cuda.amp.autocast(enabled=False)
    )

    with amp_ctx:
        # calculate normal vectors for each face
        face_normals = compute_face_normals(vt, ft)

        # Prepare and normalize sun ray vector
        if sun_ray_vector is None:
            s = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=face_normals.dtype)
        else:
            s = _to_tensor(sun_ray_vector, device=device, dtype=face_normals.dtype)
        s = s / (torch.linalg.norm(s) + 1e-12)

        # calculate dot product between face normals and sun ray vector (clamp back-facing to zero)
        dot_products = torch.clamp((face_normals * s).sum(dim=1), min=0.0)

        # Early filter to avoid sorting negatives
        pos_mask = dot_products > 0
        if pos_mask.any():
            dot_products = dot_products[pos_mask]
            ft_pos = ft[pos_mask]
        else:
            return 0.0

        # calculate area of each triangle face
        triangle_areas = compute_triangle_areas(vt, ft_pos)

        # Greedy selection by highest dot product subject to area budget (vectorized)
        sorted_vals, sorted_indices = torch.sort(dot_products, descending=True)
        sorted_areas = triangle_areas[sorted_indices]
        cum_area = torch.cumsum(sorted_areas, dim=0)
        select_mask = cum_area <= area_budget

        if not torch.any(select_mask):
            return 0.0

        selected_vals = sorted_vals[select_mask]
        selected_areas = sorted_areas[select_mask]

        # calculate the total power received by the selected faces
        total_power = torch.sum(selected_vals * selected_areas)

    # Return host float without transferring large tensors
    return -float(total_power.item()) * 1e-6
    