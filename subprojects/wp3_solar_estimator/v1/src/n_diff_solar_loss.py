from collections.abc import Callable

import numpy as np
import torch
from skimage import measure

from subprojects.wp3_solar_estimator.v1.src.kaolin_utils import get_diff_mesh


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


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute unit normals for each triangular face using PyTorch (kept on device).

    Args:
        vertices: (V, 3) float tensor of vertex positions (device can be cuda/cpu).
        faces: (F, 3) long/int tensor of vertex indices per face.
        eps: small value to avoid division by zero.

    Returns:
        (F, 3) float tensor of normalized face normals.
    """
    v = vertices
    f = faces.to(dtype=torch.long, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    n = torch.cross(e1, e2, dim=1)
    norm = torch.linalg.norm(n, dim=1, keepdim=True).clamp_min(eps)
    n_unit = n / norm
    return n_unit


def compute_triangle_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute areas for each triangular face using PyTorch (kept on device).

    Args:
        vertices: (V, 3) float tensor of vertex positions.
        faces: (F, 3) long/int tensor of vertex indices per face.

    Returns:
        (F,) float tensor of triangle areas.
    """
    v = vertices
    f = faces.to(dtype=torch.long, copy=False)

    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross_prod = torch.cross(e1, e2, dim=1)
    return 0.5 * torch.linalg.norm(cross_prod, dim=1)


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
    return -float(total_power.item())*1e-6


@torch.no_grad()
def extract_mesh_from_sdf(
    sdf_model,
    z: torch.Tensor | None = None,
    resolution: int = 128,
    bounds: tuple[float, float] = (-1.0, 1.0),
    level: float = 0.0,
    device: torch.device = torch.device("cuda"),
    chunk: int = 200_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Non-diff: sample sdf on a grid and run marching cubes.
    Returns:
      verts: (V, 3) float32 torch tensor on `device`
      faces: (F, 3) int64 torch tensor on `device`
    """
    lo, hi = bounds
    xs = torch.linspace(lo, hi, resolution, device=device)
    ys = torch.linspace(lo, hi, resolution, device=device)
    zs = torch.linspace(lo, hi, resolution, device=device)
    grid = torch.stack(torch.meshgrid(xs, ys, zs, indexing="ij"), dim=-1)  # (R,R,R,3)
    pts = grid.reshape(-1, 3)

    # Evaluate SDF in chunks to avoid OOM
    sdf_vals = []
    for i in range(0, pts.shape[0], chunk):
        s = sdf_model(pts[i : i + chunk]) if z is None else sdf_model(pts[i : i + chunk], z)
        s = s.view(-1).detach()
        sdf_vals.append(s)
    sdf = torch.cat(sdf_vals, dim=0).reshape(resolution, resolution, resolution)

    sdf_np = sdf.float().cpu().numpy()

    # spacing tells skimage how to map voxel coords to world coords
    spacing = ((hi - lo) / (resolution - 1),) * 3

    # skimage outputs verts in voxel-index space scaled by spacing, with origin at (0,0,0).
    # We shift by `lo` to put them in world coords.
    verts_np, faces_np, _, _ = measure.marching_cubes(
        volume=sdf_np, level=level, spacing=spacing
    )
    verts_np = verts_np + np.array([lo, lo, lo], dtype=np.float32)

    verts = torch.from_numpy(verts_np.astype(np.float32)).to(device)
    faces = torch.from_numpy(faces_np.astype(np.int64)).to(device)
    return verts, faces


def meshsdf_backward_step(
    sdf_model,
    z: torch.Tensor | None,
    verts: torch.Tensor,
    faces: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """
    Computes the mesh loss and backpropagates into z and sdf_model params using MeshSDF gradients.

    IMPORTANT:
      - verts/faces come from non-diff Marching Cubes (they are treated as constants).
      - We create a leaf `v` that the loss can differentiate w.r.t., then convert dL/dv
        into dL/ds using -<dL/dv, ∇s>.
    Returns:
      loss (scalar tensor)
    """
    device = verts.device

    # 1) Make a differentiable copy of vertices for the loss (not connected to MC / grid sampling)
    v = verts.detach().clone().requires_grad_(True)  # (V,3)

    # 2) Compute any differentiable mesh-based loss
    loss = loss_fn(v, faces)

    # 3) Upstream gradient: dL/dv
    (dldv,) = torch.autograd.grad(loss, v, create_graph=False, retain_graph=True)

    # 4) Evaluate SDF at vertex locations (this connects to z and model params)
    s = sdf_model(v, z).view(-1)  # (V,)

    # 5) Compute spatial gradient ∇s(v) (surface normal for a true SDF)
    (grad_xyz,) = torch.autograd.grad(
        s,
        v,
        grad_outputs=torch.ones_like(s),
        create_graph=True,   # keep graph so gradients can flow to model params / z through s
        retain_graph=True,
    )  # (V,3)

    # Normalize to get unit normals; add eps for stability
    eps = 1e-12
    n = grad_xyz / (grad_xyz.norm(dim=-1, keepdim=True) + eps)  # (V,3)

    # 6) MeshSDF conversion: dL/ds(v) = - <dL/dv, n>
    dlds = -(dldv * n).sum(dim=-1)  # (V,)

    # 7) Surrogate scalar that yields correct grads for z and model params:
    #    d/dθ sum_v (dL/ds(v) * s(v;θ)) = sum_v dL/ds(v) * ∂s/∂θ
    # Detach dL_ds to avoid unwanted second-order terms
    surrogate = (dlds.detach() * s).sum()

    # 8) Backprop into z and model params
    surrogate.backward()

    return loss


# ---------------------------
# Example loss function
# ---------------------------
def solar_mesh_loss(verts: torch.Tensor, faces: torch.Tensor, area_budget: float = 6e6) -> torch.Tensor:
    """Calculates the loss like the n_diff_solar_loss, but for a given mesh generated with extract_mesh_from_sdf. This function is differentiable w.r.t. the vertices."""

    # calculate normal vectors for each face
    face_normals = compute_face_normals(verts, faces)

    # Prepare and normalize sun ray vector
    s = torch.tensor([-1.0, 0.0, 1.0], device=verts.device, dtype=face_normals.dtype)
    s = s / (torch.linalg.norm(s) + 1e-12)

    # calculate dot product between face normals and sun ray vector (clamp back-facing to zero)
    dot_products = torch.clamp((face_normals * s).sum(dim=1), min=0.0)

    # calculate area of each triangle face
    triangle_areas = compute_triangle_areas(verts, faces)

    # calculate the total power received by the best faces that sum up to area_budget
    sorted_vals, sorted_indices = torch.sort(dot_products, descending=True)
    sorted_areas = triangle_areas[sorted_indices]
    cum_area = torch.cumsum(sorted_areas, dim=0)
    select_mask = cum_area <= area_budget
    selected_vals = sorted_vals[select_mask]
    selected_areas = sorted_areas[select_mask]
    total_power = torch.sum(selected_vals * selected_areas)

    return -total_power * 1e-6  # return negative for minimization   


# complete loss function 

def solar_loss(
    model,
    area_budget: float = 6e6,
    device="cpu",
) -> torch.Tensor:
    """Compute models loss by using the following functions:
       - extract_mesh_from_sdf
       - meshsdf_backward_step
       - solar_mesh_loss"""
    
    z = torch.zeros(64, device=device, requires_grad=True)  # latent you want to optimize
    verts, faces = extract_mesh_from_sdf(
        sdf_model=model,
        z=z,
        resolution=96,
        bounds=(-1.2, 1.2),
        level=0.0,
        device=torch.device(device)
    )

    # 2) Zero gradients
    model.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad.zero_()

    # 3) Backward: compute loss and backprop into sdf_model params and z
    def _loss_fn(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return solar_mesh_loss(v, f, area_budget)
    loss = meshsdf_backward_step(model, z, verts, faces, _loss_fn)

    return loss
