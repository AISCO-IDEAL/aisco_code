"""Solar loss computation for mesh SDF models.
Functions in this script:
- solar_loss: Computes the solar loss, calls the folloing two functions:
    - solar_mesh_loss: Calculates the solar exposure loss using the dot-product-Approach.
    - meshsdf_backward_step: performs a backward Step using the MeshSDF-Approach.
"""

from collections.abc import Callable

import torch

from subprojects.wp3_solar_estimator.v2.src.geometric_utils import compute_face_normals, compute_triangle_areas
from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh


def meshsdf_backward_step(
    sdf_model,
    verts: torch.Tensor,
    faces: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    area_budget: float = 6e6,
    device="cpu",
) -> torch.Tensor:
    v = verts.detach().clone().requires_grad_(True)
    loss = loss_fn(v, faces, area_budget, device)
    (dldv,) = torch.autograd.grad(loss, v, create_graph=False, retain_graph=True)
    s = sdf_model(v).view(-1)
    (grad_xyz,) = torch.autograd.grad(
        s,
        v,
        grad_outputs=torch.ones_like(s),
        create_graph=True,
        retain_graph=True,
    )
    eps = 1e-12
    n = grad_xyz / (grad_xyz.norm(dim=-1, keepdim=True) + eps)
    dlds = -(dldv * n).sum(dim=-1)
    surrogate = (dlds.detach() * s).sum()
    surrogate.backward()
    return loss


def solar_mesh_loss(verts: torch.Tensor, faces: torch.Tensor, area_budget: float = 6e6, device="cpu") -> torch.Tensor:
    face_normals = compute_face_normals(verts, faces)

    s = torch.tensor([-1.0, 0.0, 1.0], dtype=face_normals.dtype, device=device)
    s = s / (torch.linalg.norm(s) + 1e-12)

    dot_products = torch.clamp((face_normals * s).sum(dim=1), min=0.0)

    triangle_areas = compute_triangle_areas(verts, faces)

    sorted_vals, sorted_indices = torch.sort(dot_products, descending=True)
    sorted_areas = triangle_areas[sorted_indices]
    cum_area = torch.cumsum(sorted_areas, dim=0)
    select_mask = cum_area <= area_budget
    selected_vals = sorted_vals[select_mask]
    selected_areas = sorted_areas[select_mask]
    total_power = torch.sum(selected_vals * selected_areas)

    return total_power * 1e-6


def solar_loss(model, area_budget: float = 6e6, device="cpu") -> torch.Tensor:
    verts, faces = get_diff_mesh(model)
    model.zero_grad(set_to_none=True)

    loss = meshsdf_backward_step(model, verts, faces, solar_mesh_loss, area_budget=area_budget, device=device)

    return loss
