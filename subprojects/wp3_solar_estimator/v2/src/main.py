from collections.abc import Callable

import torch

from subprojects.wp3_solar_estimator.v2.src.geometric_utils import compute_face_normals, compute_triangle_areas
from subprojects.wp3_solar_estimator.v2.src.kaolin_utils import get_diff_mesh


def meshsdf_backward_step(
    sdf_model,
    z: torch.Tensor | None,
    verts: torch.Tensor,
    faces: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    v = verts.detach().clone().requires_grad_(True)
    loss = loss_fn(v, faces)

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

    return total_power * 1e-6


# complete loss function


def solar_loss(
    model,
    area_budget: float = 6e6,
    device="cpu",
) -> torch.Tensor:
    z = torch.zeros(64, device=device, requires_grad=True)  # latent you want to optimize

    verts, faces = get_diff_mesh(model)

    # 2) Zero gradients
    model.zero_grad(set_to_none=True)
    if z.grad is not None:
        z.grad.zero_()

    # 3) Backward: compute loss and backprop into sdf_model params and z
    def _loss_fn(v: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return solar_mesh_loss(v, f, area_budget)

    loss = meshsdf_backward_step(model, z, verts, faces, _loss_fn)

    return loss
