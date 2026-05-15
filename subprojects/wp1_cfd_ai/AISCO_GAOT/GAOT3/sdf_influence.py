from __future__ import annotations

from collections.abc import Mapping

import torch


def _flatten_sdf_output(values: torch.Tensor, n_points: int) -> torch.Tensor:
    if values.dim() > 1 and values.size(-1) == 1:
        values = values.squeeze(-1)
    if values.numel() != n_points:
        raise ValueError(f"Expected one SDF value per point, got output shape {tuple(values.shape)} for {n_points} points.")
    return values.reshape(n_points)


def compute_sdf_parameter_influence(
    sdf_model: torch.nn.Module,
    vertices: torch.Tensor,
    param_grads_by_name: Mapping[str, torch.Tensor | None],
    *,
    learning_rate: float,
    chunk_size: int = 8192,
) -> dict[str, torch.Tensor]:
    """Evaluate SDF sensitivity to a parameter-space gradient at mesh vertices.

    Returns four per-vertex fields:
    - jvp/raw_gradient: first-order influence along the raw gradient direction.
    - jvp/update_direction: first-order influence along the optimizer update direction.
    - finite/raw_gradient: finite delta for theta + lr * grad.
    - finite/update_direction: finite delta for theta - lr * grad.

    The live model parameters and .grad buffers are never modified.
    """
    named_params = dict(sdf_model.named_parameters())
    if not named_params:
        zeros = torch.zeros(vertices.shape[0], device=vertices.device, dtype=vertices.dtype)
        return {
            "jvp/raw_gradient": zeros,
            "jvp/update_direction": -zeros,
            "finite/raw_gradient": zeros,
            "finite/update_direction": zeros,
        }

    first_param = next(iter(named_params.values()))
    eval_vertices = vertices.detach().to(device=first_param.device, dtype=first_param.dtype)
    chunk_size = max(1, int(chunk_size))
    lr = float(learning_rate)

    base_params: dict[str, torch.Tensor] = {}
    raw_tangent: dict[str, torch.Tensor] = {}
    for name, param in named_params.items():
        base_param = param.detach()
        grad = param_grads_by_name.get(name)
        if grad is None:
            tangent = torch.zeros_like(base_param)
        else:
            tangent = grad.detach().to(device=base_param.device, dtype=base_param.dtype)
        base_params[name] = base_param
        raw_tangent[name] = tangent

    raw_params = {name: param + lr * raw_tangent[name] for name, param in base_params.items()}
    update_params = {name: param - lr * raw_tangent[name] for name, param in base_params.items()}

    def _forward_with(params: dict[str, torch.Tensor], points: torch.Tensor) -> torch.Tensor:
        return _flatten_sdf_output(torch.func.functional_call(sdf_model, params, (points,)), points.shape[0])

    jvp_raw_chunks: list[torch.Tensor] = []
    finite_raw_chunks: list[torch.Tensor] = []
    finite_update_chunks: list[torch.Tensor] = []
    for start in range(0, eval_vertices.shape[0], chunk_size):
        points = eval_vertices[start : start + chunk_size]

        def _chunk_forward(params: dict[str, torch.Tensor]) -> torch.Tensor:
            return _forward_with(params, points)

        _base_values, jvp_raw = torch.func.jvp(_chunk_forward, (base_params,), (raw_tangent,))
        jvp_raw_chunks.append(jvp_raw.detach())

        with torch.no_grad():
            base_values = _forward_with(base_params, points)
            finite_raw_chunks.append((_forward_with(raw_params, points) - base_values).detach())
            finite_update_chunks.append((_forward_with(update_params, points) - base_values).detach())

    jvp_raw = torch.cat(jvp_raw_chunks, dim=0)
    finite_raw = torch.cat(finite_raw_chunks, dim=0)
    finite_update = torch.cat(finite_update_chunks, dim=0)
    return {
        "jvp/raw_gradient": jvp_raw,
        "jvp/update_direction": -jvp_raw,
        "finite/raw_gradient": finite_raw,
        "finite/update_direction": finite_update,
    }
