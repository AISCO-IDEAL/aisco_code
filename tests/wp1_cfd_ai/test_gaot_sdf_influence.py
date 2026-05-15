from __future__ import annotations

import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.sdf_influence import compute_sdf_parameter_influence


class TinySDF(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.bias = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return (points * self.weight).sum(dim=-1, keepdim=True) + self.bias


def test_compute_sdf_parameter_influence_matches_linear_model() -> None:
    model = TinySDF()
    vertices = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.5, 2.0]])
    param_grads = {
        "weight": torch.tensor([0.1, 0.2, 0.3]),
        "bias": torch.tensor(0.4),
    }

    fields = compute_sdf_parameter_influence(
        model,
        vertices,
        param_grads,
        learning_rate=0.5,
        chunk_size=1,
    )

    expected_jvp = vertices @ param_grads["weight"] + param_grads["bias"]
    torch.testing.assert_close(fields["jvp/raw_gradient"], expected_jvp)
    torch.testing.assert_close(fields["jvp/update_direction"], -expected_jvp)
    torch.testing.assert_close(fields["finite/raw_gradient"], 0.5 * expected_jvp)
    torch.testing.assert_close(fields["finite/update_direction"], -0.5 * expected_jvp)


def test_compute_sdf_parameter_influence_does_not_mutate_params_or_grads() -> None:
    model = TinySDF()
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    before_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    before_grads = {name: param.grad.detach().clone() for name, param in model.named_parameters()}

    compute_sdf_parameter_influence(
        model,
        torch.tensor([[1.0, 0.0, 0.0]]),
        {"weight": torch.tensor([0.5, 0.0, 0.0]), "bias": torch.tensor(0.25)},
        learning_rate=0.1,
        chunk_size=4,
    )

    for name, param in model.named_parameters():
        torch.testing.assert_close(param, before_params[name])
        assert param.grad is not None
        torch.testing.assert_close(param.grad, before_grads[name])
