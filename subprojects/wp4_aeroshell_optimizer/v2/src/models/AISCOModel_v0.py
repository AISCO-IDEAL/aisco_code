
"""AISCO SDF model (v0).

A 10-layer SDF MLP with:
  - Smooth activations (SiLU by default; GELU available via config)
  - Repeated concatenation of the (scaled) input features at chosen layers
  - Residual additions at chosen layers (with auto-projection when shapes differ)
  - Optional chunked forward pass to keep memory bounded on large coordinate
    grids (e.g. the connectedness loss evaluating a dense voxel grid).

Every architectural choice is exposed through the YAML config and forwarded as
``model_class(**model_config)`` from
``subprojects/wp4_aeroshell_optimizer/v2/src/utils/dynamic_loading.py``.

Latent-vector conditioning is not implemented here — see the ``concat_AISCOModel``
variant for that.
"""

from __future__ import annotations

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel


class _Snake(nn.Module):
    """Snake activation: f(x) = x + (1/alpha) * sin(alpha * x)^2.

    From Ziyin, Hartwig & Ueda (2020), "Neural Networks Fail to Learn Periodic
    Functions and How to Fix It". Adds periodic capacity (good for edges/creases
    in SDFs) on top of an identity backbone, so gradients remain well-behaved
    near zero — unlike pure sine/SIREN which we found unstable here.

    ``alpha`` is a per-channel learnable parameter when ``learnable=True``.
    Larger ``alpha`` -> higher frequency content. Typical init: 0.5-1.0 for
    smooth shapes, 5-30 for highly periodic data.
    """

    def __init__(self, num_features: int, alpha_init: float = 1.0, learnable: bool = True):
        super().__init__()
        alpha = torch.full((num_features,), float(alpha_init))
        if learnable:
            self.alpha = nn.Parameter(alpha)
        else:
            self.register_buffer("alpha", alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Broadcast alpha over leading dims; channel dim is last for our Linear stack.
        a = self.alpha
        # Avoid division by zero if alpha drifts to ~0 during training.
        safe_a = torch.where(a.abs() < 1e-4, torch.full_like(a, 1e-4), a)
        return x + (torch.sin(a * x) ** 2) / safe_a


def _build_activation(name: str | None, num_features: int, snake_alpha_init: float = 1.0) -> nn.Module:
    key = (name or "linear").lower()
    factories: dict[str, type[nn.Module]] = {
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "mish": nn.Mish,
        "softplus": nn.Softplus,
        "tanh": nn.Tanh,
        "linear": nn.Identity,
        "none": nn.Identity,
        "identity": nn.Identity,
        "": nn.Identity,
    }
    if key == "snake":
        return _Snake(num_features=num_features, alpha_init=snake_alpha_init, learnable=True)
    if key not in factories:
        raise ValueError(
            f"Unknown activation '{name}'. Supported: silu, gelu, relu, leakyrelu, elu, "
            "selu, mish, softplus, tanh, snake, linear."
        )
    return factories[key]()


class AISCOModel(BaseModel):
    """Fully config-driven SDF MLP (see module docstring for architecture)."""

    def __init__(
        self,
        config: dict | None = None,
        in_features: int = 3,
        out_features: int = 1,
        hidden_dims: list[int] | None = None,
        activations: list[str] | None = None,
        concat_input_at: list[int] | None = None,
        residual_at: list[int] | None = None,
        input_scale: float = 1000.0,
        output_scale: float = 1000.0,
        bias: bool = True,
        forward_chunk_size: int = 131072,
        snake_alpha_init: float = 1.0,
    ):
        """
        Args:
            config: Optional config dict forwarded to ``BaseModel``.
            in_features: Number of input coordinate features (default 3 for x, y, z).
            out_features: Number of output features (default 1 for signed distance).
            hidden_dims: Output width of each hidden ``Linear``. Defaults to the
                spec ``[256, 256, 256, 512, 512, 512, 512, 256, 128]`` (9 hidden
                layers + 1 output layer = 10 ``Linear`` layers total).
            activations: Activation name applied AFTER each ``Linear`` (length
                ``len(hidden_dims) + 1``). Recognised: ``silu``, ``gelu``, ``relu``,
                ``leakyrelu``, ``elu``, ``selu``, ``mish``, ``softplus``, ``tanh``,
                ``snake``, ``linear``. Defaults to 9 x ``silu`` + ``linear``.
            concat_input_at: 1-indexed layer numbers whose ``Linear`` input is
                concatenated with the (scaled) input features. Defaults to
                ``[3, 5, 8]``.
            residual_at: 1-indexed layer numbers where the layer's input (after any
                concat) is added back to its post-activation output. A learnable
                projection is added automatically when shapes differ. Defaults to
                ``[5, 8]``.
            input_scale: Divides the input (so ``x / input_scale`` is fed to the
                network). Default 1000.0 mirrors ``BaseModel``.
            output_scale: Multiplies the network output (e.g. mm rescale).
            bias: Whether ``Linear`` layers carry a bias.
            forward_chunk_size: When > 0 and the input batch dimension exceeds
                this value, the forward pass is computed in chunks along the
                first dim and concatenated. This keeps activation memory bounded
                for dense grid evaluations (e.g. the connectedness loss). Set
                to ``0`` to disable chunking.
            snake_alpha_init: Initial value for the per-channel learnable
                ``alpha`` of any ``snake`` activation in ``activations``. Higher
                values bias toward higher-frequency content. Ignored when no
                ``snake`` entry is used. Default 1.0 (smooth, edge-friendly).
        """
        super().__init__(config=config, overwrite_base_init=True)

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 512, 512, 512, 512, 256, 128]
        if activations is None:
            activations = ["silu"] * 9 + ["linear"]
        if concat_input_at is None:
            concat_input_at = [3, 5, 8]
        if residual_at is None:
            residual_at = [5, 8]

        n_linear = len(hidden_dims) + 1  # hidden layers + output layer
        if len(activations) != n_linear:
            raise ValueError(
                f"activations has length {len(activations)} but must equal "
                f"len(hidden_dims) + 1 = {n_linear}."
            )
        for idx in list(concat_input_at) + list(residual_at):
            if not 1 <= idx <= n_linear:
                raise ValueError(f"Layer index {idx} out of range [1, {n_linear}].")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.hidden_dims = list(hidden_dims)
        self.activations_cfg = list(activations)
        self.concat_input_at = {int(i) for i in concat_input_at}
        self.residual_at = {int(i) for i in residual_at}
        self.input_scale = float(input_scale)
        self.output_scale = float(output_scale)
        self.forward_chunk_size = int(forward_chunk_size)
        self.snake_alpha_init = float(snake_alpha_init)

        # mirror BaseModel attributes for get_model_info_dict / debug introspection
        self.in_dim = self.in_features
        self.hidden = self.hidden_dims[0] if self.hidden_dims else 0
        self.layers = n_linear

        out_dims = [*self.hidden_dims, self.out_features]

        self.linears = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.residual_projs = nn.ModuleDict()  # keyed by str(layer_idx_1based)

        prev_dim = self.in_features
        for layer_idx in range(1, n_linear + 1):
            in_dim = prev_dim + (self.in_features if layer_idx in self.concat_input_at else 0)
            out_dim = out_dims[layer_idx - 1]

            linear = nn.Linear(in_dim, out_dim, bias=bias)
            act_name = self.activations_cfg[layer_idx - 1]

            self.linears.append(linear)
            # Snake needs the per-layer width so its learnable alpha has the
            # right shape; other activations ignore num_features.
            self.acts.append(
                _build_activation(act_name, num_features=out_dim, snake_alpha_init=self.snake_alpha_init)
            )

            if layer_idx in self.residual_at and in_dim != out_dim:
                # auto-projection so the residual add is well-defined when shapes differ
                self.residual_projs[str(layer_idx)] = nn.Linear(in_dim, out_dim, bias=False)

            prev_dim = out_dim

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x / self.input_scale
        h = x_in
        for i, (linear, act) in enumerate(zip(self.linears, self.acts, strict=True)):
            layer_idx = i + 1  # 1-indexed
            if layer_idx in self.concat_input_at:
                h = torch.cat([h, x_in], dim=-1)
            pre = h
            h = act(linear(h))
            if layer_idx in self.residual_at:
                key = str(layer_idx)
                try:
                    proj = self.residual_projs[key]
                except KeyError:
                    proj = None
                skip = proj(pre) if proj is not None else pre
                h = h + skip
        return h * self.output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chunk along the leading (batch/point) dim when the input is large to
        # bound peak activation memory. This rescues large-grid evaluations
        # (e.g. ConnectednessV1's full voxel grid) from CUDA OOM without
        # changing numerics for normal-sized batches.
        x = x / 1000.0  # Scale input to meters

        chunk = self.forward_chunk_size
        if chunk <= 0 or x.dim() == 0 or x.shape[0] <= chunk:
            return self._forward_single(x)
        outs = [self._forward_single(x[i : i + chunk]) for i in range(0, x.shape[0], chunk)]
        return torch.cat(outs, dim=0)*1000.0  # Scale output back to mm

    def get_model_info_dict(self) -> dict:
        return {
            "Model_name": "AISCOModel",
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hidden_dims": self.hidden_dims,
            "activations": self.activations_cfg,
            "concat_input_at": sorted(self.concat_input_at),
            "residual_at": sorted(self.residual_at),
            "input_scale": self.input_scale,
            "output_scale": self.output_scale,
            "forward_chunk_size": self.forward_chunk_size,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }

    def debug_info(self) -> None:
        print("===================================")
        print("AISCOModel Debug Info:")
        print(self)
        print(f"Total parameters: {sum(p.numel() for p in self.parameters())}")
        for i, linear in enumerate(self.linears):
            idx = i + 1
            tags = []
            if idx in self.concat_input_at:
                tags.append("concat-input")
            if idx in self.residual_at:
                tags.append("residual")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            print(f" Layer {idx}: {linear} act={self.activations_cfg[i]}{tag_str}")
        device = next(self.parameters()).device
        print(f" Device: {device}")
        print("===================================")
