
"""AISCO SDF model (v0).

A 10-Linear-layer MLP that maps a 3-D point ``(x, y, z)`` in mm to a single
signed-distance scalar. Every architectural knob is exposed through the YAML
config and forwarded as ``model_class(**model_config)`` from
``subprojects/wp4_aeroshell_optimizer/v2/src/utils/dynamic_loading.py``; the
default winning configuration lives in
``subprojects/wp4_aeroshell_optimizer/v2/configs/models/AISCOModel_v0.yaml``.

Conceptual data flow (one ``forward(x)`` call):

    x  --[/ input_scale]-->  x_norm
       --[Fourier encoding]-->  x_enc          (gamma(x) of dim 3*(1+2L))
       --[Linear stack]-->  h                  (10 layers, see below)
       --[* output_scale]-->  f(x)             (signed distance in mm)

Inside the Linear stack:
  * Each layer is ``act(Linear(h))``.
  * At every layer in ``concat_input_at`` the encoded input ``x_enc`` is
    concatenated onto the layer's input. This restores positional information
    that would otherwise wash out in deep layers (DeepSDF / SIREN trick).
  * At every layer in ``residual_at`` the layer's input (after any concat) is
    added back to its post-activation output; a learnable projection is
    auto-built when shapes differ.
  * ``forward()`` chunks along the leading dim when ``forward_chunk_size > 0``
    and the batch exceeds it, bounding peak activation memory for dense-grid
    SDF evaluations (e.g. the connectedness loss).

Activation choice matters for SDFs: pure-smooth profiles (all-SiLU) bias
toward ellipsoidal blobs; pure piecewise-linear profiles produce cardboard
SDFs. The winning ``profile C`` interleaves SiLU with LeakyReLU mid-stack to
get smooth gradients *and* enough piecewise-linear capacity to carve crisp
creases (e.g. bunny ear ridges). The factory below therefore only exposes the
three activations actually used by the winning config: ``silu``, ``leakyrelu``,
``linear``.

Latent-vector conditioning is not implemented here — see the
``concat_AISCOModel`` variant for that.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass import BaseModel

_ACTIVATION_FACTORIES: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "leakyrelu": nn.LeakyReLU,
    "linear": nn.Identity,
}


def _build_activation(name: str) -> nn.Module:
    key = (name or "linear").lower()
    if key not in _ACTIVATION_FACTORIES:
        raise ValueError(
            f"Unknown activation '{name}'. Supported: {sorted(_ACTIVATION_FACTORIES)}."
        )
    return _ACTIVATION_FACTORIES[key]()


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
        input_scale: float = 400.0,
        output_scale: float = 1000.0,
        bias: bool = True,
        forward_chunk_size: int = 131072,
        fourier_levels: int = 4,
    ):
        """
        Args:
            config: Optional config dict forwarded to ``BaseModel``.
            in_features: Number of input coordinate features (default 3 for x, y, z).
            out_features: Number of output features (default 1 for signed distance).
            hidden_dims: Output width of each hidden ``Linear``. Defaults to the
                winning spec ``[256, 256, 256, 512, 512, 512, 512, 384, 256]``
                (9 hidden layers + 1 output layer = 10 ``Linear`` layers total).
            activations: Activation name applied AFTER each ``Linear`` (length
                ``len(hidden_dims) + 1``). Recognised: ``silu``, ``leakyrelu``,
                ``linear``. Defaults to the winning profile C.
            concat_input_at: 1-indexed layer numbers whose ``Linear`` input is
                concatenated with the (encoded) input features. Defaults to
                ``[1, 3, 5, 7, 9]``.
            residual_at: 1-indexed layer numbers where the layer's input (after
                any concat) is added back to its post-activation output. A
                learnable projection is added automatically when shapes differ.
                Defaults to ``[4, 6, 8]``.
            input_scale: Divides the input (so ``x / input_scale`` is fed to
                the network). Default 400.0 (winning value).
            output_scale: Multiplies the network output (mm rescale). Default 1000.0.
            bias: Whether ``Linear`` layers carry a bias.
            forward_chunk_size: When > 0 and the input batch dimension exceeds
                this value, the forward pass is computed in chunks along the
                first dim and concatenated. This keeps activation memory bounded
                for dense grid evaluations (e.g. the connectedness loss). Set
                to ``0`` to disable chunking.
            fourier_levels: Number of NeRF-style sinusoidal positional-encoding
                bands applied to the (scaled) input. When ``L > 0``, the input
                fed to the network — and re-injected at ``concat_input_at``
                layers — is
                ``[x, sin(2^k π x), cos(2^k π x)]_{k=0..L-1}`` of effective
                dimension ``in_features * (1 + 2L)``. ``0`` disables the
                encoding. Default 4 (winning value).
        """
        super().__init__(config=config, overwrite_base_init=True)

        if hidden_dims is None:
            hidden_dims = [256, 256, 256, 512, 512, 512, 512, 384, 256]
        if activations is None:
            activations = [
                "silu", "silu", "leakyrelu", "silu", "leakyrelu",
                "silu", "leakyrelu", "silu", "silu", "linear",
            ]
        if concat_input_at is None:
            concat_input_at = [1, 3, 5, 7, 9]
        if residual_at is None:
            residual_at = [4, 6, 8]

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
        self.fourier_levels = int(fourier_levels)
        if self.fourier_levels < 0:
            raise ValueError(f"fourier_levels must be >= 0, got {self.fourier_levels}.")
        # Effective width of what the MLP actually sees as "input" (used for
        # both the first Linear and every concat re-injection point).
        self._enc_dim = self.in_features * (1 + 2 * self.fourier_levels)
        if self.fourier_levels > 0:
            # Standard NeRF-style frequency schedule: 2^k π for k = 0..L-1.
            # Registered as a buffer so it moves with .to(device) and is
            # serialised with the model.
            freqs = (2.0 ** torch.arange(self.fourier_levels)) * math.pi
            self.register_buffer("_fourier_freqs", freqs)

        # mirror BaseModel attributes for get_model_info_dict / debug introspection
        self.in_dim = self.in_features
        self.hidden = self.hidden_dims[0] if self.hidden_dims else 0
        self.layers = n_linear

        out_dims = [*self.hidden_dims, self.out_features]

        self.linears = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.residual_projs = nn.ModuleDict()  # keyed by str(layer_idx_1based)

        prev_dim = self._enc_dim
        for layer_idx in range(1, n_linear + 1):
            in_dim = prev_dim + (self._enc_dim if layer_idx in self.concat_input_at else 0)
            out_dim = out_dims[layer_idx - 1]

            linear = nn.Linear(in_dim, out_dim, bias=bias)
            self.linears.append(linear)
            self.acts.append(_build_activation(self.activations_cfg[layer_idx - 1]))

            if layer_idx in self.residual_at and in_dim != out_dim:
                # auto-projection so the residual add is well-defined when shapes differ
                self.residual_projs[str(layer_idx)] = nn.Linear(in_dim, out_dim, bias=False)

            prev_dim = out_dim

    def _encode(self, x_in: torch.Tensor) -> torch.Tensor:
        """Apply the optional sinusoidal positional encoding."""
        if self.fourier_levels == 0:
            return x_in
        # Shapes: x_in is (..., in_features); self._fourier_freqs is (L,).
        freqs = cast(torch.Tensor, self._fourier_freqs)
        scaled = x_in.unsqueeze(-1) * freqs  # (..., in_features, L)
        sin_part = torch.sin(scaled).flatten(-2)
        cos_part = torch.cos(scaled).flatten(-2)
        return torch.cat([x_in, sin_part, cos_part], dim=-1)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self._encode(x / self.input_scale)
        h = x_in
        for i, (linear, act) in enumerate(zip(self.linears, self.acts, strict=True)):
            layer_idx = i + 1  # 1-indexed
            if layer_idx in self.concat_input_at:
                h = torch.cat([h, x_in], dim=-1)
            pre = h
            h = act(linear(h))
            if layer_idx in self.residual_at:
                key = str(layer_idx)
                proj = self.residual_projs[key]
                skip = proj(pre) if proj is not None else pre
                h = h + skip
        return h * self.output_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Chunk along the leading (batch/point) dim when the input is large to
        # bound peak activation memory. This rescues large-grid evaluations
        # (e.g. ConnectednessV1's full voxel grid) from CUDA OOM without
        # changing numerics for normal-sized batches.
        #
        # NOTE: input/output scaling lives ONLY in _forward_single (driven by
        # input_scale / output_scale). Do NOT add extra /1000 or *1000 here —
        # doing so previously caused a 1_000_000x input attenuation and
        # asymmetric output scaling between the chunked / non-chunked paths,
        # which made the SDF collapse to a constant and produced 0 vertices
        # at mesh extraction.
        chunk = self.forward_chunk_size
        if chunk <= 0 or x.dim() == 0 or x.shape[0] <= chunk:
            return self._forward_single(x)
        outs = [self._forward_single(x[i : i + chunk]) for i in range(0, x.shape[0], chunk)]
        return torch.cat(outs, dim=0)

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
            "fourier_levels": self.fourier_levels,
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
