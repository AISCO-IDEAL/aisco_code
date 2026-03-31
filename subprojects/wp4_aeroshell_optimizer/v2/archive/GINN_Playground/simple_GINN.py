import os

import torch
from torch import nn


class SimpleGINN(nn.Module):
	def __init__(
		self,
		in_features=2,
		z_dim=2,
		hidden_dims=None,
		activation=nn.GELU,
		min_density: float = 0.0,
		max_density: float = 10.0,
		output_activation: str = "tanh",
		use_batchnorm: bool = False,
		use_layernorm: bool = False,
		dropout: float = 0.0,
		use_residual: bool = True,
	):
		super().__init__()
		self.in_features = in_features
		self.z_dim = z_dim
		self.hidden_dims = tuple(hidden_dims)
		self.min_density = float(min_density)
		self.max_density = float(max_density)
		self.output_activation = output_activation
		self.use_batchnorm = use_batchnorm
		self.use_layernorm = use_layernorm
		self.dropout = float(dropout)
		self.use_residual = use_residual

		# Normalize activation: accept an nn.Module class or an nn.Module instance.
		# We store an activation class so we can create a fresh instance per layer.
		if isinstance(activation, type) and issubclass(activation, nn.Module):
			activation_cls = activation
		elif isinstance(activation, nn.Module):
			activation_cls = type(activation)
		else:
			raise TypeError(
				"activation must be an nn.Module class or nn.Module instance, got %r"
				% (type(activation),)
		)
		self.activation_cls = activation_cls

		if self.use_batchnorm and self.use_layernorm:
			raise ValueError("Cannot set both use_batchnorm and use_layernorm")

		# Build progressive hidden blocks
		self.blocks = nn.ModuleList()
		self.residual_flags = []
		input_dim = in_features + (z_dim or 0)
		for h in self.hidden_dims:
			block = []
			block.append(nn.Linear(input_dim, h))
			if self.use_batchnorm:
				block.append(nn.BatchNorm1d(h))
			elif self.use_layernorm:
				block.append(nn.LayerNorm(h))
			block.append(self.activation_cls())
			if self.dropout > 0:
				block.append(nn.Dropout(self.dropout))
			self.blocks.append(nn.Sequential(*block))
			self.residual_flags.append(self.use_residual and input_dim == h)
			input_dim = h

		self.out_layer = nn.Linear(input_dim, 1)


	def forward(self, x: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
		if self.z_dim and z is None:
			z = torch.zeros(x.shape[0], self.z_dim, device=x.device, dtype=x.dtype)
		elif self.z_dim:
			if z.dim() == 1:
				z = z.unsqueeze(0).expand(x.shape[0], -1)
			elif z.shape[0] != x.shape[0]:
				z = z.expand(x.shape[0], -1)

		inp = x if (not self.z_dim) else torch.cat([x, z], dim=-1)
		x_in = inp
		for block, use_res in zip(self.blocks, self.residual_flags):
			x_out = block(x_in)
			if use_res:
				x_in = x_in + x_out
			else:
				x_in = x_out

		pre = self.out_layer(x_in).view(-1)

		if self.output_activation == "sigmoid":
			out = torch.sigmoid(pre)
		elif self.output_activation == "tanh":
			out = torch.tanh(pre) * 0.5 + 0.5
		elif self.output_activation == "softplus":
			out = torch.nn.functional.softplus(pre)
		else:
			raise ValueError(f"Unknown output_activation={self.output_activation}")

		return out * (self.max_density - self.min_density) + self.min_density

	def save(self, path: str, init_params: dict = None):
		os.makedirs(os.path.dirname(path), exist_ok=True)
		data = {
			"state_dict": self.state_dict(),
			"init_params": init_params
			if init_params is not None
			else {
				"in_features": self.in_features,
				"z_dim": self.z_dim,
				"hidden_dims": self.hidden_dims,
				"min_density": self.min_density,
				"max_density": self.max_density,
			},
		}
		torch.save(data, path)

	@staticmethod
	def load(path: str, device: str = "cpu") -> "SimpleGINN":
		ckpt = torch.load(path, map_location=device)
		params = ckpt.get("init_params", {})
		model = SimpleGINN(**params)
		model.load_state_dict(ckpt["state_dict"])
		model.to(device)
		model.eval()
		return model
