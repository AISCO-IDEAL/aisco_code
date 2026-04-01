import math
import os
from typing import Tuple

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v2.archive.GINN_Playground.simple_GINN import SimpleGINN


def make_grid(nx: int, ny: int, box=(0.0, 1.0, 0.0, 1.0), device: str = "cpu") -> tuple[torch.Tensor, float, tuple[int, int]]:
	x0, x1, y0, y1 = box
	xs = torch.linspace(x0, x1, nx, device=device)
	ys = torch.linspace(y0, y1, ny, device=device)
	grid_y, grid_x = torch.meshgrid(ys, xs)
	coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
	area = ((x1 - x0) / nx) * ((y1 - y0) / ny)
	return coords, area, (ny, nx)


def _next_available_path(path: str) -> str:
	"""Return a non-existing path by appending _N before the extension.

	Scans the target directory for files that match the base name and an
	optional numbered suffix, then returns the next available filename.
	"""
	if not os.path.exists(path):
		return path
	import re
	base_name = os.path.basename(path)
	dir_name = os.path.dirname(path) or '.'
	base, ext = os.path.splitext(base_name)
	pattern = re.compile(re.escape(base) + r'(?:_(\d+))?' + re.escape(ext) + r'$')
	max_idx = 0
	for fn in os.listdir(dir_name):
		m = pattern.match(fn)
		if m:
			if m.group(1):
				idx = int(m.group(1))
				max_idx = max(max_idx, idx)
			else:
				max_idx = max(max_idx, 1)
	if max_idx == 0:
		return path
	return os.path.join(dir_name, f"{base}_{max_idx+1}{ext}")


def train_density(
	model: nn.Module = None,
	nx: int = 32,
	ny: int = 32,
	box=(0.0, 1.0, 0.0, 1.0),
	z_dim: int = 2,
	epochs: int = 200,
	warmup_epochs: int = 0,
	K: int = 8,
	target_mass: float = 1.0,
	diversity_weight: float = 1.0,
	mass_weight: float = 10.0,
	tv_weight: float = 0.0,
	l2_weight: float = 1e-4,
	lr: float = 1e-3,
	device: str = None,
	save_path: str = None,
	hidden_dims=None,
	min_density = 0.0,
	max_density = 5.0,
):
	"""

	Trains a SimpleGINN model to learn a density function over a 2D grid.


    Args:
        model (nn.Module, optional): An instance of a neural network model. If None, a SimpleGINN will be created.
        nx (int): Number of grid points in the x-direction.
        ny (int): Number of grid points in the y-direction.
        box (tuple): A tuple defining the bounding box for the grid in the format (x0, x1, y0, y1).
        z_dim (int): Dimensionality of the latent variable z.
        epochs (int): Number of training epochs.
        K (int): Number of samples to draw from the latent space for each training step.
        target_mass (float): The desired total mass that the density should integrate to.
        diversity_weight (float): Weight for the diversity loss term that encourages different samples to produce different densities.
        mass_weight (float): Weight for the mass loss term that encourages the total mass to match the target mass.
        tv_weight (float): Weight for the total variation loss term that encourages smoothness in the density.
        l2_weight (float): Weight for L2 regularization on model parameters.
        lr (float): Learning rate for the optimizer.
        device (str, optional): Device to run training on. If None, it will automatically use "cuda" if available, otherwise "cpu".
        save_path (str, optional): Path to save the trained model. If None, it will save to a default location.
	"""
	torch.manual_seed(42)
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	print(
		f"Training with parameters: nx={nx}, ny={ny}, box={box}, z_dim={z_dim}, epochs={epochs}, K={K}, "
		f"target_mass={target_mass}, diversity_weight={diversity_weight}, mass_weight={mass_weight}, "
		f"tv_weight={tv_weight}, l2_weight={l2_weight}, lr={lr}, device={device}"
	)
	# allow specifying bounds for density
	hidden_dims = (256, 256, 256, 256) if hidden_dims is None else hidden_dims
	if model is None:
		model = SimpleGINN(in_features=2, z_dim=z_dim, hidden_dims=hidden_dims,
						   min_density=min_density, max_density=max_density)

	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	coords, area, (ny_, nx_) = make_grid(nx, ny, box=box, device=device)
	N = coords.shape[0]

	if save_path is None:
		base = os.path.dirname(__file__)
		save_dir = os.path.join(base, "data")
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, "simple_ginn.pt")

	# avoid overwriting existing checkpoints
	save_path = _next_available_path(save_path)

	for epoch in range(1, epochs + 1):
		model.train()
		zs = torch.randn(K, z_dim, device=device)

		dens_list = []
		for k in range(K):
			dens_k = model(coords, zs[k])
			dens_list.append(dens_k)

		dens = torch.stack(dens_list, dim=0)
		masses = dens.sum(dim=1) * area

		mass_loss = ((masses - target_mass) ** 2).mean() if epoch > warmup_epochs else torch.tensor(-1.0, device=device)

		eps = 1e-8
		eps = 1e-8
		# normalize densities per-sample to sum to 1 (probability-like) for diversity computation
		dens_sum = dens.sum(dim=1, keepdim=True)
		dens_norm = dens / (dens_sum + eps)

		pair_sum = torch.tensor(0.0, device=device)
		count = 0
		for i in range(K):
			for j in range(i + 1, K):
				pair_sum += ((dens_norm[i] - dens_norm[j]) ** 2).sum()
				count += 1

		# normalize diversity by number of pairs and number of grid points to keep scale stable
		N = dens.shape[1]
		diversity_metric = (pair_sum / (max(count, 1) * N)).item()
		diversity_loss = -pair_sum / max(count, 1)  # still maximize pairwise squared difference

		dens_resh = dens.view(K, ny_, nx_)
		tv_x = ((dens_resh[:, :, 1:] - dens_resh[:, :, :-1]) ** 2).mean()
		tv_y = ((dens_resh[:, 1:, :] - dens_resh[:, :-1, :]) ** 2).mean()
		tv_loss = tv_x + tv_y

		l2_reg = torch.tensor(0.0, device=device)
		for p in model.parameters():
			l2_reg = l2_reg + (p.pow(2).sum())

		loss = (
			mass_weight * mass_loss
			+ diversity_weight * diversity_loss
			+ tv_weight * tv_loss
			+ l2_weight * l2_reg
		)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if epoch % max(1, epochs // 10) == 0 or (epoch <= warmup_epochs and epoch % max(1, warmup_epochs // 5) == 0):
			with torch.no_grad():
				model.eval()
				total_mass = masses.mean().item()
				print(
					f"Epoch {epoch}/{epochs}: loss={loss.item():.10f}, mass_mean={total_mass:.6f}, mass_loss={mass_loss.item():.6f}, diversity={diversity_metric:.12f}, l2_loss={l2_reg.item()*l2_weight:.6f}, tv_loss={tv_loss.item()*tv_weight:.6f}"
				)

	model.save(save_path, init_params={"in_features": 2, "z_dim": z_dim, "hidden_dims": hidden_dims})
	print(f"Model saved to {save_path}")
	return save_path


if __name__ == "__main__":
	train_density(epochs=15000,warmup_epochs=0, lr=2e-4, nx=32, ny=32, K=4, z_dim=4, mass_weight=1.0, diversity_weight=10.0, tv_weight=0.3, l2_weight=0.0,max_density=5.0)
