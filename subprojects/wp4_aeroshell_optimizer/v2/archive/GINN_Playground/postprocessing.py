import os
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


def _next_available_path(path: str) -> str:
	"""Return a non-existing path by appending _N before the extension.

	Scans the target directory for files that match the base name and an
	optional numbered suffix, then returns the next available filename.
	"""
	if not os.path.exists(path):
		# if parent dir missing, caller should create it; returning path is fine
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
				# base file exists (index 0)
				max_idx = max(max_idx, 1)
	if max_idx == 0:
		# no file exists, but os.path.exists(path) was True earlier, fallback
		return path
	return os.path.join(dir_name, f"{base}_{max_idx+1}{ext}")


def _find_latest_model(path: str) -> str:
	"""If exact model path doesn't exist, search same directory for matching base names and return the latest file."""
	if os.path.exists(path):
		return path
	d = os.path.dirname(path) or '.'
	base = os.path.splitext(os.path.basename(path))[0]
	candidates = []
	for fn in os.listdir(d):
		if fn.startswith(base) and fn.endswith('.pt'):
			candidates.append(os.path.join(d, fn))
	if not candidates:
		return path
	# return most recently modified candidate
	return max(candidates, key=os.path.getmtime)

try:
	from .density import make_grid
except Exception:
	from subprojects.wp4_aeroshell_optimizer.v2.archive.GINN_Playground.density import make_grid

try:
	from .simple_GINN import SimpleGINN
except Exception:
	from subprojects.wp4_aeroshell_optimizer.v2.archive.GINN_Playground.simple_GINN import SimpleGINN


def load_model(path: str, device: str = "cpu") -> SimpleGINN:
	return SimpleGINN.load(path, device=device)


def interpolate_latent_and_visualize(
	model_path: str,
	z_a: Sequence[float],
	z_b: Sequence[float],
	steps: int = 4,
	nx: int = 32,
	ny: int = 32,
	out_path: str = None,
	device: str = "cpu",
	two_dim: bool = True,
	dims: Sequence[int] = (0, 1),
	steps_y: int = None,
):
	# if exact model file not present, try to find the latest matching checkpoint
	model_path = _find_latest_model(model_path)
	if not os.path.exists(model_path):
		raise FileNotFoundError(f"Model file not found: {model_path}")
	model = load_model(model_path, device=device)
	print(f"Loaded model from {model_path} with z_dim={model.z_dim}")
	# check latent dimensionality matches provided vectors
	z_dim = getattr(model, "z_dim", None)
	if z_dim is not None:
		print(f"Model z_dim={z_dim}: interpolating between vectors of length {z_dim}")
	coords, area, (ny_, nx_) = make_grid(nx, ny, box=(0.0, 1.0, 0.0, 1.0), device=device)

	z_a = torch.tensor(z_a, dtype=torch.float32, device=device)
	z_b = torch.tensor(z_b, dtype=torch.float32, device=device)

	if two_dim:
		steps_x = steps
		steps_y = steps if steps_y is None else steps_y

		d0, d1 = int(dims[0]), int(dims[1])

		# latent ranges between z_a and z_b
		x_vals = torch.linspace(z_a[d0], z_b[d0], steps_x, device=device)
		y_vals = torch.linspace(z_a[d1], z_b[d1], steps_y, device=device)

		# build latent grid
		grid_y, grid_x = torch.meshgrid(y_vals, x_vals, indexing="ij")

		n_latent = steps_x * steps_y
		# For z_dim > 2, set extra dims to 0 and vary only d0/d1.
		latent_grid = torch.zeros((n_latent, z_dim), dtype=z_a.dtype, device=device)
		latent_grid[:, d0] = grid_x.reshape(-1)
		latent_grid[:, d1] = grid_y.reshape(-1)

		# repeat coords for batch evaluation
		n_pts = coords.shape[0]
		coords_batch = coords.repeat(n_latent, 1)
		latent_batch = latent_grid.repeat_interleave(n_pts, dim=0)

		# evaluate model in one forward pass
		with torch.no_grad():
			dens = model(coords_batch, latent_batch).detach().cpu().numpy()

		dens = dens.reshape(steps_y, steps_x, ny_, nx_)

		fig, axes = plt.subplots(steps_y, steps_x, figsize=(2 * steps_x, 2 * steps_y))
		axes = np.array(axes).reshape(steps_y, steps_x)

		mass_grid = np.zeros((steps_y, steps_x), dtype=np.float32)
		for iy in range(steps_y):
			for ix in range(steps_x):
				ax_plot = axes[iy, ix]

				field = dens[iy, ix]

				ax_plot.imshow(field, cmap="viridis", origin="lower")
				ax_plot.axis("off")

				val0 = float(x_vals[ix])
				val1 = float(y_vals[iy])

				ax_plot.set_title(f"d{d0}={val0:.2f}\nd{d1}={val1:.2f}")

				mass = float(field.sum() * area)
				mass_grid[iy, ix] = mass
				print(f"interp ({iy+1},{ix+1}) d{d0}={val0:.2f} d{d1}={val1:.2f}, mass={mass:.6f}")

		# 2D scalar summary: mass heatmap over latent interpolation plane
		fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
		im = ax2.imshow(mass_grid, cmap="inferno", origin="lower",
				 extent=[x_vals.min().item(), x_vals.max().item(), y_vals.min().item(), y_vals.max().item()],
				 aspect="auto")
		ax2.set_title(f"Mass across latent interpolation (d{d0} vs d{d1})")
		ax2.set_xlabel(f"d{d0}")
		ax2.set_ylabel(f"d{d1}")
		fig2.colorbar(im, ax=ax2, label="mass")

		# Additional 1D cross-sections along each latent dimension (like z1- and z2-sweeps)
		mass_d0 = np.zeros(steps_x, dtype=np.float32)
		mass_d1 = np.zeros(steps_y, dtype=np.float32)
		for i in range(steps_x):
			z = torch.zeros(z_dim, dtype=z_a.dtype, device=device)
			z[d0] = x_vals[i]
			# keep d1 at 0, as requested, for z_dim>2, otherwise can be used explicitly
			with torch.no_grad():
				field = model(coords, z).detach().cpu().numpy().reshape(ny_, nx_)
			mass_d0[i] = float(field.sum() * area)

		for j in range(steps_y):
			z = torch.zeros(z_dim, dtype=z_a.dtype, device=device)
			z[d1] = y_vals[j]
			with torch.no_grad():
				field = model(coords, z).detach().cpu().numpy().reshape(ny_, nx_)
			mass_d1[j] = float(field.sum() * area)

		fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(10, 4))
		ax3a.plot(x_vals.cpu().numpy(), mass_d0, marker='o')
		ax3a.set_title(f"Sweep d{d0} (fix d{d1}={z_a[d1].item():.2f})")
		ax3a.set_xlabel(f"d{d0}")
		ax3a.set_ylabel("mass")
		ax3b.plot(y_vals.cpu().numpy(), mass_d1, marker='o')
		ax3b.set_title(f"Sweep d{d1} (fix d{d0}={z_a[d0].item():.2f})")
		ax3b.set_xlabel(f"d{d1}")
		ax3b.set_ylabel("mass")
	else:
		alphas = np.linspace(0.0, 1.0, steps)

		fig, axes = plt.subplots(1, steps, figsize=(2 * steps, 2))
		if steps == 1:
			axes = [axes]

		# use primary varying dimension d0 (default 0)
		d0 = int(dims[0]) if dims is not None else 0
		for i, a in enumerate(alphas):
			z = (1.0 - a) * z_a + a * z_b
			with torch.no_grad():
				dens = model(coords, z).detach().cpu().numpy().reshape(ny_, nx_)

			axes[i].imshow(dens, cmap="viridis", origin="lower")
			axes[i].axis("off")
			val = (1.0 - a) * float(z_a[d0]) + a * float(z_b[d0])
			axes[i].set_title(f"d{d0}={val:.2f}")
			mass = float(dens.sum() * area)
			print(f"interp {i+1}/{steps}: alpha={a:.2f}, d{d0}={val:.2f}, mass={mass:.6f}")

	out_dir = os.path.join(os.path.dirname(__file__), "data")
	os.makedirs(out_dir, exist_ok=True)
	save_path = out_path or os.path.join(out_dir, "interpolation.png")

	# avoid overwriting existing images
	save_path = _next_available_path(save_path)
	fig.savefig(save_path, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved interpolation figure to {save_path}")

	if two_dim:
		base, ext = os.path.splitext(save_path)
		heatmap_path = _next_available_path(f"{base}_latent_mass{ext}")
		fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
		plt.close(fig2)
		print(f"Saved latent mass heatmap to {heatmap_path}")

		line_path = _next_available_path(f"{base}_latent_line{ext}")
		fig3.savefig(line_path, dpi=150, bbox_inches="tight")
		plt.close(fig3)
		print(f"Saved latent line sweeps to {line_path}")

	return save_path


if __name__ == "__main__":
	base = os.path.dirname(__file__)
	# find latest numbered model file simple_ginn_N.pt (N integer) if present
	data_dir = os.path.join(base, "data")
	N = None
	if os.path.isdir(data_dir) and N is None:
		for fname in os.listdir(data_dir):
			if fname.startswith("simple_ginn_") and fname.endswith(".pt"):
				N_str = fname[len("simple_ginn_"):-len(".pt")]
				if N_str.isdigit():
					N = int(N_str)
					break


	model_path = os.path.join(data_dir, f"simple_ginn_{N}.pt") if N is not None else os.path.join(data_dir, "simple_ginn.pt")


	if os.path.exists(model_path):
		interpolate_latent_and_visualize(model_path, [-2, -2], [2, 2], steps=11, nx=64, ny=64, two_dim=True)
	else:
		print(f"No model found at {model_path}. Train a model first.")
