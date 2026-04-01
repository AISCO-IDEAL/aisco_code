# smoothness v0
import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class SmoothnessV0(ParentLoss):
    def _setup(self) -> None:
        pass

    def _loss(self, model) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network
        Returns:
            Smoothness loss (to be minimized)
        """
        # TODO: shorten this code as it was copied from v0 by an AI assistant

        # Ensure points require grad
        if not self.points.requires_grad:
            self.points.requires_grad = True

        with torch.enable_grad():
            output = model(self.points)

            # Handle case where model returns only sdf or (sdf, grad)
            if isinstance(output, tuple):
                sdf, grad_sdf = output
            else:
                sdf = output
                # Compute gradient if not returned
                grad_sdf = torch.autograd.grad(
                    outputs=sdf,
                    inputs=self.points,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

            # Compute Laplacian (sum of second derivatives)
            laplacian_sdf = torch.zeros(self.points.shape[0], device=self.points.device)
            in_dim = self.points.shape[1]

            for dim in range(in_dim):
                second_deriv = torch.autograd.grad(
                    outputs=grad_sdf[:, dim],
                    inputs=self.points,
                    grad_outputs=torch.ones_like(grad_sdf[:, dim]),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                laplacian_sdf += second_deriv[:, dim]

            n_points = laplacian_sdf.shape[0]
            smoothness_loss = (laplacian_sdf**2).mean() if n_points > 0 else torch.tensor(0.0, device=self.points.device)

            # Store diagnostics for DebugManager
            with torch.no_grad():
                self._last_diagnostics = {
                    "n_points": n_points,
                    "laplacian_min": laplacian_sdf.min().item(),
                    "laplacian_max": laplacian_sdf.max().item(),
                    "laplacian_mean": laplacian_sdf.mean().item(),
                    "laplacian_std": laplacian_sdf.std().item() if n_points > 1 else 0.0,
                    "laplacian_abs_max": laplacian_sdf.abs().max().item(),
                    "pct_high_curvature": float((laplacian_sdf.abs() > 1.0).sum().item()) / max(n_points, 1) * 100,
                    "sdf_min": sdf.min().item(),
                    "sdf_max": sdf.max().item(),
                    "sdf_mean": sdf.mean().item(),
                    "grad_sdf_norm_mean": torch.norm(grad_sdf, p=2, dim=-1).mean().item(),
                }

            return smoothness_loss
