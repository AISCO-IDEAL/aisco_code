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

            return (laplacian_sdf**2).mean()
