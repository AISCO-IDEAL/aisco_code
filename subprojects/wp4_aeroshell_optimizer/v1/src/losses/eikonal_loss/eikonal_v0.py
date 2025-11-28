# eikonal v0
import torch
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class EikonalV0(ParentLoss):
    def _setup(self) -> None:
        pass

    def _loss(self, model) -> torch.Tensor:
        """
        Arguments:
            model: The SDF network
                sdf: the sdf function values for each point
                grad_sdf: the gradients of the sdf function at each point
                    both are torch.Tensors
        Returns:
            Eikonal loss (to be minimized)
                loss: Mean absolute difference between the norm of the gradients and 1
                loss = 1/n * sum from 1 to n of  ( ||grad_sdf||_2 - 1 )^2
                    source: https://arxiv.org/pdf/2002.10099, Implicit Geometric Regularization for Learning Shapes, Page 2, 2. Method l(theta)
        """
        with torch.enable_grad():
            sdf, grad_sdf = model(self.points)
            diff = torch.norm(grad_sdf, p=2, dim=-1) - 1.0
            eikonal_loss = (
                torch.pow(diff, 2).mean()
            )  # in v0 there was no ^2 here, but it should be according to the paper
            return eikonal_loss
