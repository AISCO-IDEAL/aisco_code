import copy

import torch

from subprojects.wp3_solar_estimator.v1.src.n_diff_solar_loss import n_diff_solar_loss
from subprojects.wp4_aeroshell_optimizer.v1.src.losses.parent_loss import ParentLoss


class BruteForceSolarLoss(ParentLoss):
    def _setup(self) -> None:
        self.__epsilon: float = 1e-4

    def _loss(self, model) -> torch.Tensor:
        # Make sure we're not accumulating stale gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # Work on a deepcopy to avoid in-place modifications on the original params
        device = next(model.parameters()).device
        model_eval = copy.deepcopy(model)
        model_eval.to(device)
        model_eval.train(model.training)

        # Compute base loss without tracking gradients
        with torch.no_grad():
            base_loss = float(n_diff_solar_loss(model_eval))

        # Loop over all parameters (evaluate on the copy) and estimate gradients via finite differences
        for param_eval, param_orig in zip(model_eval.parameters(), model.parameters(), strict=False):
            if not param_orig.requires_grad:
                continue

            # Allocate gradient tensor with same shape on the original param's device
            grad = torch.zeros_like(param_orig)

            with torch.no_grad():
                flat_param = param_eval.view(-1)
                flat_grad = grad.view(-1)

                for i in range(flat_param.numel()):
                    orig = flat_param[i].item()

                    # x + epsilon on the copy
                    flat_param[i] = orig + self.__epsilon
                    loss_plus = float(n_diff_solar_loss(model_eval))

                    # derivative ≈ (f(x+ε) - f(x)) / ε
                    flat_grad[i] = (loss_plus - base_loss) / self.__epsilon

                    # restore
                    flat_param[i] = orig

            # Assign the estimated gradient back to the original model's parameter
            param_orig.grad = grad

        return torch.tensor(base_loss, dtype=torch.float32, device=self.device)
