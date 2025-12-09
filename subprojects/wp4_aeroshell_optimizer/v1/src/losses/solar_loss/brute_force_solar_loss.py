import torch
from torch.nn.utils.stateless import functional_call

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

        # Create a dict of named parameters to use with stateless functional_call
        params = dict(model.named_parameters())

        # Compute base loss at current parameter values without attaching to the original params
        with torch.no_grad():
            base_loss = float(n_diff_solar_loss(lambda *args, params=params, **kwargs: functional_call(model, params, args, kwargs)))

        # Loop over all parameters and estimate gradients via finite differences
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Allocate gradient tensor with same shape
            grad = torch.zeros_like(param)

            # Work on a flattened view for simplicity (on a clone to avoid in-place on originals)
            with torch.no_grad():
                flat_param = param.flatten()
                flat_grad = grad.flatten()

                for i in range(flat_param.numel()):
                    original_value = flat_param[i].item()

                    # Build a perturbed parameter tensor (clone to avoid touching original)
                    perturbed = param.clone().flatten()
                    perturbed[i] = original_value + self.__epsilon
                    perturbed = perturbed.view_as(param)

                    # Build a new params dict with the single perturbed parameter
                    perturbed_params = dict(params)
                    perturbed_params[name] = perturbed

                    # Recompute loss with perturbed parameters using stateless call (bind loop var)
                    loss_plus = float(
                        n_diff_solar_loss(
                            lambda *args, perturbed_params=perturbed_params, **kwargs: functional_call(
                                model, perturbed_params, args, kwargs
                            )
                        )
                    )

                    # Finite-difference derivative: (f(x+ε) - f(x)) / ε
                    flat_grad[i] = (loss_plus - base_loss) / self.__epsilon

            # Attach the computed gradient to the parameter
            param.grad = grad

        # Return the loss as a torch scalar to look like a normal loss
        return torch.tensor(base_loss, dtype=torch.float32, device=self.device)
