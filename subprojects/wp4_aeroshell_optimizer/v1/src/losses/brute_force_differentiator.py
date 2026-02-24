import torch

import copy


class BruteForceDifferentiator:
    def __init__(self, loss_function, device) -> None:
        self.device = device
        self.__epsilon: float = 1e-4
        self.__loss_function = loss_function

    def loss(self, model) -> torch.Tensor:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        device = next(model.parameters()).device
        model_eval = copy.deepcopy(model)
        model_eval.to(device)
        model_eval.train(model.training)

        with torch.no_grad():
            base_loss = float(self.__loss_function(model_eval))

        for param_eval, param_orig in zip(model_eval.parameters(), model.parameters(), strict=False):
            if not param_orig.requires_grad:
                continue

            grad = torch.zeros_like(param_orig)

            with torch.no_grad():
                flat_param = param_eval.view(-1)
                flat_grad = grad.view(-1)

                for i in range(flat_param.numel()):
                    orig = flat_param[i].item()

                    flat_param[i] = orig + self.__epsilon
                    loss_plus = float(self.__loss_function(model_eval))

                    flat_grad[i] = (loss_plus - base_loss) / self.__epsilon

                    flat_param[i] = orig

            param_orig.grad = grad

        return torch.tensor(base_loss, dtype=torch.float32, device=self.device)
