"""
weights for losses of aeroshell_optimizer are handeled here

"""

import torch


def weights_alm(lamdas: torch.Tensor, mus: torch.Tensor, nus: torch.Tensor, alphas: torch.Tensor, gamma: float, constrained_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Augmented Lagrangian Method (ALM) weights for constrained losses.

    Args:
        lamdas (torch.Tensor): Lagrange multipliers for the constrained losses.
        mus (torch.Tensor): Penalty parameters for the constrained losses.
        nus (torch.Tensor): Smoothed Violation for adaptive penalty.
        alphas (torch.Tensor): used to compute the new smoothed violation. Effectively averages violations over past epochs
        gamma (float): learning rate for updating mu.
        constrained_losses (torch.Tensor): Values of the constrained losses.
    Returns:
        torch.Tensor: the updated lambdas, mus and nus.
    """
    epsilon = 1e-8  # Small constant to prevent division by zero
    with torch.no_grad():
        for i in range(len(constrained_losses)): # iterate over constrained losses
            # update nu, which is a torch tensor
            nu_1 = alphas[i] * nus[i] + (1 - alphas[i]) * abs(constrained_losses[i])
            nus[i] = nu_1

            # update mu
            mu_1 = gamma/(nus[i] + epsilon)
            mus[i] = mu_1

            # update lambda
            lamda_1 = lamdas[i] + mus[i] * abs(constrained_losses[i])
            lamdas[i] = lamda_1

    return lamdas, mus, nus

def weights_manual(loss_names: dict, losses_config: dict, loss_families: list) -> list:
    """Get manual weight for a given loss from the losses_config dictionary.

    Args:
        losses_config (dict): Dictionary containing loss configurations, including weights.
        loss_families (list): Families of the losses (e.g., ['bb', 'solar'], etc).
        loss_names (list): Names of the losses for which to get the weights.
    Returns:
        list: The weights for the specified losses.

    """
    all_weights = []
    for f in loss_families:
        for loss_name in loss_names[f]:
            weight = losses_config[f][loss_name].get("weight", 1.0)
            all_weights.append(weight)
    return all_weights
