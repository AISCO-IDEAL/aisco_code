"""
weights for losses of aeroshell_optimizer are handeled here

"""

import torch


def weights_alm(
    lamdas: torch.Tensor,
    mus: torch.Tensor,
    nus: torch.Tensor,
    alphas: torch.Tensor,
    gamma: float,
    constrained_losses: torch.Tensor,
    thresholds: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Augmented Lagrangian Method (ALM) weights for constrained losses.

    Args:
        lamdas (torch.Tensor): Lagrange multipliers for the constrained losses.
        mus (torch.Tensor): Penalty parameters for the constrained losses.
        nus (torch.Tensor): Smoothed Violation for adaptive penalty.
        alphas (torch.Tensor): used to compute the new smoothed violation. Effectively averages violations over past epochs
        gamma (float): learning rate for updating mu.
        constrained_losses (torch.Tensor): Values of the constrained losses.
        thresholds (torch.Tensor, optional): Thresholds for the constrained losses. If provided, mu will be clamped.
    Returns:
        torch.Tensor: the updated lambdas, mus and nus.
    """
    epsilon = 1e-16  # Small constant to prevent division by zero
    # SOURCE: "Conditionally adaptive augmented Lagrangian method for physics-informed learning of forward and inverse problems using artificial neural networks", page 6 after (8)

    with torch.no_grad():
        for i in range(len(constrained_losses)):  # iterate over constrained losses
            # update nu, which is a torch tensor
            nu_1 = alphas[i] * nus[i] + (1 - alphas[i]) * abs(constrained_losses[i])
            nus[i] = nu_1

            # update mu
            mu_1 = gamma / (nus[i] + epsilon)

            if thresholds is not None:
                # Ensure mu doesn't drop below a value corresponding to the threshold
                # This prevents the penalty from becoming too weak when violation is large
                min_mu = gamma / (thresholds[i] + epsilon)
                mu_1 = torch.max(mu_1, min_mu)

            mus[i] = mu_1

            # update lambda
            lamda_1 = lamdas[i] + mus[i] * abs(constrained_losses[i])
            lamdas[i] = lamda_1

    return lamdas, mus, nus


def weights_manual(loss_names: dict, losses_config: dict, loss_families: list, device: torch.device) -> torch.Tensor:
    """Get manual weight for a given loss from the losses_config dictionary.

    Args:
        losses_config (dict): Dictionary containing loss configurations, including weights.
        loss_families (list): Families of the losses (e.g., ['bb', 'solar'], etc).
        loss_names (list): Names of the losses for which to get the weights.
    Returns:
        list: The weights for the specified losses.

    """
    all_weights = torch.tensor([], device=device)
    for f in loss_families:
        for loss_name in loss_names[f]:
            weight = losses_config[f][loss_name].get("weight", 1.0)
            # add weight to all_weights tensor
            all_weights = torch.cat((all_weights, torch.tensor([weight], device=device)))
    return all_weights
