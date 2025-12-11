"""
weights for losses of aeroshell_optimizer are handeled here

"""
def dynamic_update_alm():

def weights_alm():


def weights_manual(loss_name: str, losses_config: dict, loss_family: str): -> float:
    """Get manual weight for a given loss from the losses_config dictionary.

    Args:
        losses_config (dict): Dictionary containing loss configurations, including weights.
        loss_family (str): Family of the loss (e.g., 'bb', 'solar', etc).
        loss_name (str): Name of the loss for which to get the weight.
    Returns:
        float: The weight for the specified loss.

    """
    return losses_config[loss_family][loss_name]
