import wandb


def basic_single_log(key, value, step: int | None = None):
    """
    A simple wrapper around wandb.log to log a single key-value pair with a step.
    Args:
    - key (str): The name of the metric to log.
    - value (float or int): The value of the metric to log.
    - step (int): The current step or epoch number to associate with this log entry.
    """
    wandb.log({key: value}, step=step)

def basic_multi_log(metrics_dict: dict, step: int | None = None):
    """
    A wrapper around wandb.log to log multiple key-value pairs at once with a step.
    Args:
    - metrics_dict (dict): A dictionary where keys are metric names and values are the corresponding metric values to log.
    e.g. {"modelAccuracy": 0.95, "learning_rate": 0.001}
    - step (int): The current step or epoch number to associate with this log entry.
    """
    wandb.log(metrics_dict, step=step)

def multi_log_in_one(metric_name: str | None = None , metrics_dict: dict | None = None, step: int | None = None):
    """
    A wrapper around wandb.log to log multiple key-value pairs at once with a step, but all metrics are logged under a single key "metrics".
    This can be useful for grouping related metrics together in the wandb dashboard.
    Args:
    - metrics_dict (dict): A dictionary where keys are metric names and values are the corresponding metric values to log.
    e.g. {"modelAccuracy": 0.95, "learning_rate": 0.001}
    - step (int): The current step or epoch number to associate with this log entry.
    """
    wandb.log({f"{metric_name}": metrics_dict}, step=step)
