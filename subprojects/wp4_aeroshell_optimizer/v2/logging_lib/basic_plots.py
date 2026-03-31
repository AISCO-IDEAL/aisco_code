import matplotlib.pyplot as plt
import numpy as np
import torch

import wandb


def _safe_step(step: int | None) -> int:
    """Return an explicit non-drifting step for wandb logging."""
    return 0 if step is None else int(step)


def sanitize_metric(value):
    """
    Converts tensors or numpy arrays to python scalars for logging.
    """
    if isinstance(value, torch.Tensor | np.ndarray):
        return value.item()
    return value


def basic_single_log(key, value, step: int | None = None):
    """
    A simple wrapper around wandb.log to log a single key-value pair with a step.
    Args:
    - key (str): The name of the metric to log.
    - value (float or int): The value of the metric to log.
    - step (int): The current step or epoch number to associate with this log entry.
    """
    wandb.log({key: sanitize_metric(value)}, step=_safe_step(step))


def basic_multi_log(metrics_dict: dict, step: int | None = None):
    """
    A wrapper around wandb.log to log multiple key-value pairs at once with a step.
    Args:
    - metrics_dict (dict): A dictionary where keys are metric names and values are the corresponding metric values to log.
    e.g. {"modelAccuracy": 0.95, "learning_rate": 0.001}
    - step (int): The current step or epoch number to associate with this log entry.
    """
    sanitized_dict = {k: sanitize_metric(v) for k, v in metrics_dict.items()}
    wandb.log(sanitized_dict, step=_safe_step(step))


def multi_log_in_one(metric_name: str | None = None, metrics_dict: dict | None = None, step: int | None = None):
    """
    A wrapper around wandb.log to log multiple key-value pairs at once with a step, but all metrics are logged under a single chart with key "metrics".
    This can be useful for grouping related metrics together in the wandb dashboard.
    Args:
    - metrics_dict (dict): A dictionary where keys are metric names and values are the corresponding metric values to log.
    e.g. {"modelAccuracy": 0.95, "learning_rate": 0.001}
    - step (int): The current step or epoch number to associate with this log entry.
    """
    sanitized_dict = (
        {f"{metric_name}/{k}": sanitize_metric(v) for k, v in metrics_dict.items()} if metrics_dict is not None else {}
    )
    wandb.log(sanitized_dict, step=_safe_step(step))


def log_matplotlib_fig(fig: plt.Figure, key: str, step: int | None = None):
    """
    Logs a matplotlib figure to wandb under the specified key and step.
    Args:
    - fig (plt.Figure): The matplotlib figure to log.
    - key (str): The name of the metric to log the figure under.
    - step (int): The current step or epoch number to associate with this log entry.
    """
    wandb.log({key: wandb.Image(fig)}, step=_safe_step(step))


def log_renderer_image(image, key: str, step: int | None = None):
    """
    Logs a rendered image (e.g. from a 3D renderer) to wandb under the specified key and step.
    Args:
    - image (np.array or PIL.Image): The image to log, can be a numpy array or a PIL Image.
    - key (str): The name of the metric to log the image under.
    - step (int): The current step or epoch number to associate with this log entry.
    """
    wandb.log({key: wandb.Image(image)}, step=_safe_step(step))
