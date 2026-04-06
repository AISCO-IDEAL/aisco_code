### this file contains a patch for wandb.log to allow using monotonic steps
# for training and rendering logs without forcing everything onto W&B's global step,
# which can cause issues with out-of-order logs when using multiple loggers or threads.
# The patch checks if the "step" argument is provided in wandb.log calls and, if so,
# adds it as "training_epoch" to the logged data.
# Additionally, if any keys in the logged data start with "renders/", "animations/", or "images/",
#  it also adds the step as "render_epoch".
import wandb


def patch_wandb_log_for_monotonic_steps() -> None:
    original_log = wandb.log

    def _patched_log(data=None, *args, **kwargs):
        step = kwargs.pop("step", None)
        if not isinstance(data, dict):
            return original_log(data, *args, **kwargs)

        payload = dict(data)
        if step is not None:
            payload.setdefault("training_epoch", step)
            if any(k.startswith(("renders/", "animations/", "images/")) for k in payload):
                payload.setdefault("render_epoch", step)
        return original_log(payload, *args, **kwargs)

    wandb.log = _patched_log
