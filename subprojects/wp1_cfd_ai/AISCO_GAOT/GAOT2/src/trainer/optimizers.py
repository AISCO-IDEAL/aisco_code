from __future__ import annotations

import csv
import logging
import os
import time
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import psutil
import torch
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

if TYPE_CHECKING:
    from .base import TrainerBase


###############
# Loss functions
###############
class RelativeL2Loss(torch.nn.Module):
    """||pred - target||_2 / ||target||_2, averaged over the batch.

    1-D inputs (e.g. masked field tensors) are treated as a single sample
    with N features, giving the whole-field relative L2 error.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        diff = pred - target
        # flatten everything except the batch dim
        diff_norm = diff.flatten(1).norm(p=2, dim=1)
        target_norm = target.flatten(1).norm(p=2, dim=1).clamp(min=1e-8)
        return (diff_norm / target_norm).mean()


class L2Loss(torch.nn.Module):
    """Mean L2 norm per sample: mean_batch( ||pred - target||_2 ).

    1-D inputs are treated as a single sample with N features.
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        diff = pred - target
        return diff.flatten(1).norm(p=2, dim=1).mean()


def build_loss_fn(loss_name: str) -> torch.nn.Module:
    """Return the loss module for the given name."""
    if loss_name == "mse":
        return torch.nn.MSELoss()
    elif loss_name == "l2":
        return L2Loss()
    elif loss_name == "relative_l2":
        return RelativeL2Loss()
    else:
        raise ValueError(f"Unknown loss '{loss_name}'. Supported: ['mse', 'l2', 'relative_l2']")


###############
# Scheduler
###############
class CustomLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        total_epochs,
        warmup_epochs,
        cosine_epochs,
        exp_decay_epochs,
        initial_lr,
        max_lr,
        min_lr,
        final_lr,
        last_epoch=-1,
    ):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.exp_decay_epochs = exp_decay_epochs
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # first phase (warm up): initial_lr to max_lr
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * (self.last_epoch / max(1, self.warmup_epochs - 1))
        elif self.last_epoch < self.warmup_epochs + self.cosine_epochs:
            # second stage (cosine): max_lr to min_lr
            epoch = self.last_epoch - self.warmup_epochs
            cosine_ratio = (1 + np.cos(np.pi * epoch / self.cosine_epochs)) / 2
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_ratio
        else:
            # third stage (exponential): min_lr to final_lr
            epoch = self.last_epoch - self.warmup_epochs - self.cosine_epochs
            decay_steps = max(1, self.exp_decay_epochs - 1)
            lr = self.min_lr * ((self.final_lr / self.min_lr) ** (epoch / decay_steps))
        return [lr for _ in self.optimizer.param_groups]


###############
# Optimizer
###############
class AdamOptimizer:
    optimizer: torch.optim.Adam
    scheduler: (
        torch.optim.lr_scheduler.StepLR
        | torch.optim.lr_scheduler.CosineAnnealingLR
        | torch.optim.lr_scheduler.ExponentialLR
        | CustomLRScheduler
        | None
    )
    epoch: int
    batch_size: int
    lr: float
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config):
        # New: allow overriding betas / eps from config, with stable defaults
        betas = (config["beta1"], config["beta2"])
        eps = getattr(config, "eps", 1e-6)

        self.optimizer = torch.optim.Adam(
            params,
            lr=config.lr,
            betas=betas,
            eps=eps,
        )
        self.epoch = config.epoch
        self.lr = config.lr
        self.loss_scale = config.loss_scale
        self.eval_every_eps = config.eval_every_eps
        self.early_save_metric = config.early_save_metric.lower()

        if self.early_save_metric not in ["train", "val"]:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")

        if config.scheduler == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma
            )
        elif config.scheduler == "cos":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.scheduler_t_max, eta_min=config.scheduler_eta_min
            )
        elif config.scheduler == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config.scheduler_gamma)
        elif config.scheduler == "mix":
            warmup_epochs = int(0.1 * self.epoch)
            exp_decay_epochs = int(0.05 * self.epoch)
            cosine_epochs = self.epoch - warmup_epochs - exp_decay_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config.max_lr,
                min_lr=config.min_lr,
                final_lr=config.final_lr,
            )
        else:
            self.scheduler = None

    def optimize(self, trainer: TrainerBase, description: str = "AdamOptimizer", color: str = "blue"):
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        epochs = []
        val_epochs = []
        val_losses = []

        # Only rank 0 should display progress bar
        disable_pbar = trainer.setup_config.get("rank", 0) != 0
        pbar = tqdm(total=self.epoch, desc=description, colour=color, disable=disable_pbar)
        for epoch in range(self.epoch):
            # Ensure DistributedSampler shuffles differently each epoch
            train_sampler = getattr(trainer.train_loader, "sampler", None)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)  # type: ignore[union-attr]
            trainer.model.train()
            total_loss = 0.0
            for batch in trainer.train_loader:
                start_time = time.time()
                self.optimizer.zero_grad()
                with trainer.autocast_ctx:
                    train_loss = trainer.train_step(batch)
                train_loss.backward()
                self.optimizer.step()
                total_loss += train_loss.item()
                if isinstance(trainer.device, torch.device) and trainer.device.type == "cuda":
                    torch.cuda.synchronize()
                time_total += time.time() - start_time

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.update(1)

            if (epoch + 1) % self.eval_every_eps == 0:
                train_loss = total_loss / len(trainer.train_loader)
                losses.append(train_loss)
                epochs.append(epoch)
                val_metrics = trainer.validate(trainer.val_loader)
                val_loss = val_metrics["loss"]
                if not disable_pbar:
                    pbar.set_postfix({"loss": train_loss, "val_loss": val_loss})
                val_losses.append(val_loss)
                val_epochs.append(epoch)

                current_loss = val_loss if self.early_save_metric == "val" else train_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = deepcopy(trainer.model.state_dict())

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        if not disable_pbar:
            pbar.close()

        return {
            "train": {
                "loss": losses,
                "epoch": epochs,
            },
            "valid": {
                "loss": val_losses,
                "epoch": val_epochs,
            },
            "best": {
                "epoch": best_epoch,
                "loss": best_loss,
            },
            "time": time_total,
        }


class AdamWOptimizer:
    optimizer: torch.optim.AdamW
    scheduler: (
        torch.optim.lr_scheduler.StepLR
        | torch.optim.lr_scheduler.CosineAnnealingLR
        | torch.optim.lr_scheduler.ExponentialLR
        | CustomLRScheduler
        | None
    )
    epoch: int
    batch_size: int
    lr: float
    loss_scale: float
    eval_every_eps: int

    def __init__(self, params, config: dict):
        # New: configurable betas / eps with stable defaults
        betas = (config["beta1"], config["beta2"])
        eps = config["eps"]

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            betas=betas,
            eps=eps,
        )
        self.epoch = config["epoch"]
        self.lr = config["lr"]
        self.loss_scale = config["loss_scale"]
        self.eval_every_eps = config["eval_every_eps"]
        self.early_save_metric = config["early_save_metric"].lower()

        if self.early_save_metric not in ["train", "val"]:
            raise ValueError("`early_save_metric` must be either 'train' or 'val'.")

        if config["scheduler"] == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"]
            )
        elif config["scheduler"] == "cos":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["scheduler_t_max"], eta_min=config["scheduler_eta_min"]
            )
        elif config["scheduler"] == "exp":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=config["scheduler_gamma"])
        elif config["scheduler"] == "mix":
            warmup_epochs = int(0.1 * self.epoch)
            exp_decay_epochs = int(0.05 * self.epoch)
            cosine_epochs = self.epoch - warmup_epochs - exp_decay_epochs
            if warmup_epochs == 0:
                warmup_epochs = 1
                cosine_epochs -= 1
            if exp_decay_epochs == 0:
                exp_decay_epochs = 1
                cosine_epochs -= 1
            self.scheduler = CustomLRScheduler(
                optimizer=self.optimizer,
                total_epochs=self.epoch,
                warmup_epochs=warmup_epochs,
                cosine_epochs=cosine_epochs,
                exp_decay_epochs=exp_decay_epochs,
                initial_lr=self.lr,
                max_lr=config["max_lr"],
                min_lr=config["min_lr"],
                final_lr=config["final_lr"],
            )
        else:
            self.scheduler = None

    @staticmethod
    def _compute_grad_norms(model) -> dict[str, float]:
        """Compute per-component gradient norms for monitoring."""
        norms: dict[str, float] = {}
        total_norm_sq = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                pnorm = p.grad.data.norm(2).item()
                total_norm_sq += pnorm**2
                # Group by top-level component (encoder, processor, decoder, attn_pool, etc.)
                component = name.split(".")[0]
                norms.setdefault(component, 0.0)
                norms[component] += pnorm**2
        norms = {f"grad_norm/{k}": v**0.5 for k, v in norms.items()}
        norms["grad_norm/total"] = total_norm_sq**0.5
        return norms

    def optimize(self, trainer: TrainerBase, description: str = "AdamWOptimizer", color: str = "blue"):  # noqa: PLR0912, PLR0915
        time_total = 0.0
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses = []
        epochs = []
        val_epochs = []
        val_losses = []

        # Gradient clipping threshold
        max_grad_norm = 1.0

        # Only rank 0 should display progress bar
        disable_pbar = trainer.setup_config.get("rank", 0) != 0
        pbar = tqdm(total=self.epoch, desc=description, colour=color, disable=disable_pbar)
        global_step = 0
        save_every_steps = trainer.optimizer_config["args"]["save_every_steps"]

        # Per-epoch CSV logging (rank 0 only)
        _csv_file = None
        _csv_writer = None
        if not disable_pbar:
            _csv_path = trainer.path_config["result_path"].replace(".png", "_epoch_metrics.csv")
            os.makedirs(os.path.dirname(_csv_path), exist_ok=True)
            _csv_file = open(_csv_path, "w", newline="")  # noqa: SIM115
            _csv_writer = csv.DictWriter(
                _csv_file,
                fieldnames=["epoch", "train_loss", "val_loss", "val_rel_l2", "lr"],
                extrasaction="ignore",
            )
            _csv_writer.writeheader()
        for epoch in range(self.epoch):
            # Ensure DistributedSampler shuffles differently each epoch
            train_sampler = trainer.train_loader.sampler
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            trainer.model.train()
            total_loss: float = 0.0
            epoch_grad_norms: dict[str, float] = {}
            num_batches = 0

            for batch in trainer.train_loader:
                self.optimizer.zero_grad()

                # trainer.train_step now returns (total_weighted_loss, dict_of_head_mses)
                with trainer.autocast_ctx:
                    train_loss, head_losses = trainer.train_step(batch)

                train_loss.backward()

                # Accumulate individual head MSEs for logging
                for h, val in head_losses.items():
                    # sigma_* values are point-in-time (not cumulative MSE) — use last value
                    if h.startswith("sigma_"):
                        key = f"train/{h}"
                        epoch_grad_norms[key] = val  # overwrite: take latest batch value
                    else:
                        key = f"train/MSE_{h}"
                        epoch_grad_norms[key] = epoch_grad_norms.get(key, 0.0) + val

                # Gradient clipping to stabilise training
                # Include uncertainty-weighting sigma params if present
                params_to_clip = list(trainer.model.parameters())
                log_sigma_field = getattr(trainer, "log_sigma_field", None)
                log_sigma_scalar = getattr(trainer, "log_sigma_scalar", None)
                if log_sigma_field is not None and log_sigma_scalar is not None:
                    params_to_clip += [log_sigma_field, log_sigma_scalar]
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)

                # Accumulate gradient norms (sampled every batch)
                if not disable_pbar:
                    batch_norms = self._compute_grad_norms(trainer.model)
                    for k, v in batch_norms.items():
                        epoch_grad_norms.setdefault(k, 0.0)
                        epoch_grad_norms[k] += v

                # Scalar encoder diagnostics (fusion ratio + scale factor stats)
                raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                sc_enc = getattr(raw_model, "scalar_encoder", None)
                if sc_enc is not None:
                    tok_n = getattr(sc_enc, "_last_token_norm", 0.0)
                    sc_n = getattr(sc_enc, "_last_scalar_norm", 0.0)
                    ratio = sc_n / tok_n if tok_n > 0 else 0.0
                    epoch_grad_norms["scalar_encoder/fusion_ratio"] = (
                        epoch_grad_norms.get("scalar_encoder/fusion_ratio", 0.0) + ratio
                    )
                    sf = getattr(batch, "scale_factors", None)
                    if sf is not None:
                        sf_flat = sf.detach().float()
                        epoch_grad_norms["scalar_encoder/sf_mean"] = (
                            epoch_grad_norms.get("scalar_encoder/sf_mean", 0.0) + sf_flat.mean().item()
                        )
                        epoch_grad_norms["scalar_encoder/sf_std"] = (
                            epoch_grad_norms.get("scalar_encoder/sf_std", 0.0) + sf_flat.std().item()
                        )
                        epoch_grad_norms["scalar_encoder/sf_min"] = min(
                            epoch_grad_norms.get("scalar_encoder/sf_min", float("inf")), sf_flat.min().item()
                        )
                        epoch_grad_norms["scalar_encoder/sf_max"] = max(
                            epoch_grad_norms.get("scalar_encoder/sf_max", float("-inf")), sf_flat.max().item()
                        )

                num_batches += 1
                self.optimizer.step()
                total_loss += train_loss.detach().item()
                global_step += 1

                # Save checkpoint every N steps if configured (rank 0 only)
                if save_every_steps > 0 and global_step % save_every_steps == 0 and trainer.setup_config.get("rank", 0) == 0:
                    try:
                        step_loss = train_loss.detach().cpu().item()
                        trainer.save_ckpt()
                        logging.getLogger(__name__).info(
                            f"Step checkpoint saved at global_step={global_step} (epoch {epoch + 1}), loss={step_loss:.4f}"
                        )
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Failed to save step checkpoint: {e}")

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.update(1)

            if (epoch + 1) % self.eval_every_eps == 0:
                # Average total loss across all batches
                avg_train_loss = total_loss / num_batches
                losses.append(avg_train_loss)
                epochs.append(epoch)

                # Capture validation metrics
                val_metrics = trainer.validate(trainer.val_loader)
                val_loss = val_metrics["loss"]
                val_mse = val_metrics["mse"]
                val_rel_l2 = val_metrics["rel_l2"]

                current_lr = self.optimizer.param_groups[0]["lr"]

                if not disable_pbar:
                    pbar.set_postfix({"train_loss": avg_train_loss, "val_MSE": val_mse, "rel_l2": val_rel_l2})

                # Construct the final metrics dictionary for WandB
                metrics = {
                    "train/loss_total": avg_train_loss,
                    "train/MSE_total": avg_train_loss,  # updated below if per-head MSEs available
                    "val/loss": val_loss,
                    "val/MSE": val_mse,
                    "val/rel_l2": val_rel_l2,
                    "lr": current_lr,
                }
                # Add per-head val losses (force, cd, field)
                for k, v in val_metrics.items():
                    if k.startswith("head_"):
                        metrics[f"val/MSE_{k[5:]}"] = v

                # Add averaged head-specific MSEs and Gradient Norms
                if epoch_grad_norms and num_batches > 0:
                    train_mse_sum = 0.0
                    train_mse_count = 0
                    for k, v in epoch_grad_norms.items():
                        # sigma values and sf_min/sf_max are point-in-time, not accumulated sums
                        if "sigma_" in k or k.endswith(("/sf_min", "/sf_max")):
                            metrics[k] = v
                        else:
                            metrics[k] = v / num_batches
                        # Aggregate per-head MSEs to get overall train MSE
                        if k.startswith("train/MSE_"):
                            train_mse_sum += v / num_batches
                            train_mse_count += 1
                    if train_mse_count > 0:
                        metrics["train/MSE_total"] = train_mse_sum / train_mse_count

                val_losses.append(val_loss)
                val_epochs.append(epoch)

                # Per-epoch visualization logging
                if hasattr(trainer, "visualize_sample"):
                    try:
                        trainer.visualize_sample(epoch)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Visualization failed at epoch {epoch + 1}: {e}")
                    trainer.model.train()

                metrics["memory"] = psutil.Process(os.getpid()).memory_info().rss / 1e9

                # Detailed GPU memory breakdown (all in MB)
                if torch.cuda.is_available():
                    metrics["gpu/allocated_MB"] = torch.cuda.memory_allocated() / 1e6
                    metrics["gpu/reserved_MB"] = torch.cuda.memory_reserved() / 1e6
                    metrics["gpu/max_allocated_MB"] = torch.cuda.max_memory_allocated() / 1e6
                    metrics["gpu/max_reserved_MB"] = torch.cuda.max_memory_reserved() / 1e6
                    # free = reserved by PyTorch but not currently used by tensors
                    metrics["gpu/cache_free_MB"] = (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e6

                trainer.tracker.log_metrics(metrics, step=epoch + 1)

                # Write per-epoch row to CSV (rank 0 only)
                if _csv_writer is not None and _csv_file is not None:
                    _csv_writer.writerow(
                        {
                            "epoch": epoch + 1,
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                            "val_rel_l2": val_rel_l2,
                            "lr": current_lr,
                        }
                    )
                    _csv_file.flush()

                # Early saving logic
                current_loss = val_loss if self.early_save_metric == "val" else avg_train_loss

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_epoch = epoch
                    best_state = deepcopy(trainer.model.state_dict())

                # Periodic Artifact Saving
                if (
                    trainer.optimizer_config["args"].get("save_every_epochs", 0) > 0
                    and (epoch + 1) % trainer.optimizer_config["args"]["save_every_epochs"] == 0
                    and trainer.setup_config.get("rank", 0) == 0
                ):
                    try:
                        trainer.save_ckpt()
                        trainer.tracker.save_checkpoint_artifact(
                            epoch=epoch + 1,
                            train_loss=avg_train_loss,
                            val_loss=val_loss,
                            is_best=False,
                            config_dict=trainer.full_config_dict,
                        )
                        if trainer.setup_config.get("test_during_training", False):
                            trainer.test()
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"Periodic save failed: {e}")

        if best_state is not None:
            trainer.model.load_state_dict(best_state)

        if _csv_file is not None:
            _csv_file.close()

        if not disable_pbar:
            pbar.close()

        return {
            "train": {"loss": losses, "epoch": epochs},
            "valid": {"loss": val_losses, "epoch": val_epochs},
            "best": {"epoch": best_epoch, "loss": best_loss},
            "time": time_total,
        }
