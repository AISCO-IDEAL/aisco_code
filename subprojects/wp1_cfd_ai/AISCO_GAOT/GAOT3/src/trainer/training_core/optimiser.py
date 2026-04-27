"""Optimiser is called with config and owns AdamW setup plus the GAOT3 epoch loop."""

from __future__ import annotations

import csv
import logging
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from typing import IO, Any, TypedDict

import numpy as np
import psutil
import torch
from tqdm import tqdm


class TrainMetrics(TypedDict):
    total_loss: float
    num_batches: int
    epoch_values: dict[Any, Any]


class Optimiser:
    def __init__(self, params, config: dict) -> None:
        self.config: dict = config
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config["optimizer"]["args"]["lr"],
            weight_decay=self.config["optimizer"]["args"]["weight_decay"],
            betas=(self.config["optimizer"]["args"]["beta1"], self.config["optimizer"]["args"]["beta2"]),
            eps=self.config["optimizer"]["args"]["eps"],
        )

    def optimise(self, training_context, description: str = "AdamWOptimizer", color: str = "blue"):
        start_time = time.time()
        best_loss, best_epoch, best_state = np.inf, -1, None
        losses, epochs, val_losses, val_epochs = [], [], [], []
        disable_pbar = training_context.setup_config.get("rank", 0) != 0
        pbar = tqdm(total=self.config["optimizer"]["args"]["epoch"], desc=description, colour=color, disable=disable_pbar)
        with self._epoch_csv(training_context, disable_pbar) as (csv_file, csv_writer):
            global_step = 0
            for epoch in range(self.config["optimizer"]["args"]["epoch"]):
                train_sampler = training_context.train_loader.sampler
                if hasattr(train_sampler, "set_epoch"):
                    train_sampler.set_epoch(epoch)
                train_metrics, global_step = self._train_epoch(training_context, global_step)
                self._step_scheduler(epoch)
                pbar.update(1)
                if (epoch + 1) % self.config["optimizer"]["args"]["eval_every_eps"] != 0:
                    continue
                avg_train_loss = train_metrics["total_loss"] / max(train_metrics["num_batches"], 1)
                val_metrics = training_context.validate(training_context.val_loader, epoch_name=epoch)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self._log_epoch(
                    training_context, epoch, avg_train_loss, val_metrics, train_metrics, current_lr, pbar, disable_pbar
                )
                self._write_epoch_csv(csv_writer, csv_file, epoch, avg_train_loss, val_metrics, current_lr)
                losses.append(avg_train_loss)
                epochs.append(epoch)
                val_losses.append(val_metrics["loss"])
                val_epochs.append(epoch)
                current_loss = (
                    val_metrics["loss"]
                    if self.config["optimizer"]["args"]["early_save_metric"].lower() == "val"
                    else avg_train_loss
                )
                if current_loss < best_loss:
                    best_loss, best_epoch, best_state = current_loss, epoch, deepcopy(training_context.model.state_dict())
                self._periodic_save(training_context, epoch, avg_train_loss, val_metrics["loss"])
        if best_state is not None:
            training_context.model.load_state_dict(best_state)
        if not disable_pbar:
            pbar.close()
        return {
            "train": {"loss": losses, "epoch": epochs},
            "valid": {"loss": val_losses, "epoch": val_epochs},
            "best": {"epoch": best_epoch, "loss": best_loss},
            "time": time.time() - start_time,
        }

    def _train_epoch(self, training_context, global_step: int) -> tuple[TrainMetrics, int]:
        training_context.model.train()
        metrics: TrainMetrics = {"total_loss": 0.0, "num_batches": 0, "epoch_values": {}}
        epoch_values = metrics["epoch_values"]
        for batch in training_context.train_loader:
            self.optimizer.zero_grad()
            with training_context.autocast_ctx:
                train_loss, head_losses = training_context.train_step(batch)
            train_loss.backward()
            self._accumulate_head_losses(epoch_values, head_losses)
            self._clip_gradients(training_context)
            self._accumulate_grad_norms(training_context, epoch_values)
            self._accumulate_scalar_diagnostics(training_context, epoch_values)
            self.optimizer.step()
            metrics["total_loss"] += train_loss.detach().item()
            metrics["num_batches"] += 1
            global_step += 1
            self._save_step_checkpoint(training_context, global_step, metrics["num_batches"], train_loss)
        return metrics, global_step

    @contextmanager
    def _epoch_csv(self, training_context, disable_pbar: bool):
        if disable_pbar:
            yield None, None
            return
        csv_path = training_context.path_config["result_path"].replace(".png", "_epoch_metrics.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file, fieldnames=["epoch", "train_loss", "val_loss", "val_rel_l2", "lr"], extrasaction="ignore"
            )
            csv_writer.writeheader()
            yield csv_file, csv_writer

    def _step_scheduler(self, epoch: int) -> None:
        args = self.config["optimizer"]["args"]
        if (args["scheduler"] == "step" and (epoch + 1) % args["scheduler_step_size"] == 0) or args["scheduler"] == "exp":
            self._scale_lr(args["scheduler_gamma"])
        elif args["scheduler"] == "cos":
            lr = (
                args["scheduler_eta_min"]
                + (args["lr"] - args["scheduler_eta_min"]) * (1 + np.cos(np.pi * (epoch + 1) / args["scheduler_t_max"])) / 2
            )
            self._set_lr(lr)
        elif args["scheduler"] == "mix":
            self._set_lr(self._mix_lr(epoch + 1))

    def _mix_lr(self, epoch: int) -> float:
        args = self.config["optimizer"]["args"]
        warmup_epochs = max(1, int(0.1 * args["epoch"]))
        exp_decay_epochs = max(1, int(0.05 * args["epoch"]))
        cosine_epochs = args["epoch"] - warmup_epochs - exp_decay_epochs
        if epoch < warmup_epochs:
            return args["lr"] + (args["max_lr"] - args["lr"]) * (epoch / max(1, warmup_epochs - 1))
        if epoch < warmup_epochs + cosine_epochs:
            ratio = (1 + np.cos(np.pi * (epoch - warmup_epochs) / cosine_epochs)) / 2
            return args["min_lr"] + (args["max_lr"] - args["min_lr"]) * ratio
        decay_epoch = epoch - warmup_epochs - cosine_epochs
        return args["min_lr"] * ((args["final_lr"] / args["min_lr"]) ** (decay_epoch / max(1, exp_decay_epochs - 1)))

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _scale_lr(self, scale: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] *= scale

    def _accumulate_head_losses(self, epoch_values: dict, head_losses: dict) -> None:
        for key, value in head_losses.items():
            metric_key = f"train/{key}" if key.startswith("sigma_") else f"train/MSE_{key}"
            epoch_values[metric_key] = value if key.startswith("sigma_") else epoch_values.get(metric_key, 0.0) + value

    def _clip_gradients(self, training_context) -> None:
        params_to_clip = list(training_context.model.parameters())
        if (
            getattr(training_context, "log_sigma_field", None) is not None
            and getattr(training_context, "log_sigma_scalar", None) is not None
        ):
            params_to_clip += [training_context.log_sigma_field, training_context.log_sigma_scalar]
        torch.nn.utils.clip_grad_norm_(params_to_clip, 1.0)

    def _accumulate_grad_norms(self, training_context, epoch_values: dict) -> None:
        for name, value in self._grad_norms(training_context.model).items():
            epoch_values[name] = epoch_values.get(name, 0.0) + value

    def _accumulate_scalar_diagnostics(self, training_context, epoch_values: dict) -> None:
        raw_model = training_context.model.module if hasattr(training_context.model, "module") else training_context.model
        for enc_attr, prefix in [("scalar_encoder", "scalar_enc"), ("late_scalar_encoder", "late_scalar_enc")]:
            encoder = getattr(raw_model, enc_attr, None)
            if encoder is None:
                continue
            token_norm = getattr(encoder, "_last_token_norm", 0.0)
            if getattr(encoder, "mode", "concat") == "concat":
                scalar_norm = getattr(encoder, "_last_scalar_norm", 0.0)
                epoch_values[f"{prefix}/fusion_ratio"] = epoch_values.get(f"{prefix}/fusion_ratio", 0.0) + (
                    scalar_norm / token_norm if token_norm > 0 else 0.0
                )
            elif getattr(encoder, "mode", "concat") == "film":
                for metric, attr, default in [
                    ("film_gamma_mean", "_last_gamma_mean", 1.0),
                    ("film_beta_mean", "_last_beta_mean", 0.0),
                    ("film_gamma_std", "_last_gamma_std", 0.0),
                    ("film_perturbation", "_last_film_perturbation", 0.0),
                ]:
                    epoch_values[f"{prefix}/{metric}"] = epoch_values.get(f"{prefix}/{metric}", 0.0) + getattr(
                        encoder, attr, default
                    )
            epoch_values[f"{prefix}/token_norm"] = epoch_values.get(f"{prefix}/token_norm", 0.0) + token_norm
        encoder = getattr(raw_model, "scalar_encoder", None) or getattr(raw_model, "late_scalar_encoder", None)
        scale_factor = getattr(encoder, "_last_sf", None) if encoder is not None else None
        if scale_factor is not None:
            epoch_values["scalar_enc/sf_mean"] = epoch_values.get("scalar_enc/sf_mean", 0.0) + scale_factor["mean"]
            epoch_values["scalar_enc/sf_std"] = epoch_values.get("scalar_enc/sf_std", 0.0) + scale_factor["std"]
            epoch_values["scalar_enc/sf_min"] = min(epoch_values.get("scalar_enc/sf_min", float("inf")), scale_factor["min"])
            epoch_values["scalar_enc/sf_max"] = max(epoch_values.get("scalar_enc/sf_max", float("-inf")), scale_factor["max"])

    def _log_epoch(
        self,
        training_context,
        epoch: int,
        avg_train_loss: float,
        val_metrics: dict,
        train_metrics: TrainMetrics,
        current_lr: float,
        pbar,
        disable_pbar: bool,
    ) -> None:
        metrics = {
            "train/loss_total": avg_train_loss,
            "train/MSE_total": avg_train_loss,
            "val/loss": val_metrics["loss"],
            "val/MSE": val_metrics["mse"],
            "val/rel_l2": val_metrics["rel_l2"],
            "lr": current_lr,
        }
        if not disable_pbar:
            pbar.set_postfix({"train_loss": avg_train_loss, "val_MSE": val_metrics["mse"], "rel_l2": val_metrics["rel_l2"]})
        for key, value in val_metrics.items():
            if key.startswith("head_"):
                metrics[f"val/MSE_{key[5:]}"] = value
        self._average_epoch_values(metrics, train_metrics["epoch_values"], train_metrics["num_batches"])
        metrics["memory"] = psutil.Process(os.getpid()).memory_info().rss / 1e9
        if torch.cuda.is_available():
            metrics.update(
                {
                    "gpu/allocated_MB": torch.cuda.memory_allocated() / 1e6,
                    "gpu/reserved_MB": torch.cuda.memory_reserved() / 1e6,
                    "gpu/max_allocated_MB": torch.cuda.max_memory_allocated() / 1e6,
                    "gpu/max_reserved_MB": torch.cuda.max_memory_reserved() / 1e6,
                    "gpu/cache_free_MB": (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1e6,
                }
            )
        training_context.tracker.log_metrics(metrics, step=epoch + 1)
        if epoch < self.config["optimizer"]["args"].get("viz_warmup_epochs", 0) or (
            self.config["optimizer"]["args"].get("viz_every_eps", self.config["optimizer"]["args"]["eval_every_eps"]) > 0
            and (epoch + 1)
            % self.config["optimizer"]["args"].get("viz_every_eps", self.config["optimizer"]["args"]["eval_every_eps"])
            == 0
        ):
            try:
                training_context.visualize_sample(epoch)
            except Exception as exc:
                logging.getLogger(__name__).warning(f"Visualization failed at epoch {epoch + 1}: {exc}")
            training_context.model.train()

    def _average_epoch_values(self, metrics: dict, epoch_values: dict, num_batches: int) -> None:
        train_mse_sum, train_mse_count = 0.0, 0
        for key, value in epoch_values.items():
            metrics[key] = value if "sigma_" in key or key.endswith(("/sf_min", "/sf_max")) else value / max(num_batches, 1)
            if key.startswith("train/MSE_"):
                train_mse_sum += value / max(num_batches, 1)
                train_mse_count += 1
        if train_mse_count > 0:
            metrics["train/MSE_total"] = train_mse_sum / train_mse_count

    def _write_epoch_csv(
        self,
        csv_writer: csv.DictWriter | None,
        csv_file: IO[str] | None,
        epoch: int,
        avg_train_loss: float,
        val_metrics: dict,
        current_lr: float,
    ) -> None:
        if csv_writer is None or csv_file is None:
            return
        csv_writer.writerow(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_metrics["loss"],
                "val_rel_l2": val_metrics["rel_l2"],
                "lr": current_lr,
            }
        )
        csv_file.flush()

    def _periodic_save(self, training_context, epoch: int, avg_train_loss: float, val_loss: float) -> None:
        if (
            self.config["optimizer"]["args"].get("save_every_epochs", 0) <= 0
            or (epoch + 1) % self.config["optimizer"]["args"]["save_every_epochs"] != 0
            or training_context.setup_config.get("rank", 0) != 0
        ):
            return
        try:
            training_context.save_ckpt()
            training_context.tracker.save_checkpoint_artifact(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                is_best=False,
                config_dict=training_context.full_config_dict,
            )
            if training_context.setup_config.get("test_during_training", False):
                training_context.test()
        except Exception as exc:
            logging.getLogger(__name__).warning(f"Periodic save failed: {exc}")

    def _save_step_checkpoint(self, training_context, global_step: int, batch_index: int, train_loss) -> None:
        if (
            self.config["optimizer"]["args"]["save_every_steps"] <= 0
            or global_step % self.config["optimizer"]["args"]["save_every_steps"] != 0
            or training_context.setup_config.get("rank", 0) != 0
        ):
            return
        try:
            training_context.save_ckpt()
            logging.getLogger(__name__).info(
                f"Step checkpoint saved at global_step={global_step} (batch {batch_index}), loss={train_loss.detach().cpu().item():.4f}"
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(f"Failed to save step checkpoint: {exc}")

    def _grad_norms(self, model) -> dict[str, float]:
        norms: dict[str, float] = {}
        total_norm_sq = 0.0
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            param_norm = param.grad.data.norm(2).item()
            total_norm_sq += param_norm**2
            component = name.split(".")[0]
            norms[component] = norms.get(component, 0.0) + param_norm**2
        norms = {f"grad_norm/{key}": value**0.5 for key, value in norms.items()}
        norms["grad_norm/total"] = total_norm_sq**0.5
        return norms
