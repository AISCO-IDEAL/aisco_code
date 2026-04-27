"""ValidationCalculator is called with config and trainer context to evaluate validation batches."""

from __future__ import annotations

import logging
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.model.gaot_3d import masked_multi_head_loss
from src.utils.pressure_integration import cp_field_to_drag, cp_field_to_drag_pointcloud


class ValidationCalculator:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def validate(self, trainer, loader, epoch_name: int):
        trainer.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_rel_l2 = 0.0
        integrated_errors = []
        head_losses: dict[str, float] = {}
        num_batches = len(loader)
        mse_fn = torch.nn.MSELoss()
        with torch.no_grad():
            for batch_raw in loader:
                batch = batch_raw.to(trainer.device)
                pred = trainer.model(batch=batch, tokens_pos=trainer.latent_tokens.to(trainer.device))
                loss, mse, p_normalized, t_normalized = self._batch_losses(trainer, batch, pred, mse_fn, head_losses)
                integrated_errors.extend(self._integrated_drag_errors(trainer, batch, pred))
                total_loss += loss.item()
                total_mse += mse.item()
                valid_mask = ~torch.isnan(t_normalized)
                if valid_mask.any():
                    p_valid = p_normalized[valid_mask]
                    t_valid = t_normalized[valid_mask]
                    total_rel_l2 += (
                        torch.linalg.norm(p_valid - t_valid) / (torch.linalg.norm(t_valid) + 1e-10)
                    ).item()
        local_metrics = torch.tensor([total_loss, total_mse, total_rel_l2], dtype=torch.float64, device=trainer.device)
        world_size = 1
        if trainer.setup_config["distributed"]:
            torch.distributed.all_reduce(local_metrics, op=torch.distributed.ReduceOp.SUM)
            world_size = torch.distributed.get_world_size()
        result = {
            "loss": local_metrics[0].item() / (num_batches * world_size),
            "mse": local_metrics[1].item() / (num_batches * world_size),
            "rel_l2": local_metrics[2].item() / (num_batches * world_size),
        }
        for head_name, value in head_losses.items():
            result[f"head_{head_name}"] = value / max(num_batches, 1)
        if trainer.setup_config["rank"] == 0 and integrated_errors:
            result["average_integrated_drag_error"] = float(np.mean(integrated_errors))
            trainer.tracker.log_metrics(
                {"val/average_integrated/integrated_drag_error": result["average_integrated_drag_error"]},
                step=epoch_name,
            )
        return result

    def visualize_samples(self, trainer, epoch: int) -> None:
        if trainer.setup_config["rank"] != 0 or trainer.dataset_config.get("num_viz_samples", 1) <= 0:
            return
        for split_label, loader in [("train", getattr(trainer, "train_loader", None)), ("val", getattr(trainer, "val_loader", None))]:
            if loader is None:
                continue
            try:
                batch = next(iter(loader)).to(trainer.device)
                with torch.no_grad():
                    pred = trainer.model(batch=batch, tokens_pos=trainer.latent_tokens.to(trainer.device))
                self._log_visual_batch(trainer, split_label, batch, pred, epoch)
            except Exception as exc:
                logging.getLogger(__name__).warning(f"Failed to visualize {split_label} samples at epoch {epoch + 1}: {exc}")

    def _batch_losses(self, trainer, batch, pred, mse_fn, head_losses):
        if isinstance(pred, dict):
            targets = {"force": batch.force, "cd": batch.cd}
            if trainer.active_heads is not None:
                targets = {key: value for key, value in targets.items() if key in trainer.active_heads}
            loss = masked_multi_head_loss(pred, targets, loss_fn=trainer.loss_fn)
            mse = masked_multi_head_loss(pred, targets, loss_fn=mse_fn)
            self._add_head_losses(trainer, pred, targets, head_losses)
            if targets:
                primary_key = trainer.active_heads[0] if trainer.active_heads else next(iter(targets.keys()))
                p_normalized = pred[primary_key]
                t_normalized = targets[primary_key].to(trainer.device).view_as(p_normalized)
            elif "field" in pred and hasattr(batch, "x") and batch.x is not None:
                p_normalized = pred["field"]
                t_normalized = batch.x.to(trainer.device)
            else:
                p_normalized = next(iter(pred.values()))
                t_normalized = p_normalized.detach()
            loss = trainer.loss_handler.add_field_loss(trainer, batch, pred, loss, head_losses)
            if "field" in pred and hasattr(batch, "x") and batch.x is not None:
                field_target_normalized = batch.x.to(trainer.device)
                valid = ~torch.isnan(field_target_normalized)
                if valid.any():
                    mse = mse + trainer.dataset_config["field_loss_weight"] * mse_fn(pred["field"][valid], field_target_normalized[valid])
            return loss, mse, p_normalized, t_normalized
        target_normalized = batch.y
        return trainer.loss_fn(pred, target_normalized), mse_fn(pred, target_normalized), pred, target_normalized

    def _add_head_losses(self, trainer, pred, targets, head_losses) -> None:
        loss_fn_none = torch.nn.MSELoss(reduction="none")
        for head_name, target_raw in targets.items():
            if head_name not in pred or target_raw is None:
                continue
            target_normalized = target_raw.to(trainer.device).view_as(pred[head_name])
            valid = ~torch.isnan(target_normalized)
            if valid.any():
                head_losses[head_name] = head_losses.get(head_name, 0.0) + loss_fn_none(
                    pred[head_name][valid], target_normalized[valid]
                ).mean().item()

    def _integrated_drag_errors(self, trainer, batch, pred) -> list[float]:
        if not isinstance(pred, dict) or "field" not in pred or not hasattr(batch, "x") or batch.x is None:
            return []
        if not hasattr(batch, "c") or batch.c is None:
            return []
        pred_unnormalized = self._denormalize_field(trainer, pred["field"])
        target_unnormalized = self._denormalize_field(trainer, batch.x.to(trainer.device))
        errors = []
        for start, end in self._sample_ranges(batch):
            drag_pred = self._integrate_sample(trainer, batch, pred_unnormalized, start, end)
            drag_target = self._integrate_sample(trainer, batch, target_unnormalized, start, end)
            if drag_pred is not None and drag_target is not None:
                errors.append(abs(drag_pred - drag_target))
        return errors

    def _log_visual_batch(self, trainer, split_label: str, batch, pred, epoch: int) -> None:
        if not isinstance(pred, dict):
            return
        self._log_scalar_visuals(trainer, split_label, batch, pred, epoch)
        if "field" not in pred or not hasattr(batch, "x") or batch.x is None:
            return
        pred_unnormalized = self._denormalize_field(trainer, pred["field"])
        target_unnormalized = self._denormalize_field(trainer, batch.x.to(trainer.device))
        max_samples = min(int(trainer.dataset_config.get("num_viz_samples", 1)), len(self._sample_ranges(batch)))
        for sample_index, (start, end) in enumerate(self._sample_ranges(batch)[:max_samples]):
            self._log_field_visual(trainer, split_label, batch, target_unnormalized, pred_unnormalized, start, end, sample_index, epoch)
            drag_pred = self._integrate_sample(trainer, batch, pred_unnormalized, start, end)
            drag_target = self._integrate_sample(trainer, batch, target_unnormalized, start, end)
            metrics = {}
            if drag_pred is not None:
                metrics[f"viz/{split_label}/integrated_drag_pred_{sample_index}"] = drag_pred
            if drag_target is not None:
                metrics[f"viz/{split_label}/integrated_drag_gt_{sample_index}"] = drag_target
            if drag_pred is not None and drag_target is not None:
                metrics[f"viz/{split_label}/integrated_drag_abs_error_{sample_index}"] = abs(drag_pred - drag_target)
            if metrics:
                trainer.tracker.log_metrics(metrics, step=epoch + 1)

    def _log_scalar_visuals(self, trainer, split_label: str, batch, pred, epoch: int) -> None:
        metrics = {}
        for head_name in ("force", "cd"):
            if head_name not in pred or not hasattr(batch, head_name):
                continue
            target = getattr(batch, head_name)
            if target is None:
                continue
            pred_value, target_value = self._denormalize_head(trainer, head_name, pred[head_name], target.to(trainer.device))
            valid = ~torch.isnan(target_value)
            if valid.any():
                diff = pred_value[valid] - target_value[valid]
                metrics[f"viz/{split_label}/{head_name}_mae"] = torch.mean(torch.abs(diff)).item()
                metrics[f"viz/{split_label}/{head_name}_mse"] = torch.mean(diff**2).item()
        if metrics:
            trainer.tracker.log_metrics(metrics, step=epoch + 1)

    def _log_field_visual(self, trainer, split_label: str, batch, target, pred, start: int, end: int, sample_index: int, epoch: int) -> None:
        pos = batch.pos_raw[start:end] if hasattr(batch, "pos_raw") else batch.pos[start:end]
        coords = pos.detach().cpu().numpy()
        target_values = target[start:end].detach().cpu().numpy()
        pred_values = pred[start:end].detach().cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
            image_path = image_file.name
        try:
            self._save_field_plot(coords, target_values, pred_values, image_path)
            trainer.tracker.log_image(f"viz/{split_label}/field_{sample_index}", image_path, step=epoch + 1)
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    def _save_field_plot(self, coords, target_values, pred_values, image_path: str) -> None:
        target_values = np.asarray(target_values).squeeze()
        pred_values = np.asarray(pred_values).squeeze()
        diff_values = np.abs(pred_values - target_values)
        fig = plt.figure(figsize=(12, 4))
        for index, (title, values) in enumerate([("Ground Truth", target_values), ("Prediction", pred_values), ("Absolute Difference", diff_values)], start=1):
            ax = fig.add_subplot(1, 3, index, projection="3d")
            plot = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=values, s=2, cmap="jet")
            ax.set_title(title)
            fig.colorbar(plot, ax=ax, shrink=0.6)
        fig.tight_layout()
        fig.savefig(image_path, dpi=200)
        plt.close(fig)

    def _denormalize_field(self, trainer, value_normalized):
        u_mean = trainer.u_mean.to(value_normalized.device)
        u_std = trainer.u_std.to(value_normalized.device)
        return value_normalized * u_std + u_mean

    def _denormalize_head(self, trainer, head_name: str, pred_normalized, target_normalized):
        if hasattr(trainer, "per_head_stats") and head_name in trainer.per_head_stats:
            mean = trainer.per_head_stats[head_name]["mean"].to(pred_normalized.device)
            std = trainer.per_head_stats[head_name]["std"].to(pred_normalized.device)
            return pred_normalized * std + mean, target_normalized.view_as(pred_normalized) * std + mean
        return pred_normalized, target_normalized.view_as(pred_normalized)

    def _integrate_sample(self, trainer, batch, field_unnormalized, start: int, end: int) -> float | None:
        pos = batch.pos_raw[start:end] if hasattr(batch, "pos_raw") else batch.pos[start:end]
        normals = batch.c[start:end]
        field = field_unnormalized[start:end].squeeze()
        v_inf = float(trainer.dataset_config.get("v_inf", 25.0))
        rho = float(trainer.dataset_config.get("rho", 1.225))
        try:
            face = self._sample_face(batch, start, end)
            if face is not None:
                return cp_field_to_drag(field, normals, face, pos, v_inf=v_inf, rho=rho).item()
            return cp_field_to_drag_pointcloud(field, normals, pos, v_inf=v_inf, rho=rho).item()
        except Exception:
            logging.getLogger(__name__).debug("Integrated validation drag failed", exc_info=True)
            return None

    def _sample_face(self, batch, start: int, end: int):
        if not hasattr(batch, "face") or batch.face is None:
            return None
        face = batch.face
        valid = ((face >= start) & (face < end)).all(dim=0)
        if not valid.any():
            return None
        return face[:, valid] - start

    def _sample_ranges(self, batch) -> list[tuple[int, int]]:
        if hasattr(batch, "ptr"):
            ptr = batch.ptr.detach().cpu().tolist()
            return [(int(ptr[index]), int(ptr[index + 1])) for index in range(len(ptr) - 1)]
        return [(0, int(batch.pos.shape[0]))]
