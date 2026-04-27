"""LossFn is called with a config dict and exposes the current training loss helpers."""

from __future__ import annotations

import torch

from src.model.gaot_3d import masked_multi_head_loss


class LossFn:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.loss_fn = self._build_loss_fn(self.config["optimizer"]["args"].get("loss", "mse"))

    def calculate_loss(self, pred_normalized: torch.Tensor, target_normalized: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred_normalized, target_normalized)

    def calculate_train_loss(self, trainer, batch):
        strategy = trainer.dataset_config["training_strategy"]
        latent_tokens_normalized = trainer.latent_tokens.to(trainer.device)
        field_target_normalized = None
        if strategy == "neural_field":
            sampled_data_list, query_pos, query_batch_idx, field_target_normalized = trainer.data_handler.sample_nodes_neural_field(
                trainer,
                batch,
                trainer.dataset_config["neural_field_input_nodes"],
                trainer.dataset_config["neural_field_query_nodes_train"],
            )
            sampled_batch = trainer.collate_fn(sampled_data_list).to(trainer.device)
            query_pos = query_pos.to(trainer.device)
            query_batch_idx = query_batch_idx.to(trainer.device)
            if field_target_normalized is not None:
                field_target_normalized = field_target_normalized.to(trainer.device)
            pred_normalized = trainer.model(
                batch=sampled_batch,
                tokens_pos=latent_tokens_normalized,
                query_coord_pos=query_pos,
                query_coord_batch_idx=query_batch_idx,
            )
            batch = sampled_batch
        else:
            batch = batch.to(trainer.device)
            pred_normalized = trainer.model(batch=batch, tokens_pos=latent_tokens_normalized)

        if not isinstance(pred_normalized, dict):
            return self.loss_fn(pred_normalized, batch.y), {}

        targets_normalized = {"force": batch.force, "cd": batch.cd}
        if trainer.active_heads is not None:
            targets_normalized = {key: value for key, value in targets_normalized.items() if key in trainer.active_heads}
        scalar_loss = masked_multi_head_loss(pred_normalized, targets_normalized, loss_fn=self.loss_fn)
        head_losses = self._calculate_scalar_head_losses(trainer, pred_normalized, targets_normalized)

        if "field" not in pred_normalized:
            return scalar_loss, head_losses
        field_target_normalized = self._field_target(trainer, batch, field_target_normalized)
        if field_target_normalized is None:
            return scalar_loss, head_losses
        field_loss = self._calculate_field_loss(trainer, pred_normalized["field"], field_target_normalized)
        head_losses["field"] = field_loss.item()
        if trainer._uncertainty_weighting:
            return self._combine_uncertainty_loss(trainer, scalar_loss, field_loss, head_losses)
        return scalar_loss + trainer.dataset_config["field_loss_weight"] * field_loss, head_losses

    def add_field_loss(self, trainer, batch, pred_normalized, loss, head_losses):
        if "field" not in pred_normalized or not hasattr(batch, "x") or batch.x is None:
            return loss
        field_target_normalized = batch.x.to(trainer.device)
        valid = ~torch.isnan(field_target_normalized)
        if valid.any():
            field_loss = self.loss_fn(pred_normalized["field"][valid], field_target_normalized[valid])
            head_losses["field"] = head_losses.get("field", 0.0) + field_loss.item()
            loss = loss + trainer.dataset_config["field_loss_weight"] * field_loss
        return loss

    def _build_loss_fn(self, loss_name: str):
        if loss_name == "mse":
            return torch.nn.MSELoss()
        if loss_name == "l2":
            return self._calculate_l2_loss
        if loss_name == "relative_l2":
            return self._calculate_relative_l2_loss
        raise ValueError(f"Unknown loss '{loss_name}'. Supported: ['mse', 'l2', 'relative_l2']")

    def _calculate_relative_l2_loss(self, pred_normalized: torch.Tensor, target_normalized: torch.Tensor) -> torch.Tensor:
        if pred_normalized.ndim == 1:
            pred_normalized = pred_normalized.unsqueeze(0)
            target_normalized = target_normalized.unsqueeze(0)
        diff_normalized = pred_normalized - target_normalized
        diff_norm = diff_normalized.flatten(1).norm(p=2, dim=1)
        target_norm = target_normalized.flatten(1).norm(p=2, dim=1).clamp(min=1e-8)
        return (diff_norm / target_norm).mean()

    def _calculate_l2_loss(self, pred_normalized: torch.Tensor, target_normalized: torch.Tensor) -> torch.Tensor:
        if pred_normalized.ndim == 1:
            pred_normalized = pred_normalized.unsqueeze(0)
            target_normalized = target_normalized.unsqueeze(0)
        diff_normalized = pred_normalized - target_normalized
        return diff_normalized.flatten(1).norm(p=2, dim=1).mean()

    def _calculate_scalar_head_losses(self, trainer, pred_normalized, targets_normalized):
        head_losses = {}
        loss_fn_none = torch.nn.MSELoss(reduction="none")
        for name in targets_normalized:
            if name not in pred_normalized:
                continue
            target_normalized = targets_normalized[name]
            if target_normalized is None:
                continue
            target_normalized = target_normalized.to(trainer.device).view_as(pred_normalized[name])
            valid = ~torch.isnan(target_normalized)
            if valid.any():
                head_losses[name] = loss_fn_none(pred_normalized[name][valid], target_normalized[valid]).mean().item()
        return head_losses

    def _field_target(self, trainer, batch, field_target_normalized):
        if field_target_normalized is not None:
            return field_target_normalized
        if hasattr(batch, "x") and batch.x is not None:
            return batch.x.to(trainer.device)
        return None

    def _calculate_field_loss(self, trainer, field_pred_normalized, field_target_normalized):
        valid = ~torch.isnan(field_target_normalized)
        if valid.any():
            return self.loss_fn(field_pred_normalized[valid], field_target_normalized[valid])
        return torch.tensor(0.0, device=trainer.device)

    def _combine_uncertainty_loss(self, trainer, scalar_loss, field_loss, head_losses):
        head_losses["sigma_field"] = torch.exp(trainer.log_sigma_field).item()
        head_losses["sigma_scalar"] = torch.exp(trainer.log_sigma_scalar).item()
        combined = (
            0.5 * torch.exp(-2.0 * trainer.log_sigma_field) * field_loss
            + trainer.log_sigma_field
            + 0.5 * torch.exp(-2.0 * trainer.log_sigma_scalar) * scalar_loss
            + trainer.log_sigma_scalar
        )
        return combined.squeeze(), head_losses
