"""Field loss helpers used by the GAOT3 trainer."""

from __future__ import annotations

import torch


class LossFn:
    def __init__(self, config: dict) -> None:
        self.config: dict = config
        self.loss_fn = self._build_loss_fn(self.config["optimizer"]["args"].get("loss", "mse"))

    def calculate_loss(self, pred_normalized: torch.Tensor, target_normalized: torch.Tensor) -> torch.Tensor:
        return self._calculate_field_loss(pred_normalized, target_normalized)

    def calculate_train_loss(self, trainer, batch):
        strategy = trainer.dataset_config["training_strategy"]
        latent_tokens_normalized = trainer.latent_tokens.to(trainer.device)
        field_target_normalized = None
        if strategy == "neural_field":
            sampled_data_list, query_pos, query_batch_idx, field_target_normalized = (
                trainer.data_handler.sample_nodes_neural_field(
                    trainer,
                    batch,
                    trainer.dataset_config["neural_field_input_nodes"],
                    trainer.dataset_config["neural_field_query_nodes_train"],
                )
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
            field_target_normalized = self._field_target(trainer, batch, field_target_normalized)
            if field_target_normalized is None:
                raise ValueError("Field loss requires a field target, but neither sampled targets nor batch.x were found.")
            field_loss = self._calculate_field_loss(pred_normalized, field_target_normalized)
            return field_loss, {"field": field_loss.item()}
        if "field" not in pred_normalized:
            raise ValueError("Field loss requires the model output dict to contain a 'field' prediction.")
        field_target_normalized = self._field_target(trainer, batch, field_target_normalized)
        if field_target_normalized is None:
            raise ValueError("Field loss requires a field target, but neither sampled targets nor batch.x were found.")
        field_loss = self._calculate_field_loss(pred_normalized["field"], field_target_normalized)
        return trainer.dataset_config["field_loss_weight"] * field_loss, {"field": field_loss.item()}

    def add_field_loss(self, trainer, batch, pred_normalized, loss, head_losses):
        if "field" not in pred_normalized or not hasattr(batch, "x") or batch.x is None:
            return loss
        field_target_normalized = batch.x.to(trainer.device)
        field_loss = self._calculate_field_loss(pred_normalized["field"], field_target_normalized)
        head_losses["field"] = head_losses.get("field", 0.0) + field_loss.item()
        return loss + trainer.dataset_config["field_loss_weight"] * field_loss

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

    def _field_target(self, trainer, batch, field_target_normalized):
        if field_target_normalized is not None:
            return field_target_normalized
        if hasattr(batch, "x") and batch.x is not None:
            return batch.x.to(trainer.device)
        return None

    def _calculate_field_loss(self, field_pred_normalized, field_target_normalized):
        valid = ~torch.isnan(field_target_normalized)
        if valid.any():
            return self.loss_fn(field_pred_normalized[valid], field_target_normalized[valid])
        return torch.tensor(0.0, device=field_pred_normalized.device)
