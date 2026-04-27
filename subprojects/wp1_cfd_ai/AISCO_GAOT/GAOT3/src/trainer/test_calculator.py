"""TestCalculator is called with config and trainer context to run inference and output metrics."""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any

import torch

from .metric_calculator import MetricCalculator
from .prediction_writer import PredictionWriter

EPSILON = 1e-10


class TestCalculator:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def run_test(self, trainer) -> None:
        trainer.model.eval()
        collected = self._empty_collections(trainer)
        inference_times = []
        data_loading_times = []
        total_samples = 0
        logging.getLogger(__name__).info(f"Starting testing with metric suite: '{trainer.dataset_config['metric_suite']}'")
        with torch.no_grad():
            for batch_index, batch_raw in enumerate(trainer.test_loader):
                data_start_time = time.time()
                batch = batch_raw.to(trainer.device)
                if trainer.device.type == "cuda":
                    torch.cuda.synchronize()
                data_loading_times.append(time.time() - data_start_time)
                inference_start_time = time.time()
                pred_normalized = trainer.model(batch, trainer.latent_tokens.to(trainer.device))
                if trainer.device.type == "cuda":
                    torch.cuda.synchronize()
                inference_times.append(time.time() - inference_start_time)
                total_samples += batch.num_graphs if hasattr(batch, "num_graphs") else 1
                self._collect_batch(trainer, batch, pred_normalized, collected, batch_index)
        if trainer.setup_config["rank"] == 0:
            self._finalize(trainer, collected, inference_times, data_loading_times, total_samples)

    def _empty_collections(self, trainer) -> dict[str, Any]:
        raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        multi_head = hasattr(raw_model, "decoder") and hasattr(raw_model.decoder, "head_names")
        return {
            "multi_head": multi_head,
            "targets": {"force": [], "cd": []} if multi_head else [],
            "preds": {"force": [], "cd": []} if multi_head else [],
            "field_targets": [],
            "field_preds": [],
            "field_sample_metrics": [],
        }

    def _collect_batch(self, trainer, batch, pred_normalized, collected, batch_index: int) -> None:
        if isinstance(pred_normalized, dict):
            pred_unnormalized, target_unnormalized = self._denormalize_heads(trainer, batch, pred_normalized)
            for head_name, head_pred_unnormalized in pred_unnormalized.items():
                valid = ~torch.isnan(target_unnormalized[head_name])
                if valid.any():
                    collected["targets"][head_name].append(target_unnormalized[head_name][valid].cpu())
                    collected["preds"][head_name].append(head_pred_unnormalized[valid].detach().cpu())
            if "field" in pred_normalized and hasattr(batch, "x") and batch.x is not None:
                self._collect_field(trainer, batch, pred_normalized["field"], collected)
            if batch_index == 0 and trainer.setup_config["rank"] == 0:
                logging.getLogger(__name__).info("First test sample predictions collected.")
            return
        target_normalized = batch.y
        u_std = trainer.u_std[trainer.dataset_config["active_variables"]].to(trainer.device)
        u_mean = trainer.u_mean[trainer.dataset_config["active_variables"]].to(trainer.device)
        pred_unnormalized = pred_normalized * u_std + u_mean
        target_unnormalized = target_normalized * u_std + u_mean
        collected["targets"].append(target_unnormalized.detach().cpu())
        collected["preds"].append(pred_unnormalized.detach().cpu())

    def _denormalize_heads(self, trainer, batch, pred_normalized) -> tuple[dict, dict]:
        pred_unnormalized = {}
        target_unnormalized = {}
        for head_name, head_pred_normalized in pred_normalized.items():
            if head_name == "field":
                continue
            head_target_normalized = getattr(batch, head_name, None)
            if head_target_normalized is None:
                continue
            head_target_normalized = head_target_normalized.to(trainer.device)
            if hasattr(trainer, "per_head_stats") and head_name in trainer.per_head_stats:
                h_mean = trainer.per_head_stats[head_name]["mean"].to(trainer.device)
                h_std = trainer.per_head_stats[head_name]["std"].to(trainer.device)
                pred_unnormalized[head_name] = head_pred_normalized * h_std + h_mean
                target_unnormalized[head_name] = head_target_normalized * h_std + h_mean
            else:
                pred_unnormalized[head_name] = head_pred_normalized
                target_unnormalized[head_name] = head_target_normalized
        return pred_unnormalized, target_unnormalized

    def _collect_field(self, trainer, batch, field_pred_normalized, collected) -> None:
        field_target_normalized = batch.x.to(trainer.device)
        u_mean = trainer.u_mean.to(trainer.device)
        u_std = trainer.u_std.to(trainer.device)
        field_pred_unnormalized = field_pred_normalized * u_std + u_mean
        field_target_unnormalized = field_target_normalized * u_std + u_mean
        valid = ~torch.isnan(field_target_unnormalized)
        if valid.any():
            collected["field_preds"].append(field_pred_unnormalized[valid].detach().cpu())
            collected["field_targets"].append(field_target_unnormalized[valid].cpu())
        ptr = batch.ptr.cpu() if hasattr(batch, "ptr") else None
        filenames = batch.filename if hasattr(batch, "filename") else None
        n_graphs = batch.num_graphs if hasattr(batch, "num_graphs") else 1
        for sample_index in range(n_graphs):
            if ptr is not None:
                start, end = ptr[sample_index].item(), ptr[sample_index + 1].item()
            else:
                start, end = 0, field_pred_unnormalized.shape[0]
            sample_pred = field_pred_unnormalized[start:end]
            sample_target = field_target_unnormalized[start:end]
            sample_valid = ~torch.isnan(sample_target)
            if not sample_valid.any():
                continue
            diff = sample_pred[sample_valid] - sample_target[sample_valid]
            filename = filenames[sample_index] if filenames is not None else f"sample_{len(collected['field_sample_metrics'])}"
            collected["field_sample_metrics"].append(
                {
                    "filename": filename,
                    "n_nodes": sample_valid.sum().item(),
                    "MSE": torch.mean(diff**2).item(),
                    "MAE": torch.mean(torch.abs(diff)).item(),
                    "Rel_L2_pct": (
                        torch.linalg.norm(diff.float()) / (torch.linalg.norm(sample_target[sample_valid].float()) + EPSILON)
                    ).item()
                    * 100.0,
                }
            )

    def _finalize(self, trainer, collected, inference_times, data_loading_times, total_samples: int) -> None:
        full_preds, full_targets = self._write_head_predictions(trainer, collected)
        if not collected["multi_head"]:
            full_preds = torch.cat(collected["preds"], dim=0)
            full_targets = torch.cat(collected["targets"], dim=0)
            diff = full_preds - full_targets
            trainer.tracker.log_metrics(
                {"test/mse": torch.mean(diff**2).item(), "test/mre_percent": self._mre(diff, full_targets)}
            )
        if collected["multi_head"] and collected["field_preds"] and full_preds.numel() == 1 and full_preds.item() == 0.0:
            full_preds = torch.cat(collected["field_preds"], dim=0)
            full_targets = torch.cat(collected["field_targets"], dim=0)
        self._calculate_metrics(trainer, collected, full_preds, full_targets)
        if trainer.dataset_config["metric_suite"] == "general":
            writer = PredictionWriter(self.config)
            results_array = torch.cat([full_targets.view(-1, 1), full_preds.view(-1, 1)], dim=1).cpu().numpy()
            if collected["field_sample_metrics"] and full_preds.shape[0] > total_samples:
                writer.write_field_metrics(collected["field_sample_metrics"])
            else:
                writer.write_predictions(results_array, "True_Drag,Predicted_Drag")
        self._log_speed(trainer, inference_times, data_loading_times, total_samples)
        self._log_field_metrics(trainer, collected)

    def _write_head_predictions(self, trainer, collected):
        full_preds = torch.tensor([0.0])
        full_targets = torch.tensor([0.0])
        if not collected["multi_head"]:
            return full_preds, full_targets
        for head_name in collected["preds"]:
            if not collected["preds"][head_name]:
                continue
            head_preds = torch.cat(collected["preds"][head_name], dim=0)
            head_targets = torch.cat(collected["targets"][head_name], dim=0)
            diff = head_preds - head_targets
            trainer.tracker.log_metrics(
                {
                    f"test/{head_name}_mse": torch.mean(diff**2).item(),
                    f"test/{head_name}_mae": torch.mean(torch.abs(diff)).item(),
                    f"test/{head_name}_mre_percent": self._mre(diff, head_targets),
                }
            )
            results_array = torch.cat([head_targets.view(-1, 1), head_preds.view(-1, 1)], dim=1).cpu().numpy()
            PredictionWriter(self.config).write_predictions(
                results_array,
                f"True_{head_name},Predicted_{head_name}",
                trainer.path_config["result_path"].replace(".png", f"_{head_name}_predictions.csv"),
            )
            if full_preds.numel() == 1 and full_preds.item() == 0.0:
                full_preds, full_targets = head_preds, head_targets
        return full_preds, full_targets

    def _log_speed(self, trainer, inference_times, data_loading_times, total_samples: int) -> None:
        if not inference_times:
            return
        total_inference_time = sum(inference_times)
        total_data_loading_time = sum(data_loading_times)
        total_test_time = total_inference_time + total_data_loading_time
        self.config["datarow"]["inference time"] = total_inference_time
        trainer.tracker.log_metrics(
            {
                "perf/inference_throughput_sps": total_samples / total_inference_time,
                "perf/overall_throughput_sps": total_samples / total_test_time,
                "perf/avg_latency_ms": statistics.mean(inference_times) * 1000,
                "perf/data_loading_ratio_percent": (total_data_loading_time / total_test_time) * 100,
            }
        )

    def _log_field_metrics(self, trainer, collected) -> None:
        if not collected["field_preds"]:
            return
        field_preds = torch.cat(collected["field_preds"], dim=0)
        field_targets = torch.cat(collected["field_targets"], dim=0)
        diff = field_preds - field_targets
        trainer.tracker.log_metrics(
            {
                "test/field_mse": torch.mean(diff**2).item(),
                "test/field_mae": torch.mean(torch.abs(diff)).item(),
                "test/field_rel_l2_percent": (
                    torch.linalg.norm(diff.float()) / (torch.linalg.norm(field_targets.float()) + EPSILON)
                ).item()
                * 100.0,
            }
        )

    def _mre(self, diff, targets) -> float:
        return (torch.abs(diff) / (torch.abs(targets) + EPSILON)).mean().item() * 100.0

    def _calculate_metrics(self, trainer, collected, full_preds, full_targets) -> None:
        if trainer.dataset_config["metric_suite"] == "drivaernet":
            target_unnormalized = collected["targets"]
            pred_unnormalized = collected["preds"]
        else:
            target_unnormalized = full_targets
            pred_unnormalized = full_preds
        MetricCalculator(self.config).compute_metrics(
            trainer.dataset_config["metric_suite"],
            target_unnormalized,
            pred_unnormalized,
            metadata=trainer.metadata,
            active_variables=trainer.dataset_config["active_variables"],
        )
