"""MetricCalculator is called with a config dict and exposes current metric calculation."""

from __future__ import annotations

import torch

from ..utils.metric import compute_drivaernet_metric

EPSILON = 1e-10


class MetricCalculator:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def compute_metrics(
        self,
        metric_suite: str,
        target_unnormalized,
        pred_unnormalized,
        metadata=None,
        active_variables=None,
    ) -> dict[str, float] | None:
        if metric_suite == "poseidon":
            self.config["datarow"]["relative error (direct)"] = float("nan")
            return {"relative error (direct)": float("nan")}
        if metric_suite == "drivaernet":
            return self._compute_drivaernet_metric(target_unnormalized, pred_unnormalized, metadata, active_variables)
        if metric_suite == "general":
            return self._compute_general_metrics(pred_unnormalized, target_unnormalized)
        raise ValueError(f"Unknown metric suite: {metric_suite}")

    def _compute_drivaernet_metric(
        self, target_unnormalized, pred_unnormalized, metadata, active_variables
    ) -> dict[str, float] | None:
        if metadata is None or not isinstance(target_unnormalized, list) or not isinstance(pred_unnormalized, list):
            return None
        if active_variables is not None:
            metadata.global_mean = [metadata.global_mean[index] for index in active_variables]
            metadata.global_std = [metadata.global_std[index] for index in active_variables]
        metrics = compute_drivaernet_metric(gtr_ls=target_unnormalized, prd_ls=pred_unnormalized, metadata=metadata)
        self.config["datarow"]["MSE (x10^-2)"] = metrics["MSE"] * 100
        self.config["datarow"]["MAE (x10^-1)"] = metrics["MAE"] * 10
        self.config["datarow"]["RMSE"] = metrics["RMSE"]
        self.config["datarow"]["Max_Error"] = metrics["Max_Error"]
        self.config["datarow"]["Rel L2 Error (%)"] = metrics["Rel_L2"] * 100
        self.config["datarow"]["Rel L1 Error (%)"] = metrics["Rel_L1"] * 100
        return metrics

    def _compute_general_metrics(self, pred_unnormalized, target_unnormalized) -> dict[str, float]:
        diff_unnormalized = pred_unnormalized - target_unnormalized
        metrics = {
            "MSE": torch.mean(diff_unnormalized**2).item(),
            "MAE": torch.mean(torch.abs(diff_unnormalized)).item(),
            "Max AE": torch.max(torch.abs(diff_unnormalized)).item(),
        }
        norm_diff_l2 = torch.linalg.norm(diff_unnormalized.float())
        norm_target_l2 = torch.linalg.norm(target_unnormalized.float())
        norm_diff_l1 = torch.linalg.norm(diff_unnormalized.float(), ord=1)
        norm_target_l1 = torch.linalg.norm(target_unnormalized.float(), ord=1)
        metrics["Rel L2 Error (%)"] = (norm_diff_l2 / (norm_target_l2 + EPSILON)).item() * 100.0
        metrics["Rel L1 Error (%)"] = (norm_diff_l1 / (norm_target_l1 + EPSILON)).item() * 100.0
        self.config["datarow"]["MSE (x10^-2)"] = metrics["MSE"] * 100
        self.config["datarow"]["MAE (x10^-1)"] = metrics["MAE"] * 10
        self.config["datarow"]["Max AE"] = metrics["Max AE"]
        self.config["datarow"]["Rel L2 Error (%)"] = metrics["Rel L2 Error (%)"]
        self.config["datarow"]["Rel L1 Error (%)"] = metrics["Rel L1 Error (%)"]
        return metrics
