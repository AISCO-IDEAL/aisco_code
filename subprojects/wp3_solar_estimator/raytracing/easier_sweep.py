from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from matplotlib import pyplot as plt

import wandb
from subprojects.wp3_solar_estimator.v7.solar_loss_v7 import SolarLossEstimator
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import (
    allround_save_data,
    fetch_pretrained_model,
    reformat_wandb_name,
)


def _extract_histogram_bins_and_counts(hist_obj: Any) -> tuple[list[float], list[float]] | None:
    """Best-effort extraction of histogram bin edges and counts from wandb payloads."""
    candidates: list[dict[str, Any]] = []

    if isinstance(hist_obj, dict):
        candidates.append(hist_obj)

    if hasattr(hist_obj, "to_json"):
        try:
            json_payload = hist_obj.to_json(wandb.run)
            if isinstance(json_payload, dict):
                candidates.append(json_payload)
        except Exception:
            pass

        try:
            json_payload = hist_obj.to_json()
            if isinstance(json_payload, dict):
                candidates.append(json_payload)
        except Exception:
            pass

    attr_bins = getattr(hist_obj, "bins", None)
    attr_counts = getattr(hist_obj, "histogram", None)
    if attr_bins is not None and attr_counts is not None:
        candidates.append({"bins": attr_bins, "values": attr_counts})

    for payload in candidates:
        bins = payload.get("bins") or payload.get("bin_edges")
        counts = payload.get("values") or payload.get("histogram")
        if bins is None or counts is None:
            continue

        bins_list = list(bins)
        counts_list = list(counts)
        if len(bins_list) == len(counts_list) + 1 and len(counts_list) > 0:
            return [float(x) for x in bins_list], [float(x) for x in counts_list]

    return None


def _install_histogram_capture(histogram_key: str) -> tuple[Callable[[], None], dict[str, Any]]:
    """Patch wandb.log to capture histogram payloads for end-of-run scoring."""
    state: dict[str, Any] = {"records": []}
    original_log = wandb.log

    def _wrapped_log(data: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(data, dict) and histogram_key in data:
            extracted = _extract_histogram_bins_and_counts(data[histogram_key])
            if extracted is not None:
                bins, counts = extracted
                state["records"].append({"bins": bins, "counts": counts})
        return original_log(data, *args, **kwargs)

    wandb.log = _wrapped_log

    def _restore() -> None:
        wandb.log = original_log

    return _restore, state


def _compute_halfpipe_metrics_from_histogram(bins: list[float], counts: list[float]) -> dict[str, float]:
    """Compute JSD-to-halfpipe and a normalized similarity score in [0, 1]."""
    bin_edges = torch.tensor(bins, dtype=torch.float64)
    observed = torch.tensor(counts, dtype=torch.float64).clamp_min(0.0)

    if observed.numel() == 0:
        raise ValueError("Histogram has no bins.")
    if bin_edges.numel() != observed.numel() + 1:
        raise ValueError("Histogram bin edges length must be counts length + 1.")
    if float(observed.sum().item()) <= 0.0:
        raise ValueError("Histogram counts sum to zero.")

    p = observed / observed.sum()
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Unnormalized Beta(0.5, 0.5) density over [0, 1]: proportional to 1 / sqrt(x * (1 - x)).
    eps = 1e-12
    clamped_centers = centers.clamp(min=eps, max=1.0 - eps)
    target = 1.0 / torch.sqrt(clamped_centers * (1.0 - clamped_centers))
    q = target / target.sum()

    mix = 0.5 * (p + q)
    jsd = 0.5 * torch.sum(p * torch.log((p + eps) / (mix + eps))) + 0.5 * torch.sum(q * torch.log((q + eps) / (mix + eps)))
    jsd_value = float(jsd.item())
    jsd_max = math.log(2.0)
    score = max(0.0, min(1.0, 1.0 - (jsd_value / jsd_max)))

    return {
        "one_shot/visibility_halfpipe_jsd": jsd_value,
        "one_shot/visibility_halfpipe_score": score,
    }


def _format_bar_label(value: float) -> str:
    """Return compact labels for histogram bars."""
    rounded = round(value)
    if abs(value - rounded) < 1e-9:
        return str(int(rounded))
    return f"{value:.3g}"


def _save_visibility_histogram_plot(bins: list[float], counts: list[float], output_path: Path) -> None:
    """Render and save a histogram plot with numeric labels above each bar."""
    if len(bins) != len(counts) + 1 or len(counts) == 0:
        raise ValueError("Invalid histogram shape for plotting.")

    left_edges = bins[:-1]
    widths = [b1 - b0 for b0, b1 in itertools.pairwise(bins)]
    centers = [left + 0.5 * width for left, width in zip(left_edges, widths, strict=False)]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    bars = ax.bar(left_edges, counts, width=widths, align="edge", edgecolor="black", color="#4F81BD", alpha=0.85)
    ax.set_title("Visibility Histogram (Final)")
    ax.set_xlabel("Visibility")
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_xlim(min(bins), max(bins))

    max_count = max(counts)
    y_offset = 0.01 * max(1.0, max_count)
    for bar, center, count in zip(bars, centers, counts, strict=False):
        ax.text(center, bar.get_height() + y_offset, _format_bar_label(float(count)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving visibility histogram plot to {output_path}...")
    fig.savefig(output_path)
    plt.close(fig)


def _resolve_bulk_run_dir(model: BaseModel, general_config: dict[str, Any]) -> Path:
    """Match bulk-storage run folder structure used by allround_save_data."""
    model_hash = dict_hashing(model.get_model_info_dict())

    run_name = None
    if wandb.run is not None and wandb.run.name:
        run_name = reformat_wandb_name(wandb.run.name)

    if run_name is None:
        run_name = model_hash

    configured_storage = general_config["general_info"].get(
        "storage_path", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
    )
    return Path(configured_storage) / model_hash / run_name


def _init_wandb(general_config: dict[str, Any]) -> None:
    """Initialize wandb if enabled in config."""
    use_wandb = bool(general_config.get("wandb_general", {}).get("use_wandb", False))
    if use_wandb:
        wandb.init(
            entity=general_config["wandb_general"]["entity_name"],
            project=general_config["wandb_general"]["project_name"],
            config=general_config,
        )


def _register_and_apply_sweep_overrides(general_config: dict[str, Any]) -> None:
    """Register all configs, then apply wandb sweep overrides when present."""
    register_all_configs(general_config, v="v1")
    if wandb.run is not None:
        apply_sweep_overrides(dict(wandb.config))
        wandb.config.update(get_master_config(), allow_val_change=True)


def _instantiate_model(general_config: dict[str, Any]) -> BaseModel:
    """Instantiate the configured model class with sweep-aware config."""
    model_cfg_name = general_config["active_configs"]["model_params"]
    model_config = load_config(v="v1", cfg_name=model_cfg_name)

    model_name = general_config["model_def"]["model_name"]
    model_definition = general_config["model_def"]["model_file_path"]
    model_class = dynamic_type_loading(file_path=model_definition, class_name=model_name)

    model = cast(BaseModel, model_class(**model_config))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device)


def _load_pretrained_or_raise(model: BaseModel, general_config: dict[str, Any]) -> None:
    """Load matching pretrained weights into model, fail fast if unavailable."""
    pre_cfg_name = general_config["active_configs"].get("pretraining", "pretraining_v0_config.yaml")
    pre_cfg = load_config(v="v1", cfg_name=pre_cfg_name)

    run_locally = general_config["general_info"].get("run_locally", "False") == "True"
    if run_locally:
        pretrained_path = pre_cfg["pretraining"].get("local_pretraining_model_path", None)
    else:
        pretrained_path = pre_cfg["pretraining"].get("pretraining_model_path", None)

    pretrained_model = fetch_pretrained_model(model, pretrained_path=pretrained_path, verbose=True)
    if pretrained_model is None:
        raise FileNotFoundError(
            "No pretrained model found for this architecture. "
            "This one-shot sweep runner requires an existing pretrained checkpoint."
        )

    device = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(pretrained_model, map_location=device))
    print(f"Loaded pretrained model from {pretrained_model}")


def _run_one_shot_solar(model: BaseModel, general_config: dict[str, Any]) -> tuple[torch.Tensor, SolarLossEstimator]:
    """Run exactly one solar loss evaluation using SolarLossConfigV7."""
    solar_cfg_path = general_config["active_configs"]["solar"]
    solar_cfg = load_config(v="v1", cfg_name=solar_cfg_path)["SolarLossConfigV7"]
    solar_estimator = SolarLossEstimator(config=solar_cfg)

    model.eval()
    loss = solar_estimator.solar_loss(model=model, epoch_name=0)
    return loss, solar_estimator


def _finalize_rendering_if_enabled(solar_estimator: SolarLossEstimator) -> None:
    """Finalize rendering outputs if the estimator created a visualization manager."""
    if not getattr(solar_estimator, "rendering", False):
        return

    manager = getattr(solar_estimator, "visualisation_manager", None)
    if manager is not None:
        try:
            manager.finalize(custom_name="one_shot")
        except Exception as exc:
            print(f"Warning: rendering finalize failed: {exc}")


def _save_artifacts(model: BaseModel, general_config: dict[str, Any]) -> None:
    """Save model and mesh using the same helper used by the full pipeline."""
    training_cfg_name = general_config["active_configs"].get("training", "training_v0_config.yaml")
    training_config = load_config(v="v1", cfg_name=training_cfg_name)
    epoch = int(training_config["training_params"].get("num_epochs", 1000))

    wandb_name = wandb.run.name if wandb.run is not None else None
    allround_save_data(
        model=model,
        storage_path=general_config["general_info"].get(
            "storage_path", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
        ),
        save_model=True,
        save_mesh=True,
        verbose=bool(general_config["general_info"].get("verbose", True)),
        pretraining=False,
        wandb_run=wandb_name,
        epoch=epoch,
    )


def main() -> None:
    histogram_key = "solar_loss_metrics/visibility_histogram"
    restore_wandb_log: Callable[[], None] | None = None
    histogram_capture_state: dict[str, Any] | None = None

    general_config = load_config(v="v1", cfg_name="general_config.yaml")

    _init_wandb(general_config)
    if wandb.run is not None:
        restore_wandb_log, histogram_capture_state = _install_histogram_capture(histogram_key)

    _register_and_apply_sweep_overrides(general_config)

    # Reload after overrides so we use the patched config for everything below.
    general_config = load_config(v="v1", cfg_name="general_config.yaml")

    model = _instantiate_model(general_config)
    _load_pretrained_or_raise(model, general_config)

    loss, solar_estimator = _run_one_shot_solar(model, general_config)
    print(f"One-shot solar loss: {float(loss.detach().item()):.8f}")

    if wandb.run is not None:
        wandb.log({"one_shot/solar_loss": float(loss.detach().item())}, step=0)

    _finalize_rendering_if_enabled(solar_estimator)
    _save_artifacts(model, general_config)

    if wandb.run is not None:
        metric_payload: dict[str, Any] = {
            "one_shot/visibility_halfpipe_available": 0.0,
            "one_shot/visibility_halfpipe_histograms_captured": 0,
        }

        if histogram_capture_state is not None:
            records = histogram_capture_state.get("records", [])
            metric_payload["one_shot/visibility_halfpipe_histograms_captured"] = len(records)
            if len(records) > 0:
                latest = records[-1]
                try:
                    metric_payload.update(_compute_halfpipe_metrics_from_histogram(bins=latest["bins"], counts=latest["counts"]))
                    metric_payload["one_shot/visibility_halfpipe_available"] = 1.0

                    run_dir = _resolve_bulk_run_dir(model=model, general_config=general_config)
                    hist_plot_path = run_dir / "visibility_histogram_final.png"
                    _save_visibility_histogram_plot(bins=latest["bins"], counts=latest["counts"], output_path=hist_plot_path)
                    metric_payload["one_shot/visibility_histogram_plot_saved"] = 1.0
                    metric_payload["one_shot/visibility_histogram_plot_path"] = str(hist_plot_path)
                except Exception as exc:
                    print(f"Warning: failed to compute halfpipe metric from captured histogram: {exc}")
                    metric_payload["one_shot/visibility_histogram_plot_saved"] = 0.0
            else:
                print("Warning: no visibility histogram was captured via wandb.log; halfpipe metric was not computed.")
                metric_payload["one_shot/visibility_histogram_plot_saved"] = 0.0

        wandb.log(metric_payload, step=0)

        if restore_wandb_log is not None:
            restore_wandb_log()
        wandb.finish()

    print("Completed one-shot pretrained solar evaluation.")


if __name__ == "__main__":
    # Preserve current working-directory behavior used by existing launch scripts.
    os.chdir(os.getcwd())
    main()
