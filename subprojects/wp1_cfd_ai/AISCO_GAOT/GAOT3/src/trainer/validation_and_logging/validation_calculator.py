"""ValidationCalculator is called with config and trainer context to evaluate validation batches."""

from __future__ import annotations

import contextlib
import logging
import os
import random
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.model.gaot_3d import masked_multi_head_loss
from src.utils.pressure_integration import (
    compute_pointcloud_vertex_areas,
    compute_vertex_areas,
    cp_field_to_drag,
    cp_field_to_drag_pointcloud,
)

from ..utils.plot import plot_3d_comparison_matplotlib


class ValidationCalculator:
    def __init__(self, config: dict) -> None:
        self.config: dict = config

    def validate(self, trainer, loader, epoch_name: int):
        trainer.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_rel_l2 = 0.0
        integrated_errors = []
        signed_x_drag_weighted_errors = []
        signed_x_drag_area_weighted_errors = []
        head_losses: dict[str, float] = {}
        num_batches = len(loader)
        mse_fn = torch.nn.MSELoss()
        with torch.no_grad():
            for batch_raw in loader:
                batch = batch_raw.to(trainer.device)
                pred_normalized = trainer.model(batch=batch, tokens_pos=trainer.latent_tokens.to(trainer.device))
                loss, mse, pred_selected_normalized, target_selected_normalized = self._batch_losses(
                    trainer, batch, pred_normalized, mse_fn, head_losses
                )
                integrated_errors.extend(self._integrated_drag_errors(trainer, batch, pred_normalized))
                signed_x_drag_weighted_errors.extend(
                    self._signed_x_drag_weighted_field_errors(trainer, batch, pred_normalized)
                )
                signed_x_drag_area_weighted_errors.extend(
                    self._signed_x_drag_area_weighted_field_errors(trainer, batch, pred_normalized)
                )
                total_loss += loss.item()
                total_mse += mse.item()
                valid_mask = ~torch.isnan(target_selected_normalized)
                if valid_mask.any():
                    pred_valid_normalized = pred_selected_normalized[valid_mask]
                    target_valid_normalized = target_selected_normalized[valid_mask]
                    total_rel_l2 += (
                        torch.linalg.norm(pred_valid_normalized - target_valid_normalized)
                        / (torch.linalg.norm(target_valid_normalized) + 1e-10)
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
        if trainer.setup_config["rank"] == 0 and signed_x_drag_weighted_errors:
            result["signed_x_drag_weighted_field_error"] = float(np.mean(signed_x_drag_weighted_errors))
            trainer.tracker.log_metrics(
                {"val/signed_x_drag_weighted_field_error": result["signed_x_drag_weighted_field_error"]},
                step=epoch_name,
            )
        if trainer.setup_config["rank"] == 0 and signed_x_drag_area_weighted_errors:
            result["signed_x_drag_area_weighted_field_error"] = float(np.mean(signed_x_drag_area_weighted_errors))
            trainer.tracker.log_metrics(
                {"val/signed_x_drag_area_weighted_field_error": result["signed_x_drag_area_weighted_field_error"]},
                step=epoch_name,
            )
        return result

    def visualize_samples(self, trainer, epoch: int) -> None:
        if trainer.setup_config["rank"] != 0:
            return
        num_viz = trainer.dataset_config.get("num_viz_samples", 1)
        if num_viz <= 0:
            return
        if not hasattr(trainer, "_viz_train_samples"):
            per_dataset = trainer.dataset_config.get("viz_per_dataset", True)
            if per_dataset:
                (
                    trainer._viz_train_samples,
                    trainer._viz_train_suffixes,
                    trainer._viz_val_samples,
                    trainer._viz_val_suffixes,
                ) = self._pick_prefix_viz_samples(trainer, num_viz)
            else:
                train_s, val_s = self._pick_viz_samples(trainer)
                trainer._viz_train_samples = train_s
                trainer._viz_train_suffixes = [f"_{i + 1}" for i in range(len(train_s))]
                trainer._viz_val_samples = val_s
                trainer._viz_val_suffixes = [f"_{i + 1}" for i in range(len(val_s))]

        trainer.model.eval()
        splits = [
            (trainer._viz_train_samples, trainer._viz_train_suffixes, "train"),
            (trainer._viz_val_samples, trainer._viz_val_suffixes, "val"),
        ]
        for samples, suffixes, label in splits:
            for sample, key_suffix in zip(samples, suffixes, strict=False):
                try:
                    self._run_viz_for_sample(trainer, sample, label, epoch, key_suffix=key_suffix)
                except Exception:
                    logging.getLogger(__name__).exception(f"Viz failed for {label}{key_suffix} (epoch {epoch + 1}):")

    def _pick_prefix_viz_samples(self, trainer, n: int):
        """Select N samples per distinct dataset prefix for train and val splits.

        Filenames are expected to follow ``sample_YXXXXX.pt`` where ``Y`` is the
        dataset prefix and ``XXXXX`` is a per-dataset case id.

        Returns (train_samples, train_suffixes, val_samples, val_suffixes) where
        suffixes are W&B key suffixes like ``_1`` (n=1) or ``_1_1``, ``_1_2`` (n>1).
        """
        train_samples, train_meta = self._pick_distinct_prefix_samples(trainer.train_loader.dataset, n)
        val_samples, val_meta = self._pick_distinct_prefix_samples(trainer.val_loader.dataset, n)

        log = logging.getLogger(__name__)
        if train_meta:
            train_desc = ", ".join(f"{p}[{i}]:{idx}({name})" for p, i, idx, name in train_meta)
            log.info(f"Rank {trainer.setup_config['rank']}: Viz train prefixes — [{train_desc}]")
        if val_meta:
            val_desc = ", ".join(f"{p}[{i}]:{idx}({name})" for p, i, idx, name in val_meta)
            log.info(f"Rank {trainer.setup_config['rank']}: Viz val prefixes — [{val_desc}]")

        def _suffix(prefix: str, i: int) -> str:
            return f"_{prefix}" if n == 1 else f"_{prefix}_{i}"

        if train_samples or val_samples:
            train_suffixes = [_suffix(p, i) for p, i, _, _ in train_meta]
            val_suffixes = [_suffix(p, i) for p, i, _, _ in val_meta]
            return train_samples, train_suffixes, val_samples, val_suffixes

        # Fallback for datasets without parseable sample names.
        log.warning("Could not infer dataset prefixes from filenames; falling back to random viz samples.")
        train_s, val_s = self._pick_viz_samples(trainer)
        return train_s, [f"_{i + 1}" for i in range(len(train_s))], val_s, [f"_{i + 1}" for i in range(len(val_s))]

    def _pick_distinct_prefix_samples(self, dataset, n: int):
        """Return up to N samples per distinct filename prefix in deterministic prefix order.

        Returns (samples, meta) where meta is a list of (prefix, sample_index_within_prefix, dataset_idx, filename).
        """
        prefix_to_rows: dict[str, list[tuple[int, str]]] = {}

        split_filenames = getattr(dataset, "split_filenames", None)
        if split_filenames is not None:
            iterable: list[tuple[int, str]] = list(enumerate(split_filenames))
        else:
            iterable = [(idx, getattr(dataset[idx], "filename", "")) for idx in range(len(dataset))]

        for idx, filename in iterable:
            prefix = self._dataset_prefix_from_filename(filename)
            if prefix is None:
                continue
            prefix_to_rows.setdefault(prefix, []).append((idx, str(filename)))

        samples = []
        meta: list[tuple[str, int, int, str]] = []  # (prefix, sample_i, dataset_idx, filename)
        for prefix, rows in sorted(prefix_to_rows.items(), key=lambda item: int(item[0])):
            for sample_i, (idx, name) in enumerate(rows[:n], start=1):
                samples.append(dataset[idx])
                meta.append((prefix, sample_i, idx, name))
        return samples, meta

    def _dataset_prefix_from_filename(self, filename: str | os.PathLike[str] | None) -> str | None:
        if filename is None:
            return None
        name = os.path.basename(str(filename))
        name = name.removesuffix(".pt")
        match = re.fullmatch(r"sample_(\d+)(\d{5})", name)
        if not match:
            return None
        return match.group(1)

    def _pick_viz_samples(self, trainer):
        """Randomly select N samples from train and N from val for per-epoch visualisation."""
        n = trainer.dataset_config.get("num_viz_samples", 1)
        if n <= 0:
            return [], []
        rng = random.Random(int.from_bytes(os.urandom(4), "big"))
        train_size = len(trainer.train_loader.dataset)
        val_size = len(trainer.val_loader.dataset)
        train_indices = rng.sample(range(train_size), min(n, train_size))
        val_indices = rng.sample(range(val_size), min(n, val_size))
        train_samples = [trainer.train_loader.dataset[i] for i in train_indices]
        val_samples = [trainer.val_loader.dataset[i] for i in val_indices]
        names_train = ", ".join(
            f"{i}({getattr(sample, 'filename', 'N/A')})" for i, sample in zip(train_indices, train_samples, strict=True)
        )
        names_val = ", ".join(
            f"{i}({getattr(sample, 'filename', 'N/A')})" for i, sample in zip(val_indices, val_samples, strict=True)
        )
        logging.getLogger(__name__).info(
            f"Rank {trainer.setup_config['rank']}: Viz samples — train[{names_train}], val[{names_val}]"
        )
        return train_samples, val_samples

    def _run_viz_for_sample(self, trainer, sample, split_label: str, epoch: int, key_suffix: str = "_1") -> None:  # noqa: PLR0912, PLR0915
        """Run inference on a single sample and log field + scalar + integrated-drag images to W&B."""
        loader = PyGDataLoader([sample], batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(trainer.device)
        with torch.no_grad():
            pred = trainer.model(batch=batch, tokens_pos=trainer.latent_tokens.to(trainer.device))

        coords_tensor = batch.pos_raw if hasattr(batch, "pos_raw") else batch.pos
        coords = coords_tensor.cpu().numpy()
        fname = getattr(sample, "filename", split_label)
        suffix = key_suffix

        if isinstance(pred, dict) and "field" in pred:
            field_pred = pred["field"].cpu().numpy().squeeze()
            field_gt = batch.x.cpu().numpy().squeeze() if batch.x is not None else None

            if field_gt is not None:
                with tempfile.NamedTemporaryFile(suffix=".png", prefix="pressure_field_", delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                try:
                    plot_3d_comparison_matplotlib(
                        coords=coords,
                        u_gtr=field_gt,
                        u_prd=field_pred,
                        save_path=tmp_path,
                        variable_name="Pressure",
                        point_size=2.0,
                        cmap="jet",
                        dpi=150,
                        view_angle=(20, -120),
                        hide_grid=False,
                        equal_aspect=False,
                        show_abs_diff=True,
                        show_input_shape=False,
                        use_ground_truth_range=False,
                        fixed_clim_common=trainer._viz_fixed_clim_common,
                        fixed_clim_diff=trainer._viz_fixed_clim_diff,
                        fixed_axis_limits=trainer._viz_fixed_axis_limits,
                    )
                    trainer.tracker.log_image(
                        f"viz/{split_label}/pressure_field{suffix}",
                        tmp_path,
                        step=epoch + 1,
                        caption=f"Pressure Field [{split_label}] {fname} — Epoch {epoch + 1}",
                    )
                finally:
                    with contextlib.suppress(OSError):
                        os.remove(tmp_path)

        if isinstance(pred, dict) and "field" in pred and hasattr(batch, "c") and batch.c is not None:
            v_inf = trainer.dataset_config.get("v_inf", 25.0)
            rho = trainer.dataset_config.get("rho", 1.225)
            field_pred_t = pred["field"].cpu().float().squeeze()
            pos_cpu = (
                batch.pos_raw.cpu().float()
                if hasattr(batch, "pos_raw") and batch.pos_raw is not None
                else batch.pos.cpu().float()
            )
            normals_cpu = batch.c.cpu().float()

            drag_pred_val = float("nan")
            drag_gt_val = float("nan")
            try:
                drag_pred_val = cp_field_to_drag_pointcloud(field_pred_t, normals_cpu, pos_cpu, v_inf=v_inf, rho=rho).item()
            except Exception:
                logging.getLogger(__name__).debug("Integrated drag (pred) failed", exc_info=True)

            if batch.x is not None:
                field_gt_t = batch.x.cpu().float().squeeze()
                try:
                    drag_gt_val = cp_field_to_drag_pointcloud(field_gt_t, normals_cpu, pos_cpu, v_inf=v_inf, rho=rho).item()
                except Exception:
                    logging.getLogger(__name__).debug("Integrated drag (GT) failed", exc_info=True)

            drag_metrics = {}
            if not np.isnan(drag_pred_val):
                drag_metrics[f"viz/{split_label}/integrated_drag_pred{suffix}"] = drag_pred_val
            if not np.isnan(drag_gt_val):
                drag_metrics[f"viz/{split_label}/integrated_drag_gt{suffix}"] = drag_gt_val
            if drag_metrics:
                trainer.tracker.log_metrics(drag_metrics, step=epoch + 1)

            if not np.isnan(drag_pred_val):
                fig_drag, ax_drag = plt.subplots(figsize=(5, 4))
                labels = []
                values = []
                colors = []
                if not np.isnan(drag_gt_val):
                    labels.append("GT")
                    values.append(drag_gt_val)
                    colors.append("steelblue")
                labels.append("Pred")
                values.append(drag_pred_val)
                colors.append("coral")
                ax_drag.bar(labels, values, color=colors)
                ax_drag.set_ylabel("Drag Force [N]")
                if not np.isnan(drag_gt_val):
                    ax_drag.set_title(f"Integrated Drag  GT={drag_gt_val:.4f} N  Pred={drag_pred_val:.4f} N")
                else:
                    ax_drag.set_title(f"Integrated Drag  Pred={drag_pred_val:.4f} N")
                fig_drag.suptitle(f"Integrated Pressure Drag [{split_label}] {fname} — Epoch {epoch + 1}", fontsize=11)
                plt.tight_layout()
                trainer.tracker.log_image(
                    f"viz/{split_label}/integrated_drag{suffix}",
                    fig_drag,
                    step=epoch + 1,
                    caption=f"{split_label} {fname} epoch {epoch + 1}",
                )
                plt.close(fig_drag)

        if isinstance(pred, dict):
            scalar_parts = []
            for head in ["force", "cd"]:
                if head in pred and hasattr(batch, head):
                    pred_value = pred[head].cpu().item()
                    gt_value = getattr(batch, head)
                    if gt_value is not None and not torch.isnan(gt_value).all():
                        target_value = gt_value.cpu().item()
                        if hasattr(trainer, "per_head_stats") and head in trainer.per_head_stats:
                            head_mean = trainer.per_head_stats[head]["mean"].item()
                            head_std = trainer.per_head_stats[head]["std"].item()
                            pred_value = pred_value * head_std + head_mean
                            target_value = target_value * head_std + head_mean
                        scalar_parts.append((head, target_value, pred_value))

            if scalar_parts:
                fig, axes = plt.subplots(1, len(scalar_parts), figsize=(5 * len(scalar_parts), 4))
                if len(scalar_parts) == 1:
                    axes = [axes]
                for ax, (name, gt_value, pred_value) in zip(axes, scalar_parts, strict=False):
                    ax.bar(["GT", "Pred"], [gt_value, pred_value], color=["steelblue", "coral"])
                    ax.set_title(f"{name}  (GT={gt_value:.4f}, Pred={pred_value:.4f})")
                    ax.set_ylabel(name)
                fig.suptitle(f"Scalars [{split_label}] {fname} — Epoch {epoch + 1}", fontsize=13)
                plt.tight_layout()
                trainer.tracker.log_image(
                    f"viz/{split_label}/scalar_predictions{suffix}",
                    fig,
                    step=epoch + 1,
                    caption=f"{split_label} {fname} epoch {epoch + 1}",
                )
                plt.close(fig)

    def _batch_losses(self, trainer, batch, pred_normalized, mse_fn, head_losses):
        if isinstance(pred_normalized, dict):
            targets = {"force": batch.force, "cd": batch.cd}
            if trainer.active_heads is not None:
                targets = {key: value for key, value in targets.items() if key in trainer.active_heads}
            loss = masked_multi_head_loss(pred_normalized, targets, loss_fn=trainer.loss_fn)
            mse = masked_multi_head_loss(pred_normalized, targets, loss_fn=mse_fn)
            self._add_head_losses(trainer, pred_normalized, targets, head_losses)
            if targets:
                primary_key = trainer.active_heads[0] if trainer.active_heads else next(iter(targets.keys()))
                pred_selected_normalized = pred_normalized[primary_key]
                target_selected_normalized = targets[primary_key].to(trainer.device).view_as(pred_selected_normalized)
            elif "field" in pred_normalized and hasattr(batch, "x") and batch.x is not None:
                pred_selected_normalized = pred_normalized["field"]
                target_selected_normalized = batch.x.to(trainer.device)
            else:
                pred_selected_normalized = next(iter(pred_normalized.values()))
                target_selected_normalized = pred_selected_normalized.detach()
            loss = trainer.loss_handler.add_field_loss(trainer, batch, pred_normalized, loss, head_losses)
            if "field" in pred_normalized and hasattr(batch, "x") and batch.x is not None:
                field_target_normalized = batch.x.to(trainer.device)
                valid = ~torch.isnan(field_target_normalized)
                if valid.any():
                    mse = mse + trainer.dataset_config["field_loss_weight"] * mse_fn(
                        pred_normalized["field"][valid], field_target_normalized[valid]
                    )
            return loss, mse, pred_selected_normalized, target_selected_normalized
        target_normalized = batch.y
        return (
            trainer.loss_fn(pred_normalized, target_normalized),
            mse_fn(pred_normalized, target_normalized),
            pred_normalized,
            target_normalized,
        )

    def _add_head_losses(self, trainer, pred_normalized, targets, head_losses) -> None:
        loss_fn_none = torch.nn.MSELoss(reduction="none")
        for head_name, target_raw in targets.items():
            if head_name not in pred_normalized or target_raw is None:
                continue
            target_normalized = target_raw.to(trainer.device).view_as(pred_normalized[head_name])
            valid = ~torch.isnan(target_normalized)
            if valid.any():
                head_losses[head_name] = (
                    head_losses.get(head_name, 0.0)
                    + loss_fn_none(pred_normalized[head_name][valid], target_normalized[valid]).mean().item()
                )

    def _integrated_drag_errors(self, trainer, batch, pred_normalized) -> list[float]:
        if not isinstance(pred_normalized, dict) or "field" not in pred_normalized or not hasattr(batch, "x") or batch.x is None:
            return []
        if not hasattr(batch, "c") or batch.c is None:
            return []
        pred_unnormalized = self._denormalize_field(trainer, pred_normalized["field"])
        target_unnormalized = self._denormalize_field(trainer, batch.x.to(trainer.device))
        errors = []
        for start, end in self._sample_ranges(batch):
            drag_pred = self._integrate_sample(trainer, batch, pred_unnormalized, start, end)
            drag_target = self._integrate_sample(trainer, batch, target_unnormalized, start, end)
            if drag_pred is not None and drag_target is not None:
                errors.append(abs(drag_pred - drag_target))
        return errors

    def _signed_x_drag_weighted_field_errors(self, trainer, batch, pred_normalized) -> list[float]:
        if not isinstance(pred_normalized, dict) or "field" not in pred_normalized or not hasattr(batch, "x") or batch.x is None:
            return []
        if not hasattr(batch, "c") or batch.c is None:
            return []
        pred_unnormalized = self._denormalize_field(trainer, pred_normalized["field"])
        target_unnormalized = self._denormalize_field(trainer, batch.x.to(trainer.device))
        errors = []
        for start, end in self._sample_ranges(batch):
            pred = pred_unnormalized[start:end].squeeze(-1)
            target = target_unnormalized[start:end].squeeze(-1)
            normal_x = batch.c[start:end, 0].to(device=pred.device, dtype=pred.dtype)
            valid = ~torch.isnan(target)
            if valid.any():
                weighted_error = (pred[valid] - target[valid]) * -normal_x[valid]
                errors.append(weighted_error.mean().item())
        return errors

    def _signed_x_drag_area_weighted_field_errors(self, trainer, batch, pred_normalized) -> list[float]:
        if not isinstance(pred_normalized, dict) or "field" not in pred_normalized or not hasattr(batch, "x") or batch.x is None:
            return []
        if not hasattr(batch, "c") or batch.c is None:
            return []
        pred_unnormalized = self._denormalize_field(trainer, pred_normalized["field"])
        target_unnormalized = self._denormalize_field(trainer, batch.x.to(trainer.device))
        errors = []
        for start, end in self._sample_ranges(batch):
            pred = pred_unnormalized[start:end].squeeze(-1)
            target = target_unnormalized[start:end].squeeze(-1)
            normal_x = batch.c[start:end, 0].to(device=pred.device, dtype=pred.dtype)
            vertex_areas = self._sample_vertex_areas(batch, start, end).to(device=pred.device, dtype=pred.dtype)
            valid = ~torch.isnan(target) & torch.isfinite(vertex_areas)
            if valid.any():
                weighted_error = (pred[valid] - target[valid]) * -normal_x[valid] * vertex_areas[valid]
                errors.append(weighted_error.mean().item())
        return errors

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

    def _sample_vertex_areas(self, batch, start: int, end: int):
        pos = batch.pos_raw[start:end] if hasattr(batch, "pos_raw") else batch.pos[start:end]
        face = self._sample_face(batch, start, end)
        if face is not None:
            return compute_vertex_areas(pos, face)
        return compute_pointcloud_vertex_areas(pos)

    def _sample_ranges(self, batch) -> list[tuple[int, int]]:
        if hasattr(batch, "ptr"):
            ptr = batch.ptr.detach().cpu().tolist()
            return [(int(ptr[index]), int(ptr[index + 1])) for index in range(len(ptr) - 1)]
        return [(0, int(batch.pos.shape[0]))]
