#!/usr/bin/env python3
"""
GAOT Field Inference Script
============================

Loads a trained GAOT checkpoint and runs inference on test samples to produce:

  1. **Drag Force estimation**        — from the model's ``force`` head
  2. **Drag Coefficient estimation**   — from the model's ``cd`` head
  3. **Surface Shear Force estimation**— from ground-truth ``wall_shear`` (data attribute)
  4. **Pressure Field estimation**     — from the model's ``field`` decoder (MAGNODecoder)
  5. **Drag Force via integration**    — integrating the predicted pressure field using
                                         ``pressure_to_drag()`` over the *original* mesh

Usage
-----
  # Minimal — auto-detect config from yaml
  python scripts/inference_field.py --config config/euler_field.yaml \
         --ckpt /path/to/checkpoint.pt

  # Specify output directory
  python scripts/inference_field.py --config config/euler_field.yaml \
         --ckpt /path/to/checkpoint.pt --out results/euler_inference

  # Use specific samples (by index into the test split)
  python scripts/inference_field.py --config config/euler_field.yaml \
         --ckpt /path/to/checkpoint.pt --samples 0 1 2

  # Run on all splits (train/val/test)
  python scripts/inference_field.py --config config/euler_field.yaml \
         --ckpt /path/to/checkpoint.pt --split test
"""

import argparse
import csv
import logging
import os
import sys
import time

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.dataloader.collate_functions import create_collate_function
from src.dataloader.metadata import DATASET_METADATA
from src.dataloader.pyg_transforms import NormalizeFeatures, RescalePosition, RescalePositionNew
from src.model.gaot_3d import GAOT3D
from src.trainer.utils.setup import load_ckpt
from src.utils.pressure_integration import (
    compute_vertex_areas,
    drag_to_cd,
    pressure_to_drag,
)
from src.utils.scale import rescale, rescale_new

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EPSILON = 1e-10

# ── Default flow parameters (DrivAer-scale) ─────────────────────────────
DEFAULT_RHO = 1.225  # air density [kg/m³]
DEFAULT_U_INF = 30.0  # freestream velocity [m/s]
DEFAULT_A_REF = 2.16  # reference area [m²]
DRAG_AXIS = 0  # x-axis


# ─────────────────────────────────────────────────────────────────────────
#  CONFIG LOADING
# ─────────────────────────────────────────────────────────────────────────
def load_yaml_config(path: str):
    """Load a YAML config and return it as a nested namespace (like OmegaConf DictConfig)."""
    cfg = OmegaConf.load(path)
    return cfg


# ─────────────────────────────────────────────────────────────────────────
#  MODEL BUILDING
# ─────────────────────────────────────────────────────────────────────────
def build_model(cfg, num_input_channels: int, num_output_channels: int, device: torch.device):
    """Instantiate GAOT3D from config and move to device."""
    # Convert OmegaConf → plain dict for the new dict-config GAOT3D
    args = OmegaConf.to_container(cfg.model.args, resolve=True)

    # Inject runtime-detected sizes
    args["input_size"] = num_input_channels
    args["output_size"] = num_output_channels

    field_output_size = args.get("field_output_size", 0)

    model = GAOT3D(config=args, field_output_size=field_output_size)
    model.to(device)
    model.eval()
    return model


def load_checkpoint(model, ckpt_path: str):
    """Load checkpoint weights — handles DDP 'module.' prefix automatically."""
    log.info(f"Loading checkpoint: {ckpt_path}")
    load_ckpt(ckpt_path, model=model)
    log.info("Checkpoint loaded successfully.")
    return model


# ─────────────────────────────────────────────────────────────────────────
#  DATASET / TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────
def build_latent_tokens(cfg, metadata):
    """Create latent token grid from config (same logic as StaticTrainer3D)."""
    latent_token_size = list(cfg.model.args.latent_tokens)
    phy_domain = metadata.domain_x
    x_min, y_min, z_min = phy_domain[0]
    x_max, y_max, z_max = phy_domain[1]
    meshgrid = torch.meshgrid(
        torch.linspace(x_min, x_max, latent_token_size[0]),
        torch.linspace(y_min, y_max, latent_token_size[1]),
        torch.linspace(z_min, z_max, latent_token_size[2]),
        indexing="ij",
    )
    latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)

    use_rescale_new_flag = getattr(cfg.dataset, "use_rescale_new", False)
    if use_rescale_new_flag:
        return rescale_new(latent_queries, (-1, 1), phy_domain)
    else:
        return rescale(latent_queries, (-1, 1))


def build_transforms_and_stats(cfg, data_root: str):
    """Load normalization stats and build the same transform pipeline used during training."""
    dataset_name = cfg.dataset.name
    stats_file = os.path.join(data_root, f"{dataset_name}_norm_stats.pt")
    if not os.path.isfile(stats_file):
        raise FileNotFoundError(f"Normalization stats not found: {stats_file}")
    stats = torch.load(stats_file, weights_only=False)
    log.info(f"Loaded normalization stats from {stats_file}")

    u_mean = stats["mean"].float()
    u_std = stats["std"].float()
    c_mean = stats.get("c_mean")
    c_std = stats.get("c_std")
    if c_mean is not None:
        c_mean = c_mean.float()
    if c_std is not None:
        c_std = c_std.float()
    per_head_stats = stats.get("per_head_stats", {})

    use_rescale_new_flag = getattr(cfg.dataset, "use_rescale_new", False)
    metadata = DATASET_METADATA[cfg.dataset.metaname]
    if use_rescale_new_flag:
        phy_domain = metadata.domain_x
        rescale_transform = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=phy_domain)
    else:
        rescale_transform = RescalePosition(lims=(-1.0, 1.0))

    normalize_transform = NormalizeFeatures(
        mean=u_mean,
        std=u_std,
        c_mean=c_mean,
        c_std=c_std,
        per_head_stats=per_head_stats,
    )
    composed = Compose([rescale_transform, normalize_transform])
    return composed, per_head_stats, u_mean, u_std


def build_collate_fn(cfg, latent_tokens):
    """Create the same collate function used during training (builds edges online)."""
    magno = cfg.model.args.magno
    return create_collate_function(
        coord_dim=magno.gno_coord_dim,
        magno_radius=magno.gno_radius,
        magno_scales=list(magno.scales),
        latent_tokens=latent_tokens,
        neighbor_search_method=magno.neighbor_strategy,
        k_neighbors=magno.k_neighbors,
        asynchronous_graph_building=magno.asynchronous_graph_building,
    )


# ─────────────────────────────────────────────────────────────────────────
#  RAW DATA LOADER (no transforms — for pressure integration)
# ─────────────────────────────────────────────────────────────────────────
def load_raw_sample(data_root: str, sample_name: str) -> Data:
    """Load a single .pt sample *without* any transforms (original physical units)."""
    # Try flat layout first, then subdirectory layout
    flat = os.path.join(data_root, f"{sample_name}.pt")
    sub = os.path.join(data_root, sample_name, "data.pt")
    if os.path.isfile(flat):
        path = flat
    elif os.path.isfile(sub):
        path = sub
    else:
        raise FileNotFoundError(f"Cannot find sample '{sample_name}' at {flat} or {sub}")
    return torch.load(path, weights_only=False)


# ─────────────────────────────────────────────────────────────────────────
#  INFERENCE HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────
def _run_forward(raw_data, model, transform, collate_fn, latent_tokens, device):
    """Prepare input and run one forward pass. Returns (pred dict, inference_time_ms)."""
    data_for_model = raw_data.clone()
    data_for_model = transform(data_for_model)
    batch = collate_fn([data_for_model])
    batch = batch.to(device)
    latent_dev = latent_tokens.to(device)
    query_pos = batch.pos.to(device)
    query_batch_idx = batch.batch.to(device)
    t0 = time.perf_counter()
    pred = model(batch=batch, tokens_pos=latent_dev, query_coord_pos=query_pos, query_coord_batch_idx=query_batch_idx)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return pred, (time.perf_counter() - t0) * 1000


def _compute_shear_drag(raw_data, pos_raw, face_raw, drag_axis: int, rho: float, u_inf: float, a_ref: float) -> tuple:
    """Estimate drag from GT wall_shear. Returns (shear_drag, shear_cd)."""
    if not (hasattr(raw_data, "wall_shear") and raw_data.wall_shear is not None):
        return float("nan"), float("nan")
    ws = raw_data.wall_shear.float()
    vertex_areas = compute_vertex_areas(pos_raw, face_raw)
    shear_drag = (ws[:, drag_axis] * vertex_areas).sum().item()
    shear_cd = drag_to_cd(torch.tensor(shear_drag), rho=rho, u_inf=u_inf, A_ref=a_ref).item()
    return shear_drag, shear_cd


def _integrate_pressure_drag(
    pred_pressure, normals_raw, face_raw, pos_raw, n_verts: int, drag_axis: int, rho: float, u_inf: float, a_ref: float
) -> tuple:
    """Integrate predicted pressure field into drag. Returns (int_drag, int_cd)."""
    if pred_pressure is None:
        return float("nan"), float("nan")
    if pred_pressure.shape[0] != n_verts:
        log.warning(f"Vertex count mismatch: pred={pred_pressure.shape[0]} vs raw={n_verts}. Skipping pressure integration.")
        return float("nan"), float("nan")
    int_drag = pressure_to_drag(pred_pressure, normals_raw, face_raw, pos_raw, drag_direction=drag_axis).item()
    int_cd = drag_to_cd(torch.tensor(int_drag), rho=rho, u_inf=u_inf, A_ref=a_ref).item()
    return int_drag, int_cd


def _compute_gt_field_metrics(
    raw_data, pred_pressure, normals_raw, face_raw, pos_raw, n_verts: int, drag_axis: int, rho: float, u_inf: float, a_ref: float
) -> tuple:
    """Compute GT pressure integration and field error metrics. Returns (gt_int_drag, gt_int_cd, field_mse, field_mae)."""
    if not (hasattr(raw_data, "x") and raw_data.x is not None):
        return float("nan"), float("nan"), float("nan"), float("nan")
    gt_pressure = raw_data.x.float().squeeze(-1)
    gt_int_drag = pressure_to_drag(gt_pressure, normals_raw, face_raw, pos_raw, drag_direction=drag_axis).item()
    gt_int_cd = drag_to_cd(torch.tensor(gt_int_drag), rho=rho, u_inf=u_inf, A_ref=a_ref).item()
    field_mse = float("nan")
    field_mae = float("nan")
    if pred_pressure is not None and pred_pressure.shape[0] == n_verts:
        field_mse = F.mse_loss(pred_pressure, gt_pressure).item()
        field_mae = (pred_pressure - gt_pressure).abs().mean().item()
    return gt_int_drag, gt_int_cd, field_mse, field_mae


# ─────────────────────────────────────────────────────────────────────────
#  INFERENCE ON ONE SAMPLE
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_single(
    model,
    raw_data: Data,
    transform,
    collate_fn,
    latent_tokens: torch.Tensor,
    per_head_stats: dict,
    device: torch.device,
    rho: float = DEFAULT_RHO,
    u_inf: float = DEFAULT_U_INF,
    a_ref: float = DEFAULT_A_REF,
    drag_axis: int = DRAG_AXIS,
) -> dict:
    """
    Run full inference on one sample and return a results dict.

    The sample is loaded raw (physical units) for ground-truth comparison
    and pressure integration.  A *transformed* copy is fed to the model.

    Returns dict with keys:
        pred_force, pred_cd,
        pred_pressure  (full mesh, denormalized),
        integrated_drag, integrated_cd,
        gt_force, gt_cd,
        shear_drag, shear_cd,  (from GT wall_shear)
        n_vertices, total_area,
        inference_time_ms
    """
    # ── 1+2. Prepare input and run forward pass ──────────────────────────
    pred, dt = _run_forward(raw_data, model, transform, collate_fn, latent_tokens, device)

    # ── 3. Denormalize scalar predictions ────────────────────────────────
    result = {"inference_time_ms": dt}

    for head_name in ("force", "cd"):
        head_pred = pred.get(head_name)
        if head_pred is None:
            result[f"pred_{head_name}"] = float("nan")
            continue
        if head_name in per_head_stats:
            h_mean = per_head_stats[head_name]["mean"].to(device)
            h_std = per_head_stats[head_name]["std"].to(device)
            head_pred = head_pred * h_std + h_mean
        result[f"pred_{head_name}"] = head_pred.squeeze().cpu().item()

    # ── 4. Denormalize field prediction (pressure) ───────────────────────
    # The field decoder predicts in normalised space. We currently don't
    # normalise x (pressure) via NormalizeFeatures for the field target —
    # the field target is the per-vertex data.x which goes through the
    # same NormalizeFeatures pipeline.  To denormalize we'd need the stats
    # that were used for x.  In neural_field training, data.x is loaded
    # and normalised along with force/cd, but the field target is kept in
    # the same normalised space.  For safety, let's check if x was
    # normalised and reverse it.
    #
    # In the current config, NormalizeFeatures normalises data.y (or
    # force/cd heads).  data.x (the per-vertex pressure) is NOT explicitly
    # normalised by NormalizeFeatures — it's the raw pressure from the .pt
    # file.  So the field prediction should already be in physical units.
    # BUT, if normalisation WAS applied, we'd need to invert it.
    #
    # Since the training loop uses data.x as-is for the field target and
    # the field decoder learns to predict that, the field output is in the
    # same space as data.x.  data.x has NOT been normalised by the
    # NormalizeFeatures transform (it only normalises y, force, cd, and c).
    # So pred["field"] is in physical pressure units.

    field_pred_norm = pred.get("field")
    if field_pred_norm is not None:
        pred_pressure = field_pred_norm.squeeze(-1).cpu()  # [N]
        result["pred_pressure"] = pred_pressure
    else:
        pred_pressure = None
        result["pred_pressure"] = None

    # ── 5. Ground-truth scalars ──────────────────────────────────────────
    gt_force = float("nan")
    gt_cd = float("nan")
    if hasattr(raw_data, "force") and raw_data.force is not None:
        gt_force = raw_data.force.item()
    if hasattr(raw_data, "cd") and raw_data.cd is not None:
        gt_cd = raw_data.cd.item()
    result["gt_force"] = gt_force
    result["gt_cd"] = gt_cd

    # ── 6-8. Shear, pressure integration, and GT field metrics ──────────
    pos_raw = raw_data.pos.float()
    normals_raw = raw_data.c.float()
    face_raw = raw_data.face.long()
    n_verts = pos_raw.shape[0]

    result["shear_drag"], result["shear_cd"] = _compute_shear_drag(raw_data, pos_raw, face_raw, drag_axis, rho, u_inf, a_ref)
    int_drag, int_cd = _integrate_pressure_drag(
        pred_pressure, normals_raw, face_raw, pos_raw, n_verts, drag_axis, rho, u_inf, a_ref
    )
    result["integrated_drag"] = int_drag
    result["integrated_cd"] = int_cd
    result["n_vertices"] = n_verts
    result["total_area"] = compute_vertex_areas(pos_raw, face_raw).sum().item()

    gt_int_drag, gt_int_cd, field_mse, field_mae = _compute_gt_field_metrics(
        raw_data, pred_pressure, normals_raw, face_raw, pos_raw, n_verts, drag_axis, rho, u_inf, a_ref
    )
    result["gt_integrated_drag"] = gt_int_drag
    result["gt_integrated_cd"] = gt_int_cd
    result["field_mse"] = field_mse
    result["field_mae"] = field_mae

    return result


# ─────────────────────────────────────────────────────────────────────────
#  PRETTY-PRINT
# ─────────────────────────────────────────────────────────────────────────
def print_result(name: str, r: dict):
    """Print a nicely formatted summary for one sample."""
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  Sample: {name}   ({r['n_vertices']:,} vertices, area={r['total_area']:.4f} m²)")
    print(f"  Inference time: {r['inference_time_ms']:.1f} ms")
    print(sep)

    # ── Scalar heads ─────────────────────────────────────────────────
    print("  ┌─ Scalar Head Predictions ──────────────────────────────┐")
    print(
        f"  │  Force:  pred={r['pred_force']:>12.4f}   GT={r['gt_force']:>12.4f}    err={r['pred_force'] - r['gt_force']:>+10.4f}"
    )
    print(f"  │  Cd:     pred={r['pred_cd']:>12.6f}   GT={r['gt_cd']:>12.6f}    err={r['pred_cd'] - r['gt_cd']:>+10.6f}")
    print("  └──────────────────────────────────────────────────────────┘")

    # ── Pressure field ───────────────────────────────────────────────
    if r.get("pred_pressure") is not None:
        p = r["pred_pressure"]
        print("  ┌─ Pressure Field ─────────────────────────────────────────┐")
        print(f"  │  Predicted pressure: min={p.min().item():.4f}, max={p.max().item():.4f}, mean={p.mean().item():.4f}")
        if not np.isnan(r["field_mse"]):
            print(f"  │  Field MSE = {r['field_mse']:.6f},  MAE = {r['field_mae']:.6f}")
        print("  └──────────────────────────────────────────────────────────┘")

    # ── Pressure integration ─────────────────────────────────────────
    print("  ┌─ Drag via Pressure Integration ──────────────────────────┐")
    print(f"  │  Predicted:  F_drag = {r['integrated_drag']:>12.4f} N,  Cd = {r['integrated_cd']:>12.6f}")
    if not np.isnan(r["gt_integrated_drag"]):
        print(f"  │  GT (integ): F_drag = {r['gt_integrated_drag']:>12.4f} N,  Cd = {r['gt_integrated_cd']:>12.6f}")
    print("  └──────────────────────────────────────────────────────────┘")

    # ── Shear force ──────────────────────────────────────────────────
    if not np.isnan(r["shear_drag"]):
        print("  ┌─ Surface Shear Stress (from GT wall_shear) ──────────────┐")
        print(f"  │  Shear drag = {r['shear_drag']:>12.4f} N,  Shear Cd = {r['shear_cd']:>12.6f}")
        print("  └──────────────────────────────────────────────────────────┘")
    print()


# ─────────────────────────────────────────────────────────────────────────
#  CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────
CSV_FIELDS = [
    "sample",
    "n_vertices",
    "total_area",
    "inference_time_ms",
    # Scalar heads
    "pred_force",
    "gt_force",
    "force_error",
    "pred_cd",
    "gt_cd",
    "cd_error",
    # Pressure integration
    "integrated_drag",
    "gt_integrated_drag",
    "integ_drag_error",
    "integrated_cd",
    "gt_integrated_cd",
    "integ_cd_error",
    # Shear
    "shear_drag",
    "shear_cd",
    # Field quality
    "field_mse",
    "field_mae",
]


def result_to_row(name: str, r: dict) -> dict:
    row = {"sample": name}
    for k in CSV_FIELDS:
        if k == "sample":
            continue
        if k == "force_error":
            row[k] = r["pred_force"] - r["gt_force"]
        elif k == "cd_error":
            row[k] = r["pred_cd"] - r["gt_cd"]
        elif k == "integ_drag_error":
            row[k] = r["integrated_drag"] - r.get("gt_integrated_drag", float("nan"))
        elif k == "integ_cd_error":
            row[k] = r["integrated_cd"] - r.get("gt_integrated_cd", float("nan"))
        else:
            row[k] = r.get(k, float("nan"))
    return row


# ─────────────────────────────────────────────────────────────────────────
#  FIELD EXPORT (per-vertex .pt files for visualization)
# ─────────────────────────────────────────────────────────────────────────
def save_field_output(out_dir: str, name: str, raw_data: Data, result: dict):
    """Save a .pt file containing pos, normals, GT pressure, predicted pressure, and errors."""
    field_dir = os.path.join(out_dir, "fields")
    os.makedirs(field_dir, exist_ok=True)

    out = {
        "pos": raw_data.pos,
        "normals": raw_data.c,
        "face": raw_data.face,
    }
    if hasattr(raw_data, "x") and raw_data.x is not None:
        out["gt_pressure"] = raw_data.x.squeeze(-1)
    if result.get("pred_pressure") is not None:
        out["pred_pressure"] = result["pred_pressure"]
        if "gt_pressure" in out:
            out["error"] = result["pred_pressure"] - out["gt_pressure"]
    if hasattr(raw_data, "wall_shear") and raw_data.wall_shear is not None:
        out["wall_shear"] = raw_data.wall_shear

    out["scalars"] = {
        "pred_force": result["pred_force"],
        "pred_cd": result["pred_cd"],
        "gt_force": result["gt_force"],
        "gt_cd": result["gt_cd"],
        "integrated_drag": result["integrated_drag"],
        "integrated_cd": result["integrated_cd"],
    }

    path = os.path.join(field_dir, f"{name}_field.pt")
    torch.save(out, path)
    log.info(f"Saved field output: {path}")


# ─────────────────────────────────────────────────────────────────────────
#  VISUALIZATION HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────
def _add_scatter_panel(fig, ax, xy, values, title, xlabel, ylabel, point_size, cmap, norm=None, **kwargs):
    """Draw a scatter panel with colorbar and axis labels."""
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=point_size, cmap=cmap, norm=norm, rasterized=True, **kwargs)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)


def _draw_pressure_panels(fig, axes, xy, raw_data, result, idx, point_size, xlabel, ylabel, has_gt, has_pred, has_shear):
    """Draw GT pressure, predicted pressure, error, and wall-shear panels onto axes."""
    panel = 0
    vmin, vmax, vcenter, gt, pred_p = None, None, None, None, None
    if has_gt:
        gt = raw_data.x.squeeze(-1).cpu().numpy()[idx]
        vmin, vmax = np.percentile(gt, [2, 98])
        vcenter = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2
        try:
            norm = TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
        except ValueError:
            norm = None
        _add_scatter_panel(fig, axes[panel], xy, gt, "GT Pressure [Pa]", xlabel, ylabel, point_size, "RdBu_r", norm=norm)
        panel += 1
    if has_pred:
        pred_p = result["pred_pressure"].cpu().numpy()[idx]
        if has_gt:
            p_vmin, p_vmax = vmin, vmax
            p_vcenter = vcenter
        else:
            p_vmin, p_vmax = np.percentile(pred_p, [2, 98])
            p_vcenter = 0.0 if p_vmin < 0 < p_vmax else (p_vmin + p_vmax) / 2
        assert p_vcenter is not None
        try:
            norm_p = TwoSlopeNorm(vcenter=p_vcenter, vmin=p_vmin, vmax=p_vmax)
        except ValueError:
            norm_p = None
        _add_scatter_panel(
            fig, axes[panel], xy, pred_p, "Predicted Pressure [Pa]", xlabel, ylabel, point_size, "RdBu_r", norm=norm_p
        )
        panel += 1
    if has_gt and has_pred:
        assert gt is not None
        error = pred_p - gt
        err_abs_max = max(abs(np.percentile(error, 2)), abs(np.percentile(error, 98)))
        if err_abs_max < EPSILON:
            err_abs_max = 1.0
        mse_val = result.get("field_mse", float("nan"))
        mae_val = result.get("field_mae", float("nan"))
        err_title = f"Error (pred - GT)\nMSE={mse_val:.2f}, MAE={mae_val:.2f}"
        _add_scatter_panel(
            fig, axes[panel], xy, error, err_title, xlabel, ylabel, point_size, "coolwarm", vmin=-err_abs_max, vmax=err_abs_max
        )
        panel += 1
    if has_shear:
        ws = raw_data.wall_shear.cpu().numpy()[idx]
        ws_mag = np.linalg.norm(ws, axis=1)
        _add_scatter_panel(fig, axes[panel], xy, ws_mag, "Wall Shear |\u03c4| [Pa]", xlabel, ylabel, point_size, "inferno")


# ─────────────────────────────────────────────────────────────────────────
#  VISUALIZATION — 2D projections of pressure fields
# ─────────────────────────────────────────────────────────────────────────
def _project_2d(pos: np.ndarray):
    """Choose the best 2D projection plane (drop the axis with least variance)."""
    var = np.var(pos, axis=0)
    drop_axis = int(np.argmin(var))
    keep = [i for i in range(3) if i != drop_axis]
    labels = ["X", "Y", "Z"]
    return pos[:, keep], labels[keep[0]], labels[keep[1]]


def visualize_field(
    out_dir: str,
    name: str,
    raw_data: Data,
    result: dict,
    max_points: int = 80_000,
):
    """
    Render a multi-panel PNG comparing GT vs predicted pressure and error.

    Layout (3 or 4 panels):
      [GT pressure] [Predicted pressure] [Absolute error] [optional: wall shear mag]

    Uses 2D projection (drops the axis with least variance).
    """
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    pos = raw_data.pos.cpu().numpy()
    n_verts = pos.shape[0]

    # Subsample for plotting speed
    idx = np.random.default_rng(42).choice(n_verts, max_points, replace=False) if n_verts > max_points else np.arange(n_verts)

    pos_sub = pos[idx]
    xy, xlabel, ylabel = _project_2d(pos_sub)

    has_gt = hasattr(raw_data, "x") and raw_data.x is not None
    has_pred = result.get("pred_pressure") is not None
    has_shear = hasattr(raw_data, "wall_shear") and raw_data.wall_shear is not None

    n_panels = 0
    if has_gt:
        n_panels += 1
    if has_pred:
        n_panels += 1
    if has_gt and has_pred:
        n_panels += 1  # error panel
    if has_shear:
        n_panels += 1
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), dpi=120)
    if n_panels == 1:
        axes = [axes]

    point_size = max(0.1, min(2.0, 40_000 / len(idx)))
    _draw_pressure_panels(fig, axes, xy, raw_data, result, idx, point_size, xlabel, ylabel, has_gt, has_pred, has_shear)

    # ── Title & save ──────────────────────────────────────────────────
    suptitle_parts = [f"{name}  ({n_verts:,} vertices)"]
    suptitle_parts.append(
        f"Force: pred={result['pred_force']:.2f}, GT={result['gt_force']:.2f}  |  "
        f"Cd: pred={result['pred_cd']:.4f}, GT={result['gt_cd']:.4f}"
    )
    if not np.isnan(result.get("integrated_drag", float("nan"))):
        suptitle_parts.append(f"Integrated drag: {result['integrated_drag']:.2f} N")
    fig.suptitle("\n".join(suptitle_parts), fontsize=10, y=1.02)

    fig.tight_layout()
    save_path = os.path.join(plot_dir, f"{name}_pressure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved visualization: {save_path}")


# ─────────────────────────────────────────────────────────────────────────
#  MAIN HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────
def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the inference script."""
    parser = argparse.ArgumentParser(description="GAOT Field Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML config (e.g. config/euler_field.yaml). "
        "Defines model architecture and is used for data unless --data-config is set.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=None,
        help="Optional: path to a DIFFERENT dataset config YAML. "
        "Use this for cross-dataset inference — e.g. Euler-trained model on Ahmed data. "
        "Model architecture comes from --config, data/transforms come from --data-config.",
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint .pt file")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default: results/<config_name>_inference/)")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which data split to run inference on (default: test)",
    )
    parser.add_argument(
        "--samples", type=int, nargs="*", default=None, help="Specific sample indices within the split (default: all)"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument(
        "--save-fields", action="store_true", default=True, help="Save per-vertex predicted fields as .pt files (default: True)"
    )
    parser.add_argument("--no-save-fields", dest="save_fields", action="store_false", help="Do not save per-vertex field files")
    parser.add_argument(
        "--rho", type=float, default=DEFAULT_RHO, help=f"Freestream density [kg/m\u00b3] (default: {DEFAULT_RHO})"
    )
    parser.add_argument(
        "--u-inf", type=float, default=DEFAULT_U_INF, help=f"Freestream velocity [m/s] (default: {DEFAULT_U_INF})"
    )
    parser.add_argument("--a-ref", type=float, default=DEFAULT_A_REF, help=f"Reference area [m\u00b2] (default: {DEFAULT_A_REF})")
    return parser


def _build_split_names(dataset_cfg, args, data_root: str) -> list:
    """Read the order file, determine split indices, and return sample names for the given split."""
    order_file_name = getattr(dataset_cfg, "order_file_name", None)
    processed_folder = getattr(dataset_cfg, "processed_folder", "")
    if order_file_name:
        order_file_path = os.path.join(data_root, order_file_name)
    else:
        order_file_path = os.path.join(data_root, f"order_{processed_folder}.txt")
    with open(order_file_path) as f:
        all_sample_names = [line.strip() for line in f if line.strip()]
    log.info(f"Total samples in order file: {len(all_sample_names)}")
    indices = np.arange(len(all_sample_names))
    if getattr(dataset_cfg, "rand_dataset", False):
        rng = np.random.default_rng(seed=42)
        rng.shuffle(indices)
    train_size = dataset_cfg.train_size
    val_size = dataset_cfg.val_size
    test_size = dataset_cfg.test_size
    if args.split == "train":
        split_indices = indices[:train_size]
    elif args.split == "val":
        split_indices = indices[train_size : train_size + val_size]
    else:
        split_indices = indices[-test_size:]
    split_names = [all_sample_names[i] for i in split_indices]
    log.info(f"Split '{args.split}': {len(split_names)} samples")
    if args.samples is not None:
        split_names = [split_names[i] for i in args.samples]
        log.info(f"Running on {len(split_names)} selected samples: {split_names}")
    return split_names


def _detect_input_channels(sample_transformed, cfg) -> int:
    """Detect the number of input channels from a transformed sample and config."""
    encoder_feature_attr = getattr(cfg.model.args.magno, "encoder_feature_attr", "x")
    if hasattr(encoder_feature_attr, "__iter__") and not isinstance(encoder_feature_attr, str):
        encoder_feature_attr = list(encoder_feature_attr)
    if isinstance(encoder_feature_attr, list):
        num_channels = 0
        for attr_name in encoder_feature_attr:
            if hasattr(sample_transformed, attr_name):
                num_channels += getattr(sample_transformed, attr_name).shape[1]
            else:
                num_channels += sample_transformed.pos.shape[1]
        return num_channels
    if isinstance(encoder_feature_attr, str) and hasattr(sample_transformed, encoder_feature_attr):
        return getattr(sample_transformed, encoder_feature_attr).shape[1]
    return sample_transformed.pos.shape[1]


def _print_inference_summary(all_results: list, args) -> None:
    """Print aggregated statistics over all inferred samples."""
    n = len(all_results)
    if n == 0:
        return
    force_errors = [abs(r["pred_force"] - r["gt_force"]) for _, r in all_results if not np.isnan(r["gt_force"])]
    cd_errors = [abs(r["pred_cd"] - r["gt_cd"]) for _, r in all_results if not np.isnan(r["gt_cd"])]
    integ_errors = [
        abs(r["integrated_drag"] - r["gt_integrated_drag"])
        for _, r in all_results
        if not np.isnan(r.get("gt_integrated_drag", float("nan")))
    ]
    field_mses = [r["field_mse"] for _, r in all_results if not np.isnan(r["field_mse"])]
    times = [r["inference_time_ms"] for _, r in all_results]
    print("\n" + "=" * 72)
    print(f"  SUMMARY  ({n} samples, split={args.split})")
    if args.data_config is not None:
        print(
            f"  CROSS-DATASET: model trained on {os.path.basename(args.config)}, data from {os.path.basename(args.data_config)}"
        )
        print("  NOTE: Scalar heads (force/cd) are in training-dataset units.")
        print("        Field comparison (MSE/MAE) is meaningful; scalar comparison is NOT.")
    print("=" * 72)
    if force_errors:
        print(f"  Force MAE:           {np.mean(force_errors):.4f}")
    if cd_errors:
        print(f"  Cd MAE:              {np.mean(cd_errors):.6f}")
    if integ_errors:
        print(f"  Integrated drag MAE: {np.mean(integ_errors):.4f}")
    if field_mses:
        print(f"  Field MSE (avg):     {np.mean(field_mses):.6f}")
    print(f"  Avg inference time:  {np.mean(times):.1f} ms")
    print("=" * 72 + "\n")


def _run_inference_loop(
    split_names, model, composed_transform, collate_fn, latent_tokens, per_head_stats, device, args, data_root
):
    """Run inference on all samples. Returns (all_results, all_rows)."""
    all_results = []
    all_rows = []
    log.info(f"\n{'=' * 72}")
    log.info(f"  Running inference on {len(split_names)} samples ...")
    log.info(f"{'=' * 72}\n")
    for i, sample_name in enumerate(split_names):
        log.info(f"[{i + 1}/{len(split_names)}] Processing {sample_name} ...")
        raw_data = load_raw_sample(data_root, sample_name)
        result = infer_single(
            model=model,
            raw_data=raw_data,
            transform=composed_transform,
            collate_fn=collate_fn,
            latent_tokens=latent_tokens,
            per_head_stats=per_head_stats,
            device=device,
            rho=args.rho,
            u_inf=args.u_inf,
            a_ref=args.a_ref,
            drag_axis=DRAG_AXIS,
        )
        all_results.append((sample_name, result))
        all_rows.append(result_to_row(sample_name, result))
        print_result(sample_name, result)
        if args.save_fields and result.get("pred_pressure") is not None:
            save_field_output(args.out, sample_name, raw_data, result)
            try:
                visualize_field(args.out, sample_name, raw_data, result)
            except Exception as e:
                log.warning(f"Visualization failed for {sample_name}: {e}")
    return all_results, all_rows


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────
def _setup_run(args):
    """Resolve config, dataset, output dir, device, transforms, tokens, collate, split names.

    Returns (cfg, dataset_cfg, data_root, device, composed_transform, per_head_stats,
             latent_tokens, collate_fn, split_names).
    """
    cfg = load_yaml_config(args.config)
    if args.data_config is not None:
        data_cfg_full = load_yaml_config(args.data_config)
        dataset_cfg = data_cfg_full.dataset
        log.info(f"CROSS-DATASET MODE: model from {args.config}, data from {args.data_config}")
    else:
        dataset_cfg = cfg.dataset
    data_root = dataset_cfg.base_path
    metadata = DATASET_METADATA[dataset_cfg.metaname]
    # Output directory
    if args.out is None:
        config_stem = os.path.splitext(os.path.basename(args.config))[0]
        if args.data_config is not None:
            data_stem = os.path.splitext(os.path.basename(args.data_config))[0]
            args.out = os.path.join("results", f"{config_stem}_on_{data_stem}")
        else:
            args.out = os.path.join("results", f"{config_stem}_inference")
    os.makedirs(args.out, exist_ok=True)
    log.info(f"Output directory: {args.out}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    # Transforms
    composed_transform, per_head_stats, _u_mean, _u_std = build_transforms_and_stats(cfg, cfg.dataset.base_path)
    log.info(f"Per-head stats keys (from training): {list(per_head_stats.keys())}")
    if args.data_config is not None:
        use_rescale_new_flag = getattr(dataset_cfg, "use_rescale_new", False)
        if use_rescale_new_flag:
            composed_transform = Compose([RescalePositionNew(lims=(-1.0, 1.0), phy_domain=metadata.domain_x)])
        else:
            composed_transform = Compose([RescalePosition(lims=(-1.0, 1.0))])
    # Latent tokens & collate
    train_metadata = DATASET_METADATA[cfg.dataset.metaname]
    latent_tokens = build_latent_tokens(cfg, train_metadata)
    log.info(f"Latent tokens shape: {latent_tokens.shape}")
    collate_fn = build_collate_fn(cfg, latent_tokens)
    split_names = _build_split_names(dataset_cfg, args, data_root)
    return cfg, dataset_cfg, data_root, device, composed_transform, per_head_stats, latent_tokens, collate_fn, split_names


def main():
    args = _build_arg_parser().parse_args()
    cfg, _dataset_cfg, data_root, device, composed_transform, per_head_stats, latent_tokens, collate_fn, split_names = _setup_run(
        args
    )

    # ── Detect input channels & build model ──────────────────────────
    sample0_transformed = composed_transform(load_raw_sample(data_root, split_names[0]).clone())
    num_input_channels = _detect_input_channels(sample0_transformed, cfg)
    log.info(f"Input channels: {num_input_channels}, Output channels: 2")
    cfg.model.args.magno.precompute_edges = cfg.model.args.magno.asynchronous_graph_building
    model = load_checkpoint(build_model(cfg, num_input_channels, 2, device), args.ckpt)
    model.eval()
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Run inference, print summary, save CSV ───────────────────────
    all_results, all_rows = _run_inference_loop(
        split_names, model, composed_transform, collate_fn, latent_tokens, per_head_stats, device, args, data_root
    )
    _print_inference_summary(all_results, args)
    csv_path = os.path.join(args.out, f"inference_{args.split}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    log.info(f"Results saved to {csv_path}")
    log.info("Inference complete.")


if __name__ == "__main__":
    main()
