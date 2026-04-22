#!/usr/bin/env python3
"""
Batch inference on preprocessed .pt files
==========================================

Loads a trained GAOT checkpoint + its config.yaml, then for every .pt file
in the specified data folder runs inference and prints the integrated drag
force (predicted vs ground-truth).

Supports all three preprocessed dataset layouts produced by the GAOT3
preprocessing scripts:

  Blended Wing   flat layout:  <data_dir>/case_NNNN.pt
                 pos in chord-normalised units (dimensionless ~0-1)
                 force = NaN  →  GT from integrating data.x
                 v_inf stored per sample in data.v_inf

  Euler          subdir layout: <data_dir>/euler_run_NNN/data.pt
                 pos in metres (x-axis already flipped during preprocessing)
                 force = physical drag force [N]
                 v_inf = 25 m/s (stored in data.v_inf)

  DrivAer/Ahmed  subdir layout: <data_dir>/<prefix>_N/data.pt
                 pos in metres
                 force = drag [N] (DrivAer) or NaN (Ahmed)
                 v_inf stored per sample in data.v_inf

NOTE: The preprocessing scripts store positions without applying any extra
x-axis flip (Euler is the exception — it bakes the flip in).  This script
does NOT re-apply a flip; it feeds the data to the model exactly as stored,
consistent with the training inference pipeline (scripts/inference_field.py).

Usage
-----
  python scripts/infer_from_pt.py \\
      --ckpt  /path/to/model.pt \\
      --data  /path/to/preprocessed_data/ \\
      --v-inf 30.0 --rho 1.225 --a-ref 2.16

  # Override v_inf for all samples (otherwise data.v_inf is used per sample):
  python scripts/infer_from_pt.py --ckpt ... --data ... --v-inf 25.0 --force-v-inf
"""

import argparse
import copy
import math
import os
import sys
import warnings

import torch
import yaml
from torch_geometric.data import Batch, Data

# Make sure the project root is importable when running the script directly
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.metadata import DATASET_METADATA
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.pyg_transforms import (
    RescalePosition,
    RescalePositionNew,
)
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.pressure_integration import (
    cp_field_to_drag,
    drag_to_cd,
)
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.scale import rescale, rescale_new


# ─────────────────────────────────────────────────────────────────────────
#  SAMPLE DISCOVERY
#  Handles both flat (.pt) and subdirectory (*/data.pt) layouts.
# ─────────────────────────────────────────────────────────────────────────

def discover_samples(data_dir: str) -> list[tuple[str, str]]:
    """Return a sorted list of (sample_name, pt_path) pairs.

    Scans for:
      - flat files:         <data_dir>/<name>.pt
      - subdirectory files: <data_dir>/<name>/data.pt
    """
    entries = sorted(os.listdir(data_dir))
    samples = []
    for entry in entries:
        full = os.path.join(data_dir, entry)
        if entry.endswith(".pt") and os.path.isfile(full):
            name = entry[:-3]  # strip .pt
            # Skip dataset-level statistics artifacts; they are dicts, not PyG Data samples.
            if name.endswith("_norm_stats"):
                continue
            samples.append((name, full))
        elif os.path.isdir(full):
            subfile = os.path.join(full, "data.pt")
            if os.path.isfile(subfile):
                samples.append((entry, subfile))
    return samples


# ─────────────────────────────────────────────────────────────────────────
#  MODEL LOADING  (mirrors ModelLoader._load_model)
# ─────────────────────────────────────────────────────────────────────────

def load_gaot(ckpt_path: str, device: torch.device):
    """Load GAOT3D from a checkpoint + its sibling config.yaml.

    Returns (model, og_config, latent_tokens, use_rescale_new, phy_domain).
    """
    config_path = ckpt_path.replace("model.pt", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found at {config_path}. Expected config.yaml next to model.pt."
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise TypeError("Loaded config must be a dictionary.")

    og_config = copy.deepcopy(config)
    model_args = config["model"]["args"]
    model_args["magno"]["precompute_edges"] = False
    model_args["transformer"]["norm_eps"] = float(model_args["transformer"]["norm_eps"])

    model = GAOT3D(config=model_args).to(device)
    model.eval()

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        sd = state["model"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state

    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        warnings.warn(
            f"Checkpoint mismatch. Missing: {missing}. Unexpected: {unexpected}.",
            stacklevel=2,
        )

    # ── Build latent tokens (same logic as ModelLoader._load_model) ──
    dataset_cfg = og_config.get("dataset", {})
    use_rescale_new = dataset_cfg.get("use_rescale_new", False)

    metaname = dataset_cfg.get("metaname")
    if metaname and metaname in DATASET_METADATA:
        phy_domain = DATASET_METADATA[metaname].domain_x
    else:
        phy_domain = model_args.get("norm_domain")
        if phy_domain is None:
            raise KeyError(
                "Cannot determine physical domain. Expected dataset.metaname in "
                "DATASET_METADATA or model.args.norm_domain."
            )

    latent_token_size = model_args.get("latent_tokens")
    if latent_token_size is None:
        raise KeyError("Expected model.args.latent_tokens in saved config.")

    x_min, y_min, z_min = phy_domain[0]
    x_max, y_max, z_max = phy_domain[1]

    meshgrid = torch.meshgrid(
        torch.linspace(x_min, x_max, latent_token_size[0]),
        torch.linspace(y_min, y_max, latent_token_size[1]),
        torch.linspace(z_min, z_max, latent_token_size[2]),
        indexing="ij",
    )
    latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)

    if use_rescale_new:
        latent_tokens = rescale_new(latent_queries, (-1, 1), phy_domain).to(device)
    else:
        latent_tokens = rescale(latent_queries, (-1, 1)).to(device)

    return model, og_config, latent_tokens, use_rescale_new, phy_domain


# ─────────────────────────────────────────────────────────────────────────
#  DATA PREPARATION
#  Applies only positional rescaling — no x-flip.
#  The preprocessing scripts already store data in the model's expected
#  coordinate convention (Euler bakes the flip; others don't need it).
#  This matches the training inference pipeline (scripts/inference_field.py).
# ─────────────────────────────────────────────────────────────────────────

def prepare_batch(data: Data, og_config: dict, use_rescale_new: bool, phy_domain, device: torch.device):
    """Rescale positions into [-1, 1] and build a single-sample Batch.

    Returns (batch, pos_raw, face_raw, normals_raw) where the *_raw tensors
    are in the original physical / stored units and are used for drag integration.
    """
    pos     = data.pos.clone().to(device).float()
    face    = data.face.clone().to(device).long()
    normals = data.c.clone().to(device).float()

    # Raw copies for drag integration (before rescaling)
    pos_raw     = pos.clone()
    # Use c_raw when present (some future datasets may store it); fall back to c
    normals_raw = (
        data.c_raw.clone().to(device).float()
        if hasattr(data, "c_raw") and data.c_raw is not None
        else normals.clone()
    )

    pyg_data = Data(pos=pos, face=face, c=normals)

    # Apply the same positional rescaling as during training
    if use_rescale_new:
        transform = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=phy_domain)
    else:
        transform = RescalePosition()

    pyg_data = transform(pyg_data)

    # RescalePosition stores the pre-rescale positions as pos_raw
    if hasattr(pyg_data, "pos_raw") and pyg_data.pos_raw is not None:
        pos_raw = pyg_data.pos_raw.to(device).float()

    batch = Batch.from_data_list([pyg_data]).to(device)
    return batch, pos_raw, face, normals_raw


# ─────────────────────────────────────────────────────────────────────────
#  SINGLE-SAMPLE INFERENCE
# ─────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_one(
    data: Data,
    model: GAOT3D,
    og_config: dict,
    latent_tokens: torch.Tensor,
    use_rescale_new: bool,
    phy_domain,
    v_inf_default: float,
    rho: float,
    a_ref: float,
    force_v_inf: bool,
    device: torch.device,
) -> dict:
    """Run inference on a single Data object.

    GT drag priority:
      1. data.force  — drag force from the force report (physically exact)
      2. integrate data.x (Cp field) — used when force is NaN or absent

    v_inf priority:
      1. data.v_inf  — per-sample stored freestream (blended wing, etc.)
      2. v_inf_default (--v-inf CLI argument) when --force-v-inf is set
         or data.v_inf is absent
    """
    # ── Resolve v_inf ──
    sample_v_inf = v_inf_default
    if not force_v_inf and hasattr(data, "v_inf") and data.v_inf is not None:
        v_inf_val = data.v_inf.item() if hasattr(data.v_inf, "item") else float(data.v_inf)
        if v_inf_val > 0:
            sample_v_inf = v_inf_val

    batch, pos_raw, face_raw, normals_raw = prepare_batch(
        data, og_config, use_rescale_new, phy_domain, device
    )

    pred_out = model.forward(batch=batch, tokens_pos=latent_tokens)
    pred_cp = pred_out["field"]
    if pred_cp.dim() == 2 and pred_cp.size(-1) == 1:
        pred_cp = pred_cp.squeeze(-1)

    # ── Predicted drag via Cp integration ──
    pred_drag = cp_field_to_drag(
        cp_field=pred_cp,
        normals=normals_raw,
        face=face_raw.T,
        pos=pos_raw,
        v_inf=sample_v_inf,
        rho=rho,
    ).item()

    q_inf     = 0.5 * rho * sample_v_inf**2
    pred_cd   = pred_drag / (q_inf * a_ref)

    # ── GT drag: prefer data.force, fall back to integrating data.x ──
    gt_drag    = float("nan")
    gt_cd      = float("nan")
    gt_source  = "none"

    force_attr = getattr(data, "force", None)
    if force_attr is not None:
        force_val = force_attr.item() if hasattr(force_attr, "item") else float(force_attr)
        if not math.isnan(force_val):
            gt_drag   = force_val
            gt_cd     = drag_to_cd(torch.tensor(gt_drag), rho=rho, u_inf=sample_v_inf, A_ref=a_ref).item()
            gt_source = "force_report"

    if math.isnan(gt_drag) and hasattr(data, "x") and data.x is not None:
        gt_cp   = data.x.to(device).float().squeeze(-1)
        gt_drag = cp_field_to_drag(
            cp_field=gt_cp,
            normals=normals_raw,
            face=face_raw.T,
            pos=pos_raw,
            v_inf=sample_v_inf,
            rho=rho,
        ).item()
        gt_cd     = drag_to_cd(torch.tensor(gt_drag), rho=rho, u_inf=sample_v_inf, A_ref=a_ref).item()
        gt_source = "cp_integration"

    return {
        "pred_drag":  pred_drag,
        "pred_cd":    pred_cd,
        "gt_drag":    gt_drag,
        "gt_cd":      gt_cd,
        "gt_source":  gt_source,
        "v_inf_used": sample_v_inf,
    }


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run GAOT inference on preprocessed .pt files and compare integrated drag with GT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",         required=True,  help="Path to model.pt checkpoint")
    p.add_argument("--data",         required=True,  help="Folder containing preprocessed .pt files (flat or subdirectory layout)")
    p.add_argument("--v-inf",        type=float, default=30.0,  help="Fallback freestream velocity [m/s] when data.v_inf is absent")
    p.add_argument("--rho",          type=float, default=1.225, help="Air density [kg/m³]")
    p.add_argument("--a-ref",        type=float, default=2.16,  help="Reference area for Cd [m²]")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--force-v-inf",
        action="store_true",
        help="Ignore data.v_inf and always use --v-inf for all samples.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"Loading checkpoint: {args.ckpt}")
    model, og_config, latent_tokens, use_rescale_new, phy_domain = load_gaot(args.ckpt, device)
    print(f"Model loaded on {device}\n")

    samples = discover_samples(args.data)
    if not samples:
        raise FileNotFoundError(
            f"No .pt samples found in {args.data}.\n"
            "Expected flat files (<name>.pt) or subdirectories (<name>/data.pt)."
        )
    print(f"Found {len(samples)} samples in {args.data}\n")

    col_w = 44
    header = (
        f"{'Sample':<{col_w}}  {'v_inf':>6}  {'GT src':>14}  "
        f"{'GT drag [N]':>12}  {'Pred drag [N]':>13}  {'Δdrag [N]':>10}  "
        f"{'GT Cd':>8}  {'Pred Cd':>8}"
    )
    print(header)
    print("-" * len(header))

    total_gt   = 0.0
    total_pred = 0.0
    count      = 0

    for sample_name, pt_path in samples:
        data = torch.load(pt_path, weights_only=False)

        if not isinstance(data, Data):
            print(f"{'SKIP':<{col_w}}  {sample_name}: expected torch_geometric.data.Data, got {type(data).__name__}")
            continue

        try:
            m = infer_one(
                data=data,
                model=model,
                og_config=og_config,
                latent_tokens=latent_tokens,
                use_rescale_new=use_rescale_new,
                phy_domain=phy_domain,
                v_inf_default=args.v_inf,
                rho=args.rho,
                a_ref=args.a_ref,
                force_v_inf=args.force_v_inf,
                device=device,
            )
        except Exception as exc:
            print(f"{'ERROR':<{col_w}}  {sample_name}: {exc}")
            continue

        gt_drag   = m["gt_drag"]
        pred_drag = m["pred_drag"]
        delta     = pred_drag - gt_drag if not math.isnan(gt_drag) else float("nan")
        gt_src    = m["gt_source"]
        v_used    = m["v_inf_used"]

        print(
            f"{sample_name:<{col_w}}  {v_used:>6.1f}  {gt_src:>14}  "
            f"{gt_drag:>12.2f}  {pred_drag:>13.2f}  {delta:>10.2f}  "
            f"{m['gt_cd']:>8.4f}  {m['pred_cd']:>8.4f}"
        )

        if not math.isnan(gt_drag):
            total_gt   += gt_drag
            total_pred += pred_drag
            count      += 1

    if count > 0:
        mean_gt   = total_gt   / count
        mean_pred = total_pred / count
        # Use fallback v_inf for mean Cd (mix of v_inf is not ideal but gives a rough Cd)
        q_inf = 0.5 * args.rho * args.v_inf**2
        print("-" * len(header))
        print(
            f"{'MEAN (GT-known samples)':<{col_w}}  {'':>6}  {'':>14}  "
            f"{mean_gt:>12.2f}  {mean_pred:>13.2f}  {mean_pred - mean_gt:>10.2f}  "
            f"{mean_gt / (q_inf * args.a_ref):>8.4f}  {mean_pred / (q_inf * args.a_ref):>8.4f}"
        )
    print(f"\nSamples evaluated: {count} / {len(samples)}")


if __name__ == "__main__":
    main()
