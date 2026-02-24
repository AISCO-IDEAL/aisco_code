#!/usr/bin/env python3
"""
Integrate GAOT-predicted pressure (and optionally shear) fields into drag force.

Takes GAOT inference output — per-vertex pressure predictions on a surface mesh —
and integrates them to produce a single scalar drag force and Cd value.

This can operate in three modes:

  1. **Single .pt file** — load one GAOT data.pt, integrate its pressure field:
       python integrate_pressure_field.py /path/to/data.pt

  2. **Dataset directory** — integrate all samples in a processed dataset:
       python integrate_pressure_field.py /path/to/processed_dir/

  3. **Model inference** — load a trained GAOT checkpoint and run inference on
     a dataset, then integrate the predicted field:
       python integrate_pressure_field.py /path/to/processed_dir/ --ckpt model.pt --config pressure_field.yaml

Typical flow:
  GAOT training → checkpoint → inference on test set → this script integrates
  the predicted pressure field → drag force + Cd → compare against ground truth.

Integration formula:
    F_drag = Σ_i  p_i · (n_i · e_x) · A_i               (pressure drag)
    F_fric = Σ_i  τ_x,i · A_i                            (friction drag)
    Cd     = F_drag / (0.5 · ρ · U² · A_ref)
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.pressure_integration import (
    compute_vertex_areas,
    drag_to_cd,
    pressure_shear_to_drag,
    pressure_to_drag,
)


# ---------------------------------------------------------------------------
#  Default flow parameters
# ---------------------------------------------------------------------------
DEFAULT_RHO = 1.225       # Air density [kg/m³] at sea level
DEFAULT_U_INF = 30.0      # Freestream velocity [m/s]
DEFAULT_A_REF = 2.16      # Reference area [m²] (DrivAer-style)
DRAG_AXIS = 0             # x-axis for drag direction


def integrate_single_sample(data, rho=DEFAULT_RHO, u_inf=DEFAULT_U_INF,
                            a_ref=DEFAULT_A_REF, drag_axis=DRAG_AXIS,
                            use_shear=False, pressure_key="x"):
    """
    Integrate pressure (and optionally shear) from one PyG Data sample.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Must have: pos [N,3], c [N,3] (normals), face [3,F],
                   and a pressure field (default: data.x [N,1]).
        Optionally: wall_shear [N,3] for friction drag.
    rho : float
        Freestream density.
    u_inf : float
        Freestream velocity.
    a_ref : float
        Reference area.
    drag_axis : int
        Axis index for drag direction (0=x, 1=y, 2=z).
    use_shear : bool
        Include wall shear stress (friction drag) in total drag.
    pressure_key : str
        Attribute name for pressure field. Default "x".

    Returns
    -------
    dict with keys:
        drag_pressure : float   — pressure drag [N]
        drag_total    : float   — total drag (pressure + friction) [N]
        cd_pressure   : float   — Cd from pressure drag only
        cd_total      : float   — Cd from total drag
        gt_force      : float   — ground truth force (NaN if unavailable)
        gt_cd         : float   — ground truth Cd (NaN if unavailable)
        n_vertices    : int     — number of surface vertices
        total_area    : float   — total integrated surface area
    """
    pressure = getattr(data, pressure_key, None)
    if pressure is None:
        raise ValueError(f"Data has no attribute '{pressure_key}' — no pressure field to integrate.")

    pos = data.pos.float()
    normals = data.c.float()
    face = data.face.long()

    # Pressure drag
    drag_p = pressure_to_drag(pressure, normals, face, pos, drag_direction=drag_axis)

    # Total drag (pressure + friction)
    if use_shear and hasattr(data, "wall_shear") and data.wall_shear is not None:
        drag_total = pressure_shear_to_drag(
            pressure, data.wall_shear.float(), normals, face, pos,
            drag_direction=drag_axis,
        )
    else:
        drag_total = drag_p

    # Cd
    cd_p = drag_to_cd(drag_p, rho=rho, u_inf=u_inf, A_ref=a_ref)
    cd_total = drag_to_cd(drag_total, rho=rho, u_inf=u_inf, A_ref=a_ref)

    # Ground truth
    gt_force = float("nan")
    gt_cd = float("nan")
    if hasattr(data, "force") and data.force is not None:
        gt_force = data.force.item()
    if hasattr(data, "cd") and data.cd is not None:
        gt_cd = data.cd.item()

    # Surface area
    vertex_areas = compute_vertex_areas(pos, face)
    total_area = vertex_areas.sum().item()

    return {
        "drag_pressure": drag_p.item(),
        "drag_total": drag_total.item(),
        "cd_pressure": cd_p.item(),
        "cd_total": cd_total.item(),
        "gt_force": gt_force,
        "gt_cd": gt_cd,
        "n_vertices": pos.shape[0],
        "total_area": total_area,
    }


def format_result(name, result, show_gt=True):
    """Format one integration result as a human-readable string."""
    lines = [f"  {name}:"]
    lines.append(f"    Pressure drag = {result['drag_pressure']:>12.4f} N")
    if abs(result["drag_total"] - result["drag_pressure"]) > 1e-6:
        lines.append(f"    Total drag    = {result['drag_total']:>12.4f} N  (pressure + friction)")
    lines.append(f"    Cd (pressure) = {result['cd_pressure']:>12.6f}")
    if abs(result["cd_total"] - result["cd_pressure"]) > 1e-6:
        lines.append(f"    Cd (total)    = {result['cd_total']:>12.6f}")
    if show_gt and not np.isnan(result["gt_force"]):
        lines.append(f"    GT force      = {result['gt_force']:>12.4f} N")
    if show_gt and not np.isnan(result["gt_cd"]):
        lines.append(f"    GT Cd         = {result['gt_cd']:>12.6f}")
    lines.append(f"    Vertices: {result['n_vertices']:,}   Area: {result['total_area']:.4f} m²")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Mode 1: Single .pt file
# ---------------------------------------------------------------------------
def integrate_single_file(pt_path, args):
    """Load a single .pt file and integrate its pressure field."""
    data = torch.load(pt_path, weights_only=False)
    result = integrate_single_sample(
        data,
        rho=args.rho, u_inf=args.u_inf, a_ref=args.a_ref,
        drag_axis=args.drag_axis, use_shear=args.use_shear,
        pressure_key=args.pressure_key,
    )
    name = getattr(data, "filename", os.path.basename(pt_path))
    print(format_result(name, result))
    return result


# ---------------------------------------------------------------------------
#  Mode 2: Dataset directory
# ---------------------------------------------------------------------------
def integrate_dataset(data_dir, args):
    """Integrate pressure fields for all samples in a dataset directory."""
    import glob

    # Find all data.pt files
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*/data.pt")))
    if not pt_files:
        # Check if data_dir itself contains data.pt
        single = os.path.join(data_dir, "data.pt")
        if os.path.exists(single):
            return integrate_single_file(single, args)
        print(f"No data.pt files found in {data_dir}")
        return

    print(f"Integrating {len(pt_files)} samples from {data_dir}")
    print(f"  rho={args.rho}, u_inf={args.u_inf}, a_ref={args.a_ref}, drag_axis={args.drag_axis}")
    print(f"  use_shear={args.use_shear}, pressure_key={args.pressure_key}")
    print()

    results = []
    for pt_path in pt_files:
        data = torch.load(pt_path, weights_only=False)
        name = getattr(data, "filename", os.path.basename(os.path.dirname(pt_path)))
        try:
            result = integrate_single_sample(
                data,
                rho=args.rho, u_inf=args.u_inf, a_ref=args.a_ref,
                drag_axis=args.drag_axis, use_shear=args.use_shear,
                pressure_key=args.pressure_key,
            )
            result["name"] = name
            results.append(result)
            print(format_result(name, result))
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    if not results:
        return

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    drag_vals = [r["drag_pressure"] for r in results]
    cd_vals = [r["cd_pressure"] for r in results]
    print(f"  Samples:       {len(results)}")
    print(f"  Drag (pressure): mean={np.mean(drag_vals):.4f}  std={np.std(drag_vals):.4f}  "
          f"min={np.min(drag_vals):.4f}  max={np.max(drag_vals):.4f}")
    print(f"  Cd (pressure):   mean={np.mean(cd_vals):.6f}  std={np.std(cd_vals):.6f}  "
          f"min={np.min(cd_vals):.6f}  max={np.max(cd_vals):.6f}")

    # Compare against ground truth if available
    gt_forces = [r["gt_force"] for r in results if not np.isnan(r["gt_force"])]
    gt_cds = [r["gt_cd"] for r in results if not np.isnan(r["gt_cd"])]
    if gt_forces:
        pred_forces = [r["drag_pressure"] for r, _ in zip(results, gt_forces)]
        errors = [abs(p - g) for p, g in zip(pred_forces, gt_forces)]
        rel_errors = [abs(p - g) / max(abs(g), 1e-6) for p, g in zip(pred_forces, gt_forces)]
        print(f"\n  vs GT force ({len(gt_forces)} samples):")
        print(f"    MAE  = {np.mean(errors):.4f} N")
        print(f"    MAPE = {np.mean(rel_errors)*100:.2f}%")
    if gt_cds:
        pred_cds = [r["cd_pressure"] for r, _ in zip(results, gt_cds)]
        cd_errors = [abs(p - g) for p, g in zip(pred_cds, gt_cds)]
        cd_rel = [abs(p - g) / max(abs(g), 1e-6) for p, g in zip(pred_cds, gt_cds)]
        print(f"\n  vs GT Cd ({len(gt_cds)} samples):")
        print(f"    MAE  = {np.mean(cd_errors):.6f}")
        print(f"    MAPE = {np.mean(cd_rel)*100:.2f}%")

    # Save CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "name", "drag_pressure", "drag_total", "cd_pressure", "cd_total",
                "gt_force", "gt_cd", "n_vertices", "total_area",
            ])
            w.writeheader()
            w.writerows(results)
        print(f"\n  Results saved to: {args.output_csv}")

    return results


# ---------------------------------------------------------------------------
#  Mode 3: Model inference + integration
# ---------------------------------------------------------------------------
def integrate_from_model(data_dir, args):
    """Run GAOT inference on a dataset, then integrate the predicted field."""
    from omegaconf import OmegaConf
    from torch_geometric.data import Data, Batch

    from src.model import init_model

    # Load config
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model

    # Determine input channels
    encoder_attr = model_cfg.args.magno.get("encoder_feature_attr", "x")
    if isinstance(encoder_attr, list):
        # e.g. ["pos", "c"] → 3 + 3 = 6
        input_size = sum(3 for _ in encoder_attr)  # assuming all are 3D
    else:
        input_size = 3

    # Create model
    model = init_model(
        input_size=input_size,
        output_size=2,  # force + cd scalar heads
        model=model_cfg.name,
        config=model_cfg.args,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Handle DDP checkpoint (keys may have 'module.' prefix)
    state_dict = ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print(f"Model loaded from {args.ckpt}")
    print(f"  field_output_size = {model.field_output_size}")
    if model.field_output_size == 0:
        print("  WARNING: Model has no field decoder (field_output_size=0).")
        print("  Cannot integrate predicted pressure field. Use a model trained with field_output_size > 0.")
        return

    # Load dataset
    import glob
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*/data.pt")))
    print(f"  Running inference on {len(pt_files)} samples...")

    results = []
    latent_tokens = model.latent_tokens.to(device)

    with torch.no_grad():
        for pt_path in pt_files:
            data = torch.load(pt_path, weights_only=False)
            name = getattr(data, "filename", os.path.basename(os.path.dirname(pt_path)))

            batch = Batch.from_data_list([data]).to(device)

            # Run model
            pred = model(batch=batch, tokens_pos=latent_tokens)

            if "field" not in pred:
                print(f"  {name}: no 'field' in prediction, skipping")
                continue

            # Replace data.x with predicted pressure for integration
            pred_pressure = pred["field"].cpu()  # [N, field_output_size]
            data_copy = data.clone()
            data_copy.x = pred_pressure

            try:
                result = integrate_single_sample(
                    data_copy,
                    rho=args.rho, u_inf=args.u_inf, a_ref=args.a_ref,
                    drag_axis=args.drag_axis, use_shear=args.use_shear,
                    pressure_key="x",
                )
                result["name"] = name

                # Also store predicted scalar values
                if "force" in pred:
                    result["pred_scalar_force"] = pred["force"][0].item()
                if "cd" in pred:
                    result["pred_scalar_cd"] = pred["cd"][0].item()

                results.append(result)
                print(format_result(name, result))
            except Exception as e:
                print(f"  {name}: integration failed — {e}")

    # Summary
    if results:
        print("\n" + "=" * 60)
        print(f"SUMMARY — {len(results)} samples")
        print("=" * 60)
        drag_vals = [r["cd_pressure"] for r in results]
        print(f"  Cd (integrated): mean={np.mean(drag_vals):.6f}  std={np.std(drag_vals):.6f}")

        if args.output_csv:
            with open(args.output_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                w.writeheader()
                w.writerows(results)
            print(f"  Results saved to: {args.output_csv}")

    return results


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Integrate GAOT pressure field predictions into drag force.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Integrate ground truth pressure from a single .pt file
  python integrate_pressure_field.py /path/to/euler_run_1/data.pt

  # Integrate all samples in a processed dataset directory
  python integrate_pressure_field.py /path/to/processed_euler_pressure/ --use_shear

  # Run GAOT inference + integration from a checkpoint
  python integrate_pressure_field.py /path/to/processed_dir/ \\
      --ckpt /path/to/field_model.pt --config config/pressure_field.yaml

  # Save results to CSV
  python integrate_pressure_field.py /path/to/dataset/ --output_csv results.csv
        """,
    )
    parser.add_argument(
        "input", type=str,
        help="Path to a .pt file or a dataset directory containing */data.pt"
    )
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO,
                        help=f"Freestream density [kg/m³] (default: {DEFAULT_RHO})")
    parser.add_argument("--u_inf", type=float, default=DEFAULT_U_INF,
                        help=f"Freestream velocity [m/s] (default: {DEFAULT_U_INF})")
    parser.add_argument("--a_ref", type=float, default=DEFAULT_A_REF,
                        help=f"Reference area [m²] (default: {DEFAULT_A_REF})")
    parser.add_argument("--drag_axis", type=int, default=DRAG_AXIS, choices=[0, 1, 2],
                        help="Drag direction axis: 0=x, 1=y, 2=z (default: 0)")
    parser.add_argument("--use_shear", action="store_true",
                        help="Include wall shear stress (friction drag) if available")
    parser.add_argument("--pressure_key", type=str, default="x",
                        help="Data attribute name for pressure field (default: 'x')")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Save per-sample results to this CSV file")

    # Model inference mode
    parser.add_argument("--ckpt", type=str, default=None,
                        help="GAOT checkpoint path — enables inference mode")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML for model architecture (required with --ckpt)")

    args = parser.parse_args()

    # Determine mode
    if args.ckpt:
        if args.config is None:
            parser.error("--config is required when using --ckpt (inference mode)")
        integrate_from_model(args.input, args)
    elif os.path.isfile(args.input) and args.input.endswith(".pt"):
        integrate_single_file(args.input, args)
    elif os.path.isdir(args.input):
        integrate_dataset(args.input, args)
    else:
        parser.error(f"Input '{args.input}' is not a .pt file or directory")


if __name__ == "__main__":
    main()
