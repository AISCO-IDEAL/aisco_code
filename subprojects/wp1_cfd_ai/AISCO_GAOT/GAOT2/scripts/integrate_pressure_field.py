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
    Cd     = F_drag / (0.5 * rho * U^2 * A_ref)
"""

import argparse
import csv
import glob
import os
import sys

# Add project root to path so that 'src.*' imports work regardless of cwd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch

from src.model.gaot_3d import GAOT3D
from src.utils.pressure_integration import (
    compute_frontal_area,
    compute_vertex_areas,
    cp_to_pressure,
    drag_to_cd,
    pressure_shear_to_drag,
    pressure_to_drag,
)

# ---------------------------------------------------------------------------
#  Default flow parameters
# ---------------------------------------------------------------------------
DEFAULT_RHO = 1.225  # Air density [kg/m³] at sea level
DEFAULT_U_INF = None  # When None, read from data.v_inf per sample
DEFAULT_A_REF = None  # When None, compute frontal area from mesh per sample
DRAG_AXIS = 0  # x-axis for drag direction
SMALL_DIFF = 1e-6  # threshold for treating two floats as equal

# Per-dataset defaults (fallbacks when data.v_inf is missing)
DATASET_DEFAULTS = {
    "euler": {"v_inf": 25.0, "rho": 1.225},
    "drivaer": {"v_inf": 38.89, "rho": 1.225},
    "ahmed": {"v_inf": 1.0, "rho": 1.225},
    "blended_wing": {"v_inf": None, "rho": 1.225},  # v_inf varies per sample (Mach)
}


def _resolve_v_inf(data, u_inf_override):
    """Return the freestream velocity to use for this sample.

    Priority: CLI override > data.v_inf attribute > error.
    """
    if u_inf_override is not None:
        return u_inf_override
    if hasattr(data, "v_inf") and data.v_inf is not None:
        return float(data.v_inf.item() if hasattr(data.v_inf, "item") else data.v_inf)
    raise ValueError(
        "No freestream velocity available. Either pass --u_inf on the CLI or ensure data.v_inf is stored in the .pt file."
    )


def _resolve_a_ref(data, a_ref_override, drag_axis):
    """Return the reference area to use for this sample.

    Priority: CLI override > compute frontal area from mesh.
    """
    if a_ref_override is not None:
        return a_ref_override
    # Auto-compute from mesh projected frontal area
    pos = data.pos.float()
    face = data.face.long()
    normals = data.c.float()
    return compute_frontal_area(pos, face, normals, drag_direction=drag_axis)


def integrate_single_sample(
    data,
    rho=DEFAULT_RHO,
    u_inf=DEFAULT_U_INF,
    a_ref=DEFAULT_A_REF,
    drag_axis=DRAG_AXIS,
    use_shear=False,
    pressure_key="x",
    is_cp=False,
):
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
    u_inf : float or None
        Freestream velocity.  If None, read from ``data.v_inf``.
    a_ref : float or None
        Reference area.  If None, computed as the projected frontal area
        of the mesh perpendicular to the drag axis.
    drag_axis : int
        Axis index for drag direction (0=x, 1=y, 2=z).
    use_shear : bool
        Include wall shear stress (friction drag) in total drag.
    pressure_key : str
        Attribute name for pressure field. Default "x".
    is_cp : bool
        If True, the pressure field is Cp (pressure coefficient) and will
        be converted to Pascals using ``u_inf`` and ``rho`` before integration.

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
        a_ref         : float   — reference area used for Cd
        v_inf         : float   — freestream velocity used
    """
    pressure = getattr(data, pressure_key, None)
    if pressure is None:
        raise ValueError(f"Data has no attribute '{pressure_key}' — no pressure field to integrate.")

    pos = data.pos.float()
    normals = data.c.float()
    face = data.face.long()

    # Resolve per-sample v_inf and a_ref
    sample_v_inf = _resolve_v_inf(data, u_inf)
    sample_a_ref = _resolve_a_ref(data, a_ref, drag_axis)

    # Convert Cp -> Pa if needed
    pressure = cp_to_pressure(pressure.float(), v_inf=sample_v_inf, rho=rho) if is_cp else pressure.float()

    # Pressure drag
    drag_p = pressure_to_drag(pressure, normals, face, pos, drag_direction=drag_axis)

    # Total drag (pressure + friction)
    if use_shear and hasattr(data, "wall_shear") and data.wall_shear is not None:
        drag_total = pressure_shear_to_drag(
            pressure,
            data.wall_shear.float(),
            normals,
            face,
            pos,
            drag_direction=drag_axis,
        )
    else:
        drag_total = drag_p

    # Cd
    cd_p = drag_to_cd(drag_p, rho=rho, u_inf=sample_v_inf, A_ref=sample_a_ref)
    cd_total = drag_to_cd(drag_total, rho=rho, u_inf=sample_v_inf, A_ref=sample_a_ref)

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
        "a_ref": sample_a_ref,
        "v_inf": sample_v_inf,
    }


def format_result(name, result, show_gt=True):
    """Format one integration result as a human-readable string."""
    lines = [f"  {name}:"]
    lines.append(f"    Pressure drag = {result['drag_pressure']:>12.4f} N")
    if abs(result["drag_total"] - result["drag_pressure"]) > SMALL_DIFF:
        lines.append(f"    Total drag    = {result['drag_total']:>12.4f} N  (pressure + friction)")
    lines.append(f"    Cd (pressure) = {result['cd_pressure']:>12.6f}")
    if abs(result["cd_total"] - result["cd_pressure"]) > SMALL_DIFF:
        lines.append(f"    Cd (total)    = {result['cd_total']:>12.6f}")
    if show_gt and not np.isnan(result["gt_force"]):
        lines.append(f"    GT force      = {result['gt_force']:>12.4f} N")
    if show_gt and not np.isnan(result["gt_cd"]):
        lines.append(f"    GT Cd         = {result['gt_cd']:>12.6f}")
    lines.append(
        f"    Vertices: {result['n_vertices']:,}   Area: {result['total_area']:.4f} m²"
        f"   A_ref: {result['a_ref']:.4f} m²   v_inf: {result['v_inf']:.2f} m/s"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Mode 1: Single .pt file
# ---------------------------------------------------------------------------
def integrate_single_file(pt_path, args):
    """Load a single .pt file and integrate its pressure field."""
    data = torch.load(pt_path, weights_only=False)
    result = integrate_single_sample(
        data,
        rho=args.rho,
        u_inf=args.u_inf,
        a_ref=args.a_ref,
        drag_axis=args.drag_axis,
        use_shear=args.use_shear,
        pressure_key=args.pressure_key,
        is_cp=args.is_cp,
    )
    name = getattr(data, "filename", os.path.basename(pt_path))
    print(format_result(name, result))
    return result


# ---------------------------------------------------------------------------
#  Mode 2: Dataset directory
# ---------------------------------------------------------------------------
def _find_pt_files(data_dir):
    """Find all .pt sample files in a dataset directory.

    Supports two layouts:
      - ``<data_dir>/<subdir>/data.pt``   (euler, drivaer, ahmed)
      - ``<data_dir>/<name>.pt``          (blended_wing)
    Excludes known non-sample files (norm_stats, order files, etc.).
    """
    EXCLUDE_PREFIXES = ("norm_stats", "order_", "metadata")

    # Layout 1: subdirectory/data.pt
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*/data.pt")))
    if pt_files:
        return pt_files

    # Layout 2: flat directory with *.pt
    pt_files = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
    pt_files = [f for f in pt_files if not any(os.path.basename(f).startswith(pre) for pre in EXCLUDE_PREFIXES)]
    return pt_files


def integrate_dataset(data_dir, args):
    """Integrate pressure fields for all samples in a dataset directory."""

    pt_files = _find_pt_files(data_dir)
    if not pt_files:
        print(f"No .pt sample files found in {data_dir}")
        return

    u_inf_str = f"{args.u_inf}" if args.u_inf is not None else "per-sample"
    a_ref_str = f"{args.a_ref}" if args.a_ref is not None else "frontal-area"
    print(f"Integrating {len(pt_files)} samples from {data_dir}")
    print(f"  rho={args.rho}, u_inf={u_inf_str}, a_ref={a_ref_str}, drag_axis={args.drag_axis}")
    print(f"  is_cp={args.is_cp}, use_shear={args.use_shear}, pressure_key={args.pressure_key}")
    print()

    results = []
    for pt_path in pt_files:
        data = torch.load(pt_path, weights_only=False)
        # Name: prefer data.filename, else parent dir name (layout 1) or stem (layout 2)
        if hasattr(data, "filename") and data.filename:
            name = data.filename
        elif os.path.basename(pt_path) == "data.pt":
            name = os.path.basename(os.path.dirname(pt_path))
        else:
            name = os.path.splitext(os.path.basename(pt_path))[0]
        try:
            result = integrate_single_sample(
                data,
                rho=args.rho,
                u_inf=args.u_inf,
                a_ref=args.a_ref,
                drag_axis=args.drag_axis,
                use_shear=args.use_shear,
                pressure_key=args.pressure_key,
                is_cp=args.is_cp,
            )
            result["name"] = name
            results.append(result)
            print(format_result(name, result))
        except Exception as e:
            print(f"  {name}: FAILED — {e}")

    if not results:
        return

    _print_dataset_summary(results, args)
    return results


def _print_dataset_summary(results: list, args) -> None:
    """Print summary statistics and optionally save CSV for dataset-integration results."""
    drag_vals = [r["drag_pressure"] for r in results]
    cd_vals = [r["cd_pressure"] for r in results]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Samples:       {len(results)}")
    print(
        f"  Drag (pressure): mean={np.mean(drag_vals):.4f}  std={np.std(drag_vals):.4f}  "
        f"min={np.min(drag_vals):.4f}  max={np.max(drag_vals):.4f}"
    )
    print(
        f"  Cd (pressure):   mean={np.mean(cd_vals):.6f}  std={np.std(cd_vals):.6f}  "
        f"min={np.min(cd_vals):.6f}  max={np.max(cd_vals):.6f}"
    )
    gt_forces = [r["gt_force"] for r in results if not np.isnan(r["gt_force"])]
    gt_cds = [r["gt_cd"] for r in results if not np.isnan(r["gt_cd"])]
    if gt_forces:
        pred_forces = [r["drag_pressure"] for r, _ in zip(results, gt_forces, strict=False)]
        errors = [abs(p - g) for p, g in zip(pred_forces, gt_forces, strict=False)]
        rel_errors = [abs(p - g) / max(abs(g), SMALL_DIFF) for p, g in zip(pred_forces, gt_forces, strict=False)]
        print(f"\n  vs GT force ({len(gt_forces)} samples):")
        print(f"    MAE  = {np.mean(errors):.4f} N")
        print(f"    MAPE = {np.mean(rel_errors) * 100:.2f}%")
    if gt_cds:
        pred_cds = [r["cd_pressure"] for r, _ in zip(results, gt_cds, strict=False)]
        cd_errors = [abs(p - g) for p, g in zip(pred_cds, gt_cds, strict=False)]
        cd_rel = [abs(p - g) / max(abs(g), SMALL_DIFF) for p, g in zip(pred_cds, gt_cds, strict=False)]
        print(f"\n  vs GT Cd ({len(gt_cds)} samples):")
        print(f"    MAE  = {np.mean(cd_errors):.6f}")
        print(f"    MAPE = {np.mean(cd_rel) * 100:.2f}%")
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "drag_pressure",
                    "drag_total",
                    "cd_pressure",
                    "cd_total",
                    "gt_force",
                    "gt_cd",
                    "n_vertices",
                    "total_area",
                    "a_ref",
                    "v_inf",
                ],
            )
            w.writeheader()
            w.writerows(results)
        print(f"\n  Results saved to: {args.output_csv}")


# ---------------------------------------------------------------------------
#  Mode 3: Model inference + integration
# ---------------------------------------------------------------------------
def integrate_from_model(data_dir, args):  # noqa: PLR0915
    """Run GAOT inference on a dataset, then integrate the predicted field."""

    # Load config
    cfg = OmegaConf.load(args.config)
    model_cfg = cfg.model

    # Determine input channels
    magno_cfg = model_cfg.args.magno
    encoder_attr = (
        magno_cfg.get("encoder_feature_attr", "x")
        if isinstance(magno_cfg, dict)
        else getattr(magno_cfg, "encoder_feature_attr", "x")
    )
    input_size = sum(3 for _ in encoder_attr) if isinstance(encoder_attr, list) else 3

    # Create model using dict config
    model_args_raw = OmegaConf.to_container(model_cfg.args, resolve=True)
    model_args: dict = dict(model_args_raw) if isinstance(model_args_raw, dict) else {}
    model_args["input_size"] = input_size
    model_args["output_size"] = 2  # force + cd scalar heads
    field_output_size = model_args.get("field_output_size", 0)

    model = GAOT3D(config=model_args, field_output_size=field_output_size)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # Handle DDP checkpoint (keys may have 'module.' prefix)
    state_dict = ckpt if isinstance(ckpt, dict) and "model_state_dict" not in ckpt else ckpt.get("model_state_dict", ckpt)
    if any(k.startswith("module.") for k in state_dict):
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
    pt_files = _find_pt_files(data_dir)
    if not pt_files:
        print(f"No .pt sample files found in {data_dir}")
        return
    print(f"  Running inference on {len(pt_files)} samples...")

    results = []
    latent_tokens = model.latent_tokens.to(device)

    with torch.no_grad():
        for pt_path in pt_files:
            data = torch.load(pt_path, weights_only=False)
            # Name: prefer data.filename, else parent dir name (layout 1) or stem (layout 2)
            if hasattr(data, "filename") and data.filename:
                name = data.filename
            elif os.path.basename(pt_path) == "data.pt":
                name = os.path.basename(os.path.dirname(pt_path))
            else:
                name = os.path.splitext(os.path.basename(pt_path))[0]

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
                    rho=args.rho,
                    u_inf=args.u_inf,
                    a_ref=args.a_ref,
                    drag_axis=args.drag_axis,
                    use_shear=args.use_shear,
                    pressure_key="x",
                    is_cp=args.is_cp,
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
    _print_model_integration_summary(results, args)

    return results


def _print_model_integration_summary(results: list, args) -> None:
    """Print summary statistics and optionally save CSV for model-integration results."""
    if not results:
        return
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


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Integrate GAOT pressure field predictions into drag force.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Integrate ground truth Cp field from a single .pt file (auto v_inf & a_ref)
  python integrate_pressure_field.py /path/to/euler_run_1/data.pt --is_cp

  # Integrate all samples in a dataset directory (Cp field, auto v_inf & a_ref)
  python integrate_pressure_field.py /path/to/processed_euler_cp_full/ --is_cp

  # Override v_inf/a_ref for the whole dataset (optional)
  python integrate_pressure_field.py /path/to/processed_euler_cp_full/ --is_cp \\
      --u_inf 25.0 --a_ref 0.35

  # DrivAer dataset
  python integrate_pressure_field.py /path/to/processed_drivaer_cp_full/ --is_cp

  # Blended wing (flat *.pt layout, variable v_inf per sample)
  python integrate_pressure_field.py /path/to/processed_blended_wing_cp_full/ --is_cp

  # Include friction drag from wall shear
  python integrate_pressure_field.py /path/to/dataset/ --is_cp --use_shear

  # Run GAOT inference + integration from a checkpoint
  python integrate_pressure_field.py /path/to/processed_dir/ \\
      --ckpt /path/to/field_model.pt --config config/pressure_field.yaml --is_cp

  # Save results to CSV
  python integrate_pressure_field.py /path/to/dataset/ --is_cp --output_csv results.csv
        """,
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to a .pt file or a dataset directory. Supports both subdirectory layout (*/data.pt) and flat layout (*.pt).",
    )
    parser.add_argument("--rho", type=float, default=DEFAULT_RHO, help=f"Freestream density [kg/m³] (default: {DEFAULT_RHO})")
    parser.add_argument(
        "--u_inf",
        type=float,
        default=DEFAULT_U_INF,
        help="Freestream velocity [m/s]. If omitted, read from each sample's v_inf attribute.",
    )
    parser.add_argument(
        "--a_ref",
        type=float,
        default=DEFAULT_A_REF,
        help="Reference area [m²]. If omitted, computed as frontal area from each sample's mesh.",
    )
    parser.add_argument(
        "--drag_axis", type=int, default=DRAG_AXIS, choices=[0, 1, 2], help="Drag direction axis: 0=x, 1=y, 2=z (default: 0)"
    )
    parser.add_argument("--use_shear", action="store_true", help="Include wall shear stress (friction drag) if available")
    parser.add_argument("--pressure_key", type=str, default="x", help="Data attribute name for pressure field (default: 'x')")
    parser.add_argument(
        "--is_cp",
        action="store_true",
        help="Treat the pressure field as Cp (pressure coefficient). "
        "It will be converted to Pascals via p = Cp * 0.5 * rho * u_inf^2 before integration.",
    )
    parser.add_argument("--output_csv", type=str, default=None, help="Save per-sample results to this CSV file")

    # Model inference mode
    parser.add_argument("--ckpt", type=str, default=None, help="GAOT checkpoint path — enables inference mode")
    parser.add_argument("--config", type=str, default=None, help="Config YAML for model architecture (required with --ckpt)")

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
