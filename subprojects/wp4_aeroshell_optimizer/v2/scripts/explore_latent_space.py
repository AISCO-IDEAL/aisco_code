"""Latent space validity explorer for pretrained ConcatWIREV5 checkpoints.

Runs three complementary exploration modes on a saved checkpoint and saves
PLY meshes + per-point SDF statistics to an output directory.

Modes
-----
1  circle  — 24 equally-spaced samples around the pretrained circle (r = z_radius).
             Shows whether the model is valid along the trained manifold.
2  radial  — For each pretrained z direction, sweep r ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0}.
             Shows how far the model can be extrapolated in/out from the trained circle.
3  interp  — Straight-line interpolation between every adjacent pair of pretrained z vectors.
             Shows whether the model is valid between trained anchor points.

Usage
-----
    python3 explore_latent_space.py \\
        --checkpoint /cluster/work/fuge/.../pretrained_latent_b08f8ced4e.pth \\
        --out /cluster/work/fuge/AISCO/latent_exploration/b08f8ced4e \\
        [--modes 123] [--n_circle 24] [--n_interp 7] [--resolution 64] [--device cpu]

Outputs
-------
    {out}/mode1_circle/           — PLYs named circle_NN_z=[z0,z1].ply
    {out}/mode2_radial/           — PLYs named slot_N_r=R.ply
    {out}/mode3_interp/           — PLYs named pair_N-M_step_K.ply
    {out}/sdf_stats.csv           — per-sample SDF min/max/neg_frac/mesh_valid
    {out}/summary.txt             — human-readable summary printed to stdout and file
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from skimage.measure import marching_cubes
from torch import Tensor

# ---------------------------------------------------------------------------
# Model reconstruction
# ---------------------------------------------------------------------------

def _build_model_from_info(info: dict[str, Any], device: torch.device) -> Any:
    """Reconstruct a ConcatWIREV5 from the model_info_dict.json sidecar.

    Uses importlib.util.spec_from_file_location with sys.path manipulation to
    load ConcatWIREV5 without triggering the wp4 package __init__.py which
    would pull in the fixed_renderer CUDA extension unavailable on CPU nodes.
    """
    import importlib.util  # noqa: PLC0415
    import types  # noqa: PLC0415

    root = str(Path(__file__).parents[4])  # AISCO_code root
    if root not in sys.path:
        sys.path.insert(0, root)

    # Stub out basic_MLP_v1 to prevent the wp4 __init__.py import chain from
    # reaching VisualisationManager → fixed_renderer CUDA (unavailable on CPU nodes).
    # The stub is a no-op module; concat_WIRE_v5 does not need BasicMLP.
    stub_key = "subprojects.wp4_aeroshell_optimizer.v2.src.models.basic_MLP_v1"
    if stub_key not in sys.modules:
        stub = types.ModuleType(stub_key)
        stub.BasicMLP = None  # type: ignore[attr-defined]
        sys.modules[stub_key] = stub

    src_file = Path(__file__).parents[1] / "src" / "models" / "concat_WIRE_v5.py"
    spec = importlib.util.spec_from_file_location(
        "subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5",
        src_file,
    )
    assert spec is not None and spec.loader is not None, f"Cannot find model source: {src_file}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    ConcatWIREV5 = mod.ConcatWIREV5

    return ConcatWIREV5(
        z_dim=info["z_dim"],
        hidden_features=info["hidden_features"],
        hidden_layers=info["hidden_layers"],
        coords=info.get("coords", 3),
        out_features=info.get("out_features", 1),
        first_omega_min=info["first_omega_min"],
        first_omega_max=info["first_omega_max"],
        first_scale_min=info["first_scale_min"],
        first_scale_max=info["first_scale_max"],
        hidden_omega_min=info["hidden_omega_min"],
        hidden_omega_max=info["hidden_omega_max"],
        hidden_scale_min=info["hidden_scale_min"],
        hidden_scale_max=info["hidden_scale_max"],
        skip_layer_idx=info.get("skip_layer_idx", 2),
        train_omega=info.get("trainable_omega", False),
        train_scale=info.get("trainable_scale", False),
        use_spectral_norm=info.get("use_spectral_norm", False),
        use_smooth_output=info.get("use_smooth_output", False),
        bbox_center=info.get("bbox_center"),
        bbox_half_extent=float(info.get("bbox_half_extent", 1000.0)),
    ).to(device)


# ---------------------------------------------------------------------------
# SDF statistics helper
# ---------------------------------------------------------------------------

def _sdf_stats(model: Any, z_vec: Tensor, prism: dict[str, list[float]], device: torch.device, n_pts: int = 8000) -> dict[str, float]:
    """Return SDF min/max/neg_frac over a uniform prism grid for the given z vector."""
    lo = torch.tensor([prism["x"][0], prism["y"][0], prism["z"][0]], device=device)
    hi = torch.tensor([prism["x"][1], prism["y"][1], prism["z"][1]], device=device)
    pts = lo + torch.rand(n_pts, 3, device=device) * (hi - lo)
    model.set_latent_vector(z_vec)
    with torch.no_grad():
        sdf = model(pts).squeeze()
    mn = float(sdf.min().item())
    mx = float(sdf.max().item())
    neg_frac = float((sdf < 0).float().mean().item())
    return {"sdf_min": mn, "sdf_max": mx, "neg_frac": neg_frac}


# ---------------------------------------------------------------------------
# Mesh extraction + PLY save
# ---------------------------------------------------------------------------

def _extract_and_save(
    model: Any,
    z_vec: Tensor,
    ply_path: Path,
    prism: dict[str, list[float]],
    resolution: int,
    device: torch.device,
) -> bool:
    """Extract mesh at z_vec via direct grid evaluation + marching cubes.

    Avoids MeshSDFExtractor whose internal param-version cache returns the same
    mesh for every call when only _active_z changes (not model weights).
    Returns True if the mesh was non-empty.
    """
    model.set_latent_vector(z_vec)

    # Build a uniform grid over the prism bbox.
    xs = torch.linspace(prism["x"][0], prism["x"][1], resolution, device=device)
    ys = torch.linspace(prism["y"][0], prism["y"][1], resolution, device=device)
    zs = torch.linspace(prism["z"][0], prism["z"][1], resolution, device=device)
    # grid shape: (res, res, res, 3)
    gx, gy, gz = torch.meshgrid(xs, ys, zs, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)

    with torch.no_grad():
        sdf_flat = model(pts).squeeze()  # (res^3,)

    sdf_vol = sdf_flat.cpu().numpy().reshape(resolution, resolution, resolution)

    # Marching cubes requires at least one sign change.
    if sdf_vol.min() >= 0.0 or sdf_vol.max() <= 0.0:
        return False

    try:
        verts_idx, faces_idx, _, _ = marching_cubes(sdf_vol, level=0.0)
    except Exception as exc:
        print(f"    [WARN] marching_cubes failed: {exc}")
        return False

    if len(verts_idx) == 0:
        return False

    # Map voxel indices back to world coordinates.
    lo = np.array([prism["x"][0], prism["y"][0], prism["z"][0]], dtype=np.float32)
    hi = np.array([prism["x"][1], prism["y"][1], prism["z"][1]], dtype=np.float32)
    verts_world = lo + verts_idx / (resolution - 1) * (hi - lo)

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=verts_world, faces=faces_idx)
    mesh.export(str(ply_path))
    return True


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------

def _mode1_circle(
    model: Any,
    z_dim: int,
    z_radius: float,
    n_samples: int,
    prism: dict[str, list[float]],
    resolution: int,
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Mode 1: sample N equally-spaced points on the pretrained circle."""
    records: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Mode 1] Circle scan: {n_samples} pts, r={z_radius}")

    for idx in range(n_samples):
        angle = 2.0 * math.pi * idx / n_samples
        z0 = z_radius * math.cos(angle)
        z1 = z_radius * math.sin(angle)
        z_vec = torch.zeros(z_dim, device=device)
        z_vec[0] = z0
        z_vec[1] = z1

        stats = _sdf_stats(model, z_vec, prism, device)
        ply_name = f"circle_{idx:02d}_z=[{z0:.3f},{z1:.3f}].ply"
        valid = _extract_and_save(model, z_vec, out_dir / ply_name, prism, resolution, device)

        row = {"mode": "circle", "label": f"circle_{idx:02d}", "z0": round(z0, 4), "z1": round(z1, 4),
               "r": z_radius, "angle_deg": round(math.degrees(angle), 1), "mesh_valid": valid, **stats}
        records.append(row)
        status = "OK" if valid else "NO MESH"
        print(f"  [{idx+1:2d}/{n_samples}] angle={math.degrees(angle):6.1f}°  "
              f"sdf=[{stats['sdf_min']:+8.1f}, {stats['sdf_max']:+8.1f}]  "
              f"neg={stats['neg_frac']:.3f}  {status}")

    return records


def _mode2_radial(
    model: Any,
    Z_pretrained: Tensor,
    shape_names: list[str],
    radii: list[float],
    prism: dict[str, list[float]],
    resolution: int,
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Mode 2: sweep each pretrained z direction across multiple radii."""
    records: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    n_slots = Z_pretrained.shape[0]
    print(f"\n[Mode 2] Radial sweep: {n_slots} directions x {len(radii)} radii")

    for slot_idx in range(n_slots):
        z_dir = Z_pretrained[slot_idx]
        r_pretrained = float(torch.norm(z_dir).item())
        # Unit direction vector (handle z=[0,0] at origin)
        z_unit = z_dir / r_pretrained if r_pretrained > 1e-6 else torch.zeros_like(z_dir)

        name = shape_names[slot_idx] if slot_idx < len(shape_names) else f"slot{slot_idx:02d}"
        print(f"\n  Slot {slot_idx} ({name})  pretrained r={r_pretrained:.3f}  "
              f"dir=[{z_dir[0].item():.3f}, {z_dir[1].item():.3f}]")

        for r in radii:
            z_vec = (z_unit * r).to(device)
            stats = _sdf_stats(model, z_vec, prism, device)
            ply_name = f"slot{slot_idx:02d}_{name}_r={r:.2f}.ply"
            valid = _extract_and_save(model, z_vec, out_dir / ply_name, prism, resolution, device)

            marker = " ← PRETRAINED" if abs(r - r_pretrained) < 0.01 else ""
            status = "OK" if valid else "NO MESH"
            print(f"    r={r:.2f}  sdf=[{stats['sdf_min']:+8.1f}, {stats['sdf_max']:+8.1f}]  "
                  f"neg={stats['neg_frac']:.3f}  {status}{marker}")

            row = {"mode": "radial", "label": f"slot{slot_idx:02d}_r={r:.2f}",
                   "slot": slot_idx, "shape": name, "r": r, "r_pretrained": round(r_pretrained, 4),
                   "z0": round(z_vec[0].item(), 4), "z1": round(z_vec[1].item(), 4),
                   "mesh_valid": valid, **stats}
            records.append(row)

    return records


def _mode3_interp(
    model: Any,
    Z_pretrained: Tensor,
    shape_names: list[str],
    n_steps: int,
    prism: dict[str, list[float]],
    resolution: int,
    device: torch.device,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """Mode 3: straight-line interpolation between every adjacent pretrained pair."""
    records: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    n_slots = Z_pretrained.shape[0]
    print(f"\n[Mode 3] Interpolation paths: {n_slots} adjacent pairs x {n_steps} steps")

    # Walk the circle in order: slot 1..N-1 adjacent pairs (slot 0 = origin, skip or include)
    # Pair all consecutive slots 0->1, 1->2, ..., N-2->N-1, N-1->1 (wrap circle from slot 1 to slot N-1)
    pairs: list[tuple[int, int]] = [(i, i + 1) for i in range(n_slots - 1)]
    if n_slots > 2:
        pairs.append((n_slots - 1, 1))  # close the circle between last and first circle slot

    for (i, j) in pairs:
        z_a = Z_pretrained[i].to(device)
        z_b = Z_pretrained[j].to(device)
        name_a = shape_names[i] if i < len(shape_names) else f"slot{i:02d}"
        name_b = shape_names[j] if j < len(shape_names) else f"slot{j:02d}"
        print(f"\n  Pair {i}→{j} ({name_a} → {name_b})")

        for step in range(n_steps):
            t = step / (n_steps - 1) if n_steps > 1 else 0.0
            z_vec = z_a * (1.0 - t) + z_b * t

            stats = _sdf_stats(model, z_vec, prism, device)
            ply_name = f"pair_{i:02d}-{j:02d}_step{step:02d}_t={t:.2f}.ply"
            valid = _extract_and_save(model, z_vec, out_dir / ply_name, prism, resolution, device)

            status = "OK" if valid else "NO MESH"
            print(f"    t={t:.2f}  z=[{z_vec[0].item():+.3f}, {z_vec[1].item():+.3f}]  "
                  f"sdf=[{stats['sdf_min']:+8.1f}, {stats['sdf_max']:+8.1f}]  "
                  f"neg={stats['neg_frac']:.3f}  {status}")

            row = {"mode": "interp", "label": f"pair{i:02d}-{j:02d}_step{step:02d}",
                   "slot_a": i, "slot_b": j, "shape_a": name_a, "shape_b": name_b,
                   "t": round(t, 4), "z0": round(z_vec[0].item(), 4), "z1": round(z_vec[1].item(), 4),
                   "mesh_valid": valid, **stats}
            records.append(row)

    return records


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _write_summary(records: list[dict[str, Any]], out_dir: Path, checkpoint_path: Path) -> None:
    """Write CSV and text summary of all exploration results."""
    # CSV — union fieldnames across all records so mode-specific fields (slot,
    # r_pretrained, shape, …) don't cause a DictWriter key error.
    csv_path = out_dir / "sdf_stats.csv"
    if records:
        all_keys: list[str] = []
        seen: set[str] = set()
        for rec in records:
            for k in rec:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            writer.writeheader()
            writer.writerows(records)
        print(f"\nCSV written: {csv_path}")

    # Text summary
    lines = [
        f"Latent space exploration — {checkpoint_path.name}",
        f"Output: {out_dir}",
        "",
        "=== SDF validity summary ===",
    ]
    for mode_tag in ("circle", "radial", "interp"):
        mode_records = [r for r in records if r["mode"] == mode_tag]
        if not mode_records:
            continue
        valid = sum(1 for r in mode_records if r["mesh_valid"])
        total = len(mode_records)
        min_sdf = min(r["sdf_min"] for r in mode_records)
        pct = valid / total * 100 if total else 0
        lines.append(f"  Mode {'1:circle' if mode_tag=='circle' else '2:radial' if mode_tag=='radial' else '3:interp':10s}  "
                     f"mesh_valid={valid}/{total} ({pct:.0f}%)  global_sdf_min={min_sdf:+.1f}")

    # Identify broken radial entries
    radial = [r for r in records if r["mode"] == "radial"]
    if radial:
        lines.append("")
        lines.append("=== Radial: first invalid radius per slot ===")
        slots_seen: dict[int, list[dict[str, Any]]] = {}
        for r in radial:
            slots_seen.setdefault(r["slot"], []).append(r)
        for slot_idx, slot_records in sorted(slots_seen.items()):
            first_invalid = next((r for r in slot_records if not r["mesh_valid"]), None)
            r_pretrained = slot_records[0]["r_pretrained"]
            if first_invalid:
                lines.append(f"  Slot {slot_idx} ({slot_records[0]['shape']:30s})  "
                              f"first_invalid_r={first_invalid['r']:.2f}  pretrained_r={r_pretrained:.2f}")
            else:
                lines.append(f"  Slot {slot_idx} ({slot_records[0]['shape']:30s})  "
                              f"all radii valid  pretrained_r={r_pretrained:.2f}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    (out_dir / "summary.txt").write_text(summary_text + "\n")


# ---------------------------------------------------------------------------
# Checkpoint loading helper (extracted to keep main() under statement limit)
# ---------------------------------------------------------------------------

def _load_checkpoint(
    ckpt_path: Path,
    device: torch.device,
) -> tuple[Any, Tensor, int, float, list[str], dict[str, list[float]]]:
    """Load model, pretrained Z vectors, shape names, and prism bounds from a checkpoint.

    Args:
        ckpt_path: Path to pretrained_latent_*.pth file.
        device: Target compute device.

    Returns:
        Tuple of (model, Z_pretrained, z_dim, z_radius, shape_names, prism).
    """
    info_path = ckpt_path.parent / "model_info_dict.json"
    if not info_path.exists():
        print(f"ERROR: model_info_dict.json not found next to checkpoint: {info_path}", file=sys.stderr)
        sys.exit(1)

    with open(info_path) as f:
        model_info = json.load(f)

    print(f"Building {model_info['name']}  z_dim={model_info['z_dim']}  "
          f"hidden={model_info['hidden_features']}x{model_info['hidden_layers']}")
    model = _build_model_from_info(model_info, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    Z_pretrained: Tensor = ckpt["z_pretrained"].to(device)
    n_slots, z_dim = Z_pretrained.shape
    print(f"Loaded {n_slots} pretrained z vectors (z_dim={z_dim})")
    for i, z in enumerate(Z_pretrained):
        r = float(torch.norm(z).item())
        print(f"  slot {i}: z=[{z[0].item():+.4f}, {z[1].item():+.4f}]  r={r:.4f}")

    z_radii = [float(torch.norm(Z_pretrained[i]).item()) for i in range(n_slots)]
    circle_radii = [r for r in z_radii if r > 0.1]
    z_radius = float(sum(circle_radii) / len(circle_radii)) if circle_radii else 2.0
    print(f"Inferred z_radius = {z_radius:.4f}")

    shapes_dir = ckpt_path.parent / "pretrained_shapes"
    shape_names: list[str] = []
    if shapes_dir.exists():
        for ply in sorted(shapes_dir.glob("slot*.ply")):
            name_part = ply.stem
            if "_" in name_part:
                name_part = name_part.split("_", 1)[1]
            shape_names.append(name_part)
    if not shape_names:
        shape_names = [f"slot{i:02d}" for i in range(n_slots)]
    print(f"Shape names: {shape_names}")

    from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm  # noqa: PLC0415
    prism_raw = get_outer_prism_bounds_mm()
    prism = {"x": list(prism_raw["x"]), "y": list(prism_raw["y"]), "z": list(prism_raw["z"])}
    print(f"Outer prism: x={prism['x']}, y={prism['y']}, z={prism['z']}")

    return model, Z_pretrained, z_dim, z_radius, shape_names, prism


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Latent space validity explorer")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_latent_*.pth")
    parser.add_argument("--out", default=None,
                        help="Output directory for PLYs and stats. Defaults to "
                             "<checkpoint_dir>/latent_space_exploration_<hash> where "
                             "<hash> is the suffix of the checkpoint filename.")
    parser.add_argument("--modes", default="123", help="Modes to run (e.g. '12' or '1' or '123')")
    parser.add_argument("--n_circle", type=int, default=24, help="Samples on the pretrained circle (Mode 1)")
    parser.add_argument("--radii", default="0.0,0.5,1.0,1.5,2.0,2.5,3.0",
                        help="Comma-separated radii for Mode 2")
    parser.add_argument("--n_interp", type=int, default=7, help="Steps per interpolation path (Mode 3)")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Marching cubes grid resolution (lower = faster)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Derive output dir from checkpoint if not specified.
    # Convention: pretrained_latent_<hash>.pth → <ckpt_dir>/latent_space_exploration_<hash>
    if args.out is not None:
        out_dir = Path(args.out)
    else:
        stem = ckpt_path.stem  # e.g. "pretrained_latent_b08f8ced4e"
        prefix = "pretrained_latent_"
        ckpt_hash = stem.removeprefix(prefix)
        out_dir = ckpt_path.parent / f"latent_space_exploration_{ckpt_hash}"
        print(f"[explore] Auto-derived output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, Z_pretrained, z_dim, z_radius, shape_names, prism = _load_checkpoint(ckpt_path, device)

    radii = [float(r) for r in args.radii.split(",")]
    all_records: list[dict[str, Any]] = []

    if "1" in args.modes:
        records = _mode1_circle(
            model=model, z_dim=z_dim, z_radius=z_radius,
            n_samples=args.n_circle, prism=prism, resolution=args.resolution,
            device=device, out_dir=out_dir / "mode1_circle",
        )
        all_records.extend(records)

    if "2" in args.modes:
        records = _mode2_radial(
            model=model, Z_pretrained=Z_pretrained, shape_names=shape_names,
            radii=radii, prism=prism, resolution=args.resolution,
            device=device, out_dir=out_dir / "mode2_radial",
        )
        all_records.extend(records)

    if "3" in args.modes:
        records = _mode3_interp(
            model=model, Z_pretrained=Z_pretrained, shape_names=shape_names,
            n_steps=args.n_interp, prism=prism, resolution=args.resolution,
            device=device, out_dir=out_dir / "mode3_interp",
        )
        all_records.extend(records)

    _write_summary(all_records, out_dir, ckpt_path)


if __name__ == "__main__":
    main()
