#!/usr/bin/env python
"""Single-shape pretraining (Plan 014, v0.1.0).

Fits a non-conditional SDF model (no latent z) to ONE target shape using the
same sampling / loss machinery as ``pretraining_latent.py``.

Design notes (vs the legacy single-shape pipeline this replaced):
  * Surface-concentrated sampling (``_sample_points_for_shape``) instead of a
    50/50 uniform/jittered split with a hardcoded σ = 5 % of bbox.  The
    default near-surface σ is 0.5 – 2 % of the bbox diagonal, which is what
    actually pins the surface during training.
  * **L1-all** loss on (surface ∪ domain) pts as the mandatory term,
    replacing ``MSE_surface + eikonal_weight * eikonal``.  L1-all directly
    forces both correct surface location and correct far-field SDF magnitude
    — the eikonal regulariser becomes redundant and the ellipsoid attractor
    disappears.
  * Optional IGR-style surface-normal supervision and a light off-surface
    eikonal regulariser are layered on top (see
    ``pretraining_single_config.yaml`` for conceptual notes).
  * DeepSDF-style static surface cache (sampled once, reused every epoch).
    Optional refresh via ``surface_resample_cadence``.
  * Hash-tagged checkpoint folder so different sweep configs land in distinct
    directories (no overwrites between agents).
  * Marching-cubes extraction uses the asymmetric outer_prism bounding box
    (the actual sampling region) rather than a hardcoded ±N mm cube.

Sweep-compatible: standard ``register_all_configs`` / ``apply_sweep_overrides``
pattern, so sweep YAML keys of the form
``pretraining_single_config.training_loop.X`` and
``models/AISCOModel_v0.X`` work directly.

Usage (smoke test — 5 epochs, no mesh extraction):
    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_single.py --epochs 5 --no-mesh

Usage (full run via sweep agent):
    wandb agent <sweep_id>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb
from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm, scale_outer_prism
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
    _load_or_build_surface_cache,
    _load_ply_tensors,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _apply_override_config,
    _extract_mesh,
    _sample_points_for_shape,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.device_detection import detect_environment
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing


def _build_shape_params(shape: dict[str, Any], ply_tensors: tuple[Tensor, Tensor, float] | None) -> dict[str, Any]:
    """Translate a shape dict from the config into the kwargs expected by
    ``_sample_points_for_shape``.

    Mirrors ``pretraining_latent._build_shape_params`` but for a single shape
    (no slot bookkeeping).
    """
    shape_type: str = shape["shape_type"]
    if shape_type == "sphere":
        return {"radius": float(shape["radius"])}
    if shape_type == "cube":
        return {"side_length": float(shape["side_length"])}
    if shape_type == "ply":
        assert ply_tensors is not None, "ply_tensors must be provided when shape_type == 'ply'"
        ply_verts, ply_faces, bbox_extent = ply_tensors
        return {
            "ply_verts": ply_verts,
            "ply_faces": ply_faces,
            "bbox_extent": bbox_extent,
            "surface_noise_sigma_factor": float(shape.get("surface_noise_sigma_factor", 0.05)),
        }
    raise ValueError(f"Unknown shape_type '{shape_type}'. Valid values: 'sphere', 'cube', 'ply'.")


def _sample_surface_points_with_normals(ply_verts: Tensor, ply_faces: Tensor, n_pts: int, device: str) -> tuple[Tensor, Tensor]:
    """Sample exact on-surface points + per-point unit normals from a PLY mesh.

    Faces are sampled with probability proportional to their area (so dense /
    fine-detail regions are not undersampled), then each face is sampled with
    a uniform barycentric coordinate.  The returned normal for each point is
    the unit face normal (per-vertex normals would average across creases and
    blur the supervision signal we explicitly want to inject — see the docstring
    of ``main`` for why we add this loss term).

    Args:
        ply_verts: (V, 3) float tensor of mesh vertices on CPU (already in mm,
            in AISCO axis convention, as returned by ``_load_ply_tensors``).
        ply_faces: (F, 3) int64 tensor of triangle vertex indices (CPU).
        n_pts: Number of on-surface points to draw.
        device: Target device for the returned tensors.

    Returns:
        Tuple ``(pts, normals)`` of shape ``(n_pts, 3)`` each, on ``device``.
        ``normals`` are unit-length.
    """
    verts = ply_verts.to(torch.float32)
    faces = ply_faces.to(torch.long)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals_raw = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_areas = 0.5 * torch.linalg.norm(face_normals_raw, dim=-1)
    # Guard against zero-area faces (degenerate triangles in the input PLY).
    face_areas = torch.clamp(face_areas, min=1e-12)
    face_normals = face_normals_raw / (2.0 * face_areas.unsqueeze(-1))

    face_idx = torch.multinomial(face_areas, n_pts, replacement=True)
    # Uniform barycentric sampling on the unit triangle: (1-√r1, √r1·(1-r2), √r1·r2).
    r1 = torch.rand(n_pts)
    r2 = torch.rand(n_pts)
    sqrt_r1 = torch.sqrt(r1)
    b0 = (1.0 - sqrt_r1).unsqueeze(-1)
    b1 = (sqrt_r1 * (1.0 - r2)).unsqueeze(-1)
    b2 = (sqrt_r1 * r2).unsqueeze(-1)
    pts = b0 * v0[face_idx] + b1 * v1[face_idx] + b2 * v2[face_idx]
    normals = face_normals[face_idx]
    return pts.to(device), normals.to(device)


def _grad_of_sdf(model: SupportsModelContract, pts: Tensor) -> Tensor:
    """Return ``∇f(x)`` for an SDF network ``f``.

    Standard IGR-style trick: enable ``requires_grad`` on the inputs and
    backprop a sum-of-outputs to recover the per-point gradient in a single
    autograd call.  ``create_graph=True`` so the gradient itself is
    differentiable (needed because the eikonal / normal losses depend on it).
    """
    pts = pts.detach().requires_grad_(True)
    pred = model(pts).squeeze(-1)
    grad = torch.autograd.grad(
        outputs=pred.sum(),
        inputs=pts,
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad


def _compute_pretrain_config_hash(config: dict[str, Any], model: SupportsModelContract) -> str:
    """Stable 10-char hash over model architecture + sampling/training params.

    Two runs with identical parameters produce the same hash; any meaningful
    parameter change yields a new hash.  Used to give each run its own output
    folder and (for PLY) its own SDF disk-cache key.
    """
    key: dict[str, Any] = {
        "model": dict_hashing(model.get_model_info_dict()),
        "shape": config["shape"],
        "training_loop": config["training_loop"],
    }
    return dict_hashing(key)


def _resolve_checkpoint_path(config: dict[str, Any], model: SupportsModelContract) -> Path:
    """Return ``<base_dir>/<hash>/<basename>_<hash>.pth``.

    Mirrors ``pretraining_latent._resolve_checkpoint_path`` so single and latent
    runs share the same on-disk layout convention.
    """
    base = Path(config["output"]["checkpoint_path"])
    config_hash = _compute_pretrain_config_hash(config, model)
    return base.parent / config_hash / f"{base.stem}_{config_hash}{base.suffix}"


def _init_wandb_and_apply_sweep(active_configs: dict[str, str]) -> bool:
    """Register configs, init wandb, and apply any sweep overrides.

    Order is critical: ``register_all_configs`` must run BEFORE
    ``apply_sweep_overrides`` so that sweep keys like
    ``models/AISCOModel_v0.activations`` resolve to a registered config.

    Returns:
        ``True`` when wandb is enabled (and a run is active), else ``False``.
    """
    register_all_configs({"active_configs": active_configs}, v="v2")

    pt_cfg = load_config(v="v2", cfg_name="pretraining_single_config.yaml")
    use_wandb: bool = pt_cfg.get("wandb", {}).get("use_wandb", False)

    if use_wandb:
        wandb.init(
            entity=pt_cfg["wandb"]["entity_name"],
            project=pt_cfg["wandb"]["project_name"],
        )

    if wandb.run is not None:
        apply_sweep_overrides(dict(wandb.config))
        # Re-publish the resolved (post-override) master config to W&B so the
        # run page shows the actual values that were used.
        wandb.config.update(get_master_config(), allow_val_change=True)

    return use_wandb and wandb.run is not None


def _build_active_configs() -> dict[str, str]:
    """Build the active_configs map for register_all_configs.

    Includes pretraining_single + every model config listed in models.json so
    that ``models/<model_yaml>.X`` sweep keys are resolvable.
    """
    environment = detect_environment()
    if environment == "aisco_beast":
        models_json_path = (
            Path.home()
            / "Documents"
            / "AISCO_code"
            / "subprojects"
            / "wp4_aeroshell_optimizer"
            / "v2"
            / "configs"
            / "models.json"
        )
    elif environment == "wsl":
        # note from eschaerer: the path must be updated, this is just a placeholder
        models_json_path = Path.home() / "note from eschaerer: the path must be updated, this is just a placeholder"
    else:
        models_json_path = (
            Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / "v2" / "configs" / "models.json"
        )
    models_inventory = json.loads(models_json_path.read_text())
    active: dict[str, str] = {"pretraining_single": "pretraining_single_config.yaml"}
    for model_info in models_inventory["models"].values():
        cfg_file: str | None = model_info.get("config_file")
        if cfg_file:
            active[Path(cfg_file).stem] = cfg_file
    return active


def main(epochs_override: int | None = None, no_mesh: bool = False) -> None:  # noqa: PLR0912, PLR0915 — training loop is intrinsically sequential
    """Run single-shape pretraining: sample, fit, save checkpoint, optionally extract mesh.

    Args:
        epochs_override: When provided, overrides ``training_loop.epochs`` from
            the YAML (useful for smoke tests, e.g. ``--epochs 5``).
        no_mesh: When True, skip the post-training marching-cubes extraction.
    """
    # ── Register configs + wandb init + sweep overrides ──────────────────────
    use_wandb = _init_wandb_and_apply_sweep(_build_active_configs())

    # ── Load (post-override) config and build model ─────────────────────────
    config = load_config(v="v2", cfg_name="pretraining_single_config.yaml")

    # CLI override: lets `--epochs 5` smoke-test without editing YAML.
    if epochs_override is not None:
        config = _apply_override_config(config, {"training_loop.epochs": epochs_override})

    torch.manual_seed(int(config.get("random_seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name: str = config["model_name"]
    model, _, _, _ = smart_model_init(model_name)
    model = model.to(device)
    print(f"[single] Loaded {model.__class__.__name__} (model_name={model_name!r}) on {device}")

    # ── Resolve hash-tagged checkpoint path & fast-resume ───────────────────
    ckpt_path = _resolve_checkpoint_path(config, model)
    config_hash = _compute_pretrain_config_hash(config, model)
    run_dir = ckpt_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    force: bool = bool(config.get("force_pretraining", True))
    if not force and ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"[single] Fast resume: loaded checkpoint from '{ckpt_path}'")
        if not no_mesh:
            outer_prism_resume = scale_outer_prism(
                get_outer_prism_bounds_mm(), float(config["output"].get("bbox_scale_factor", 1.0))
            )
            mesh_res_resume = int(config["output"].get("mesh_resolution", 256))
            model.eval()
            _extract_mesh(model, ckpt_path, device, mesh_res_resume, outer_prism_resume)
        return

    # ── Pre-load PLY (if any) — same axis transforms / watertight check as latent ──
    shape: dict[str, Any] = config["shape"]
    ply_tensors: tuple[Tensor, Tensor, float] | None = None
    if shape["shape_type"] == "ply":
        ply_path = shape["ply_path"]
        if not ply_path or not Path(ply_path).exists():
            raise FileNotFoundError(f"PLY mesh not found: {ply_path}")
        ply_tensors = _load_ply_tensors(ply_path, ply_scale=float(shape.get("ply_scale", 1.0)))

    shape_params = _build_shape_params(shape, ply_tensors)

    # ── Hyperparameters ─────────────────────────────────────────────────────
    loop_cfg: dict[str, Any] = config["training_loop"]
    epochs: int = int(loop_cfg["epochs"])
    lr: float = float(loop_cfg["learning_rate"])
    batch_size: int = int(loop_cfg["batch_size"])
    logging_interval: int = int(loop_cfg.get("logging_interval", 25))
    surface_resample_cadence: int = int(loop_cfg.get("surface_resample_cadence", 0))

    near_surface_fraction: float = float(loop_cfg.get("near_surface_fraction", 0.50))
    domain_fraction: float = float(loop_cfg.get("domain_fraction", 0.50))
    near_surface_tight_sigma: float | None = (
        float(loop_cfg["near_surface_tight_sigma"]) if loop_cfg.get("near_surface_tight_sigma") is not None else None
    )
    near_surface_wide_sigma: float | None = (
        float(loop_cfg["near_surface_wide_sigma"]) if loop_cfg.get("near_surface_wide_sigma") is not None else None
    )

    # Tier-1 architectural-loss knobs (Plan 015).
    normal_loss_weight: float = float(loop_cfg.get("normal_loss_weight", 0.0))
    normal_batch_size: int = int(loop_cfg.get("normal_batch_size", 512))
    normal_resample_cadence: int = int(loop_cfg.get("normal_resample_cadence", 50))
    eikonal_loss_weight: float = float(loop_cfg.get("eikonal_loss_weight", 0.0))
    eikonal_batch_size: int = int(loop_cfg.get("eikonal_batch_size", 1024))
    use_normal_loss = normal_loss_weight > 0.0 and shape["shape_type"] == "ply"
    use_eikonal_loss = eikonal_loss_weight > 0.0
    if normal_loss_weight > 0.0 and shape["shape_type"] != "ply":
        print(
            f"[single] WARNING: normal_loss_weight={normal_loss_weight} requested but shape_type "
            f"is '{shape['shape_type']}'; disabling (no mesh normals available).",
            file=sys.stderr,
        )

    if near_surface_fraction + domain_fraction > 1.0 + 1e-9:
        raise ValueError(
            f"near_surface_fraction + domain_fraction must sum to ≤ 1.0, "
            f"got {near_surface_fraction} + {domain_fraction} = {near_surface_fraction + domain_fraction}"
        )

    output_cfg: dict[str, Any] = config["output"]
    bbox_scale_factor = float(output_cfg.get("bbox_scale_factor", 1.0))
    outer_prism = scale_outer_prism(get_outer_prism_bounds_mm(), bbox_scale_factor)
    mesh_res: int = int(output_cfg.get("mesh_resolution", 256))
    sdf_cache_dir = Path(output_cfg["sdf_cache_dir"]) if output_cfg.get("sdf_cache_dir") else None

    n_dom: int = int(batch_size * domain_fraction)
    n_surf: int = batch_size - n_dom

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    l1_fn = torch.nn.L1Loss()

    # ── Build (or load from disk) initial surface cache ──────────────────────
    # PLY: cached on disk under sdf_cache_dir (if set) — keyed on PLY content
    # hash + sampling params, so identical sweeps across agents reuse the cache.
    # sphere/cube: cached only in memory (no disk cache).
    if shape["shape_type"] == "ply":
        surf_pts, surf_gt = _load_or_build_surface_cache(
            ply_path=shape["ply_path"],
            ply_scale=float(shape.get("ply_scale", 1.0)),
            shape_params=shape_params,
            batch_size=batch_size,
            n_dom=n_dom,
            device=device,
            cache_dir=sdf_cache_dir,
            near_surface_fraction=near_surface_fraction,
            domain_fraction=domain_fraction,
            near_surface_tight_sigma=near_surface_tight_sigma,
            near_surface_wide_sigma=near_surface_wide_sigma,
        )
    else:
        pts_init, gt_init = _sample_points_for_shape(
            shape["shape_type"],
            shape_params,
            batch_size,
            device,
            near_surface_fraction=near_surface_fraction,
            domain_fraction=domain_fraction,
            near_surface_tight_sigma=near_surface_tight_sigma,
            near_surface_wide_sigma=near_surface_wide_sigma,
        )
        surf_pts, surf_gt = pts_init[n_dom:], gt_init[n_dom:]

    surf_pts = surf_pts.to(device)
    surf_gt = surf_gt.to(device)

    resample_label = (
        "static (DeepSDF-style: sampled once, reused all epochs)"
        if surface_resample_cadence == 0
        else f"refreshed every {surface_resample_cadence} epochs"
    )
    print(
        f"[single] Hash={config_hash} | shape={shape['shape_type']}/{shape.get('name', '?')} | "
        f"epochs={epochs} lr={lr} batch={batch_size} (n_surf={n_surf}, n_dom={n_dom})"
    )
    print(f"[single] Surface cache: {resample_label}.")

    # ── Optional: on-surface points + per-point unit normals (for normal loss) ─
    norm_pts: Tensor | None = None
    norm_normals: Tensor | None = None
    if use_normal_loss:
        assert ply_tensors is not None  # narrowed by use_normal_loss
        ply_verts_cpu, ply_faces_cpu, _ = ply_tensors
        norm_pts, norm_normals = _sample_surface_points_with_normals(ply_verts_cpu, ply_faces_cpu, normal_batch_size, device)
        print(
            f"[single] Normal-loss enabled: λ_n={normal_loss_weight} on {normal_batch_size} pts/step "
            f"(refresh every {normal_resample_cadence} ep, 0 = static)."
        )
    if use_eikonal_loss:
        print(
            f"[single] Eikonal regulariser enabled: λ_eik={eikonal_loss_weight} on "
            f"{eikonal_batch_size} domain pts/step (off-surface only)."
        )

    # Per-axis sampling bounds for the eikonal regulariser — mirrors
    # outer_prism so eikonal pts share the support of the L1 domain pts.
    eik_bounds = (
        torch.tensor(
            [
                [outer_prism["x"][0], outer_prism["x"][1]],
                [outer_prism["y"][0], outer_prism["y"][1]],
                [outer_prism["z_sampling"][0], outer_prism["z_mesh"][1]],
            ],
            dtype=torch.float32,
            device=device,
        )
        if use_eikonal_loss
        else None
    )

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in tqdm(range(epochs), desc="single_pretrain", unit="ep"):
        model.train()
        optimizer.zero_grad()

        # Optionally refresh surface cache
        if surface_resample_cadence > 0 and epoch > 0 and epoch % surface_resample_cadence == 0:
            pts_re, gt_re = _sample_points_for_shape(
                shape["shape_type"],
                shape_params,
                batch_size,
                device,
                near_surface_fraction=near_surface_fraction,
                domain_fraction=domain_fraction,
                near_surface_tight_sigma=near_surface_tight_sigma,
                near_surface_wide_sigma=near_surface_wide_sigma,
            )
            surf_pts = pts_re[n_dom:].to(device)
            surf_gt = gt_re[n_dom:].to(device)

        # Refresh on-surface cache for normal loss (cheap; pure CPU mesh sampling).
        if use_normal_loss and normal_resample_cadence > 0 and epoch > 0 and epoch % normal_resample_cadence == 0:
            assert ply_tensors is not None
            ply_verts_cpu, ply_faces_cpu, _ = ply_tensors
            norm_pts, norm_normals = _sample_surface_points_with_normals(ply_verts_cpu, ply_faces_cpu, normal_batch_size, device)

        # Fresh domain pts every epoch — surface pts come from the cache.
        pts_fresh, gt_fresh = _sample_points_for_shape(
            shape["shape_type"],
            shape_params,
            batch_size,
            device,
            near_surface_fraction=near_surface_fraction,
            domain_fraction=domain_fraction,
            near_surface_tight_sigma=near_surface_tight_sigma,
            near_surface_wide_sigma=near_surface_wide_sigma,
        )
        domain_pts = pts_fresh[:n_dom]
        domain_gt = gt_fresh[:n_dom]

        all_pts = torch.cat([surf_pts, domain_pts], dim=0)
        all_gt = torch.cat([surf_gt, domain_gt], dim=0)

        pred = model(all_pts).squeeze()

        # NaN guard before computing loss
        if not math.isfinite(pred.sum().item() if pred.numel() > 0 else 0.0):
            print(
                f"[single] NaN/Inf in model output at epoch {epoch}. No checkpoint written. Try lowering learning_rate.",
                file=sys.stderr,
            )
            sys.exit(1)

        # L1-all on (surface ∪ domain) pts in metres for numerical scale. # noqa: RUF003
        # GT SDF magnitudes far from the surface already encode |∇f| ≈ 1, so
        # the L1-all term implicitly does most of the eikonal work.  The
        # explicit normal / eikonal losses below add the *missing* surface-
        # orientation supervision that L1-on-distance can't provide.
        l1_loss = l1_fn(pred / 1000.0, all_gt / 1000.0)
        total_loss = l1_loss

        normal_loss_val = 0.0
        eikonal_loss_val = 0.0

        # ---- Normal loss (IGR-style) -------------------------------------
        # Pin f(x_surf) → 0 *and* align ∇f(x_surf) with the mesh normal.
        # This is the single biggest fix for the "potato + blobs, no fine
        # detail" symptom: L1-on-distance doesn't constrain surface
        # orientation, so the network is free to fit any smooth blob with
        # roughly the right volume.
        if use_normal_loss and norm_pts is not None and norm_normals is not None:
            grad_surf = _grad_of_sdf(model, norm_pts)  # mm-domain grad of mm-domain SDF (unitless ratio)
            grad_norm = grad_surf / (torch.linalg.norm(grad_surf, dim=-1, keepdim=True) + 1e-8)
            normal_alignment = (1.0 - (grad_norm * norm_normals).sum(dim=-1)).mean()
            # Also constrain f(x_surf) = 0 (in metres for scale parity with L1).
            f_surf = model(norm_pts).squeeze(-1)
            on_surface_mse = (f_surf / 1000.0).abs().mean()
            normal_loss = normal_alignment + on_surface_mse
            total_loss = total_loss + normal_loss_weight * normal_loss
            normal_loss_val = float(normal_loss.detach().item())

        # ---- Light eikonal regulariser (off-surface only) ----------------
        # Sample fresh uniform pts in the outer_prism box and push |∇f| → 1.
        # Keeping this OFF the near-surface band avoids the v4 ellipsoid
        # collapse, while still stabilising gradient magnitudes globally.
        if use_eikonal_loss and eik_bounds is not None:
            eik_pts = torch.rand(eikonal_batch_size, 3, device=device) * (eik_bounds[:, 1] - eik_bounds[:, 0]) + eik_bounds[:, 0]
            grad_eik = _grad_of_sdf(model, eik_pts)
            eikonal_loss = ((torch.linalg.norm(grad_eik, dim=-1) - 1.0) ** 2).mean()
            total_loss = total_loss + eikonal_loss_weight * eikonal_loss
            eikonal_loss_val = float(eikonal_loss.detach().item())

        # Diagnostic-only: surface-only L1 (comparable across runs).
        with torch.no_grad():
            surf_loss = l1_fn(pred[:n_surf] / 1000.0, surf_gt / 1000.0).item()
            dom_loss = l1_fn(pred[n_surf:] / 1000.0, domain_gt / 1000.0).item()

        if not math.isfinite(total_loss.item()):
            print(
                f"[single] NaN/Inf loss at epoch {epoch}. No checkpoint written.",
                file=sys.stderr,
            )
            sys.exit(1)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % logging_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            tqdm.write(
                f"[single] epoch={epoch:5d} | total={total_loss.item():.6f} "
                f"| l1={l1_loss.item():.6f} | surf={surf_loss:.6f} | dom={dom_loss:.6f} "
                f"| n={normal_loss_val:.4f} | eik={eikonal_loss_val:.4f} | lr={lr_now:.2e}"
            )
            if use_wandb:
                wandb.log(
                    {
                        # Sweep metric: keep the same name as v4 so existing
                        # sweep YAMLs (metric.name: pretraining/total_loss) work.
                        "pretraining/total_loss": total_loss.item(),
                        "pretraining/l1_loss": l1_loss.item(),
                        "pretraining/surface_loss": surf_loss,
                        "pretraining/domain_loss": dom_loss,
                        "pretraining/normal_loss": normal_loss_val,
                        "pretraining/eikonal_loss": eikonal_loss_val,
                        "pretraining/lr": lr_now,
                    },
                    step=epoch,
                )

    # ── Save checkpoint ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), ckpt_path)
    print(f"\u2713 Saved checkpoint to {ckpt_path}")
    print(f"[single] All outputs → {run_dir}/")

    # ── Optional post-training mesh extraction ──────────────────────────────
    if no_mesh:
        print("[mesh] Skipped (--no-mesh)")
        return
    try:
        model.eval()
        _extract_mesh(model, ckpt_path, device, mesh_res, outer_prism)
    except Exception as exc:
        print(f"[mesh] ERROR: {exc}. Checkpoint is intact at {ckpt_path}.", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-shape SDF pretraining")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count from config (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        dest="no_mesh",
        help="Skip post-training PLY mesh extraction.",
    )
    args, _ = parser.parse_known_args()  # ignore wandb sweep flags passed as CLI args
    main(epochs_override=args.epochs, no_mesh=args.no_mesh)
