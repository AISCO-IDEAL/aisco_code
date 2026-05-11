"""Latent-population pretraining for ConcatWIREV2 (Plan 005 / v0.3.0).

Trains model weights so that each latent slot z_i approximates a target SDF
(sphere, cube, or PLY mesh).  All N shapes are batched into a single forward
pass for efficiency.

Entry point: ``pretraining_latent(model, config, device)``
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm, scale_outer_prism
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _extract_mesh,
    _sample_points_for_shape,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization import SamplingVisualizer
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager import (
    CentralVisManager,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.points_SDFmesh_to_ply import (
    _POINT_COLOUR_GREEN,
    _POINT_COLOUR_RED,
    _POINT_COLOUR_WHITE,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.latent_utils import build_unit_circle_Z
from subprojects.wp5.unified_mesh_extraction import clear_mesh_cache


def _load_ply_tensors(ply_path: str, ply_scale: float = 1.0) -> tuple[Tensor, Tensor, float]:
    """Load a PLY mesh, transform to AISCO convention, and return (verts, faces, bbox_extent).

    Applies three deterministic transforms after scaling:
    1. Axis permutation (data-driven): the PLY axis with the largest coordinate
       range maps to AISCO_X (length); the axis with the second-largest range maps
       to AISCO_Y (width); the remaining axis maps to AISCO_Z (height).  If the
       PLY is already correctly oriented (largest extent already on axis 0, second
       on axis 1) no reordering takes place.
    2. X centering: shifts so that the car is symmetrically centred on the X axis.
    3. Z grounding: shifts so that Z_min equals the outer-prism ground clearance (100mm).

    Returned tensors are always on **CPU**.  Move to device only when needed (e.g.
    inside ``_sample_points_for_shape`` for the GPU BVH query).  Keeping PLY geometry
    on CPU prevents CUDA-context pollution from spreading to visualization code.

    Args:
        ply_path: Absolute path to the PLY file.
        ply_scale: Multiplicative scale applied to raw vertex coordinates.  Use
            ``100.0`` when the PLY is stored in decimetres (the AISCO pretraining
            convention).  Use ``1.0`` when the PLY is already in mm.

    Returns:
        Tuple of (ply_verts, ply_faces, bbox_extent) where bbox_extent is the
        diagonal of the mesh bounding box after all transforms (in mm).
    """
    raw_mesh = trimesh.load(ply_path)
    if not isinstance(raw_mesh, trimesh.Trimesh):
        raise TypeError(
            f"Expected a Trimesh object when loading '{ply_path}', got {type(raw_mesh)}. "
            "Ensure the PLY file contains a single solid mesh."
        )
    if not raw_mesh.is_watertight:
        shape_name_raw = Path(ply_path).stem
        print(f"[PLY Loader] {shape_name_raw}: not watertight — attempting trimesh repair …")
        trimesh.repair.fill_holes(raw_mesh)
        trimesh.repair.fix_normals(raw_mesh)
        if raw_mesh.is_watertight:
            print(f"[PLY Loader] {shape_name_raw}: repair succeeded — mesh is now watertight.")
        else:
            print(
                f"[PLY Loader] WARNING: {shape_name_raw}: repair did not produce a watertight mesh. "
                "GT SDF values near open edges may be inaccurate. "
                "For best results, manually repair with 'meshlabserver' (close_holes.mlx) or 'pymeshfix'."
            )
    verts_np = np.array(raw_mesh.vertices, dtype="float32") * ply_scale
    faces_np = np.array(raw_mesh.faces, dtype="int64")

    # --- Axis permutation: map largest-extent axis → X, second → Y, smallest → Z ---
    raw_extents = verts_np.max(axis=0) - verts_np.min(axis=0)  # shape (3,)
    perm: list[int] = list(np.argsort(raw_extents)[::-1])  # descending by extent
    verts_np = verts_np[:, perm]

    # --- Winding fix: odd permutations flip coordinate handedness → invert face normals.
    # Detect parity by counting inversions; if odd, reverse face winding (swap cols 1 & 2).
    perm_inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
    if perm_inversions % 2 == 1:
        faces_np = faces_np[:, [0, 2, 1]]  # restore CCW winding after odd-permutation reflection

    # --- X centering: shift so that X_center = 0 (car centred on length axis) ---
    x_center = (float(verts_np[:, 0].max()) + float(verts_np[:, 0].min())) / 2.0
    verts_np[:, 0] -= x_center

    # --- Y centering: shift so that Y_center = 0 (car symmetric about lateral axis) ---
    y_center = (float(verts_np[:, 1].max()) + float(verts_np[:, 1].min())) / 2.0
    verts_np[:, 1] -= y_center

    # --- Z grounding: shift so Z_min = ground_clearance (90 + 10 = 100mm) ---
    prism = get_outer_prism_bounds_mm()
    ground_clearance = abs(prism["z"][0]) + 10.0  # to avoid the pretrained cars to be cut off at the bottom
    z_shift = ground_clearance - float(verts_np[:, 2].min())
    verts_np[:, 2] += z_shift

    # --- Telemetry ---
    shape_name = Path(ply_path).stem
    extents = verts_np.max(axis=0) - verts_np.min(axis=0)
    ctr = (verts_np.max(axis=0) + verts_np.min(axis=0)) / 2.0
    z_min_after = float(verts_np[:, 2].min())
    x_center_after = float(ctr[0])
    y_center_after = float(ctr[1])
    print(
        f"[PLY Loader] {shape_name}: "
        f"perm={perm}  winding_fix={'yes' if perm_inversions % 2 == 1 else 'no'}  "
        f"ext=({extents[0]:.0f},{extents[1]:.0f},{extents[2]:.0f})mm  "
        f"center=({ctr[0]:.0f},{ctr[1]:.0f},{ctr[2]:.0f})mm  "
        f"Z_min={z_min_after:.0f}mm  X_center={x_center_after:.0f}mm  Y_center={y_center_after:.0f}mm"
    )

    # --- Outer-prism violation warnings ---
    prism_length = prism["x"][1] - prism["x"][0]
    prism_width = prism["y"][1] - prism["y"][0]
    prism_height = float(prism["z"][1])
    axis_checks = [("X", float(extents[0]), prism_length), ("Y", float(extents[1]), prism_width)]
    for axis_name, extent_mm, limit_mm in axis_checks:
        if extent_mm > limit_mm:
            pct = 100.0 * (extent_mm - limit_mm) / limit_mm
            print(
                f"[PLY Loader] WARNING: {shape_name} {axis_name}-extent {extent_mm:.0f}mm "
                f"exceeds outer_prism limit {limit_mm:.0f}mm by {pct:.1f}%"
            )
    z_max_after = float(verts_np[:, 2].max())
    if z_max_after > prism_height:
        pct = 100.0 * (z_max_after - prism_height) / prism_height
        print(
            f"[PLY Loader] WARNING: {shape_name} Z_max {z_max_after:.0f}mm "
            f"exceeds outer_prism height {prism_height:.0f}mm by {pct:.1f}%"
        )

    bbox_diag = float(np.linalg.norm(extents))
    ply_verts = torch.tensor(verts_np)  # CPU — move to device in _sample_points_for_shape for BVH
    ply_faces = torch.tensor(faces_np)  # CPU
    return ply_verts, ply_faces, max(bbox_diag, 1.0)


def _build_shape_params(
    shape: dict[str, Any],
    ply_cache: dict[int, tuple[Tensor, Tensor, float]],
    slot_idx: int,
) -> dict[str, Any]:
    """Convert a latent_population entry into a shape_params dict.

    Args:
        shape: Entry from ``config["latent_population"]``.
        ply_cache: Pre-loaded PLY tensors keyed by slot index.
        slot_idx: Index of this slot (used for PLY cache lookup).

    Returns:
        Dict suitable for ``_sample_points_for_shape``.
    """
    shape_type: str = shape["shape_type"]
    if shape_type == "sphere":
        return {"radius": float(shape["radius"])}
    if shape_type == "cube":
        return {"side_length": float(shape["side_length"])}
    if shape_type == "ply":
        ply_verts, ply_faces, bbox_extent = ply_cache[slot_idx]
        sigma_factor: float = float(shape.get("surface_noise_sigma_factor", 0.05))
        return {
            "ply_verts": ply_verts,
            "ply_faces": ply_faces,
            "bbox_extent": bbox_extent,
            "surface_noise_sigma_factor": sigma_factor,
        }
    raise ValueError(
        f"Unknown shape_type '{shape_type}' in latent_population slot {slot_idx}. Valid values: 'sphere', 'cube', 'ply'."
    )


def _load_or_build_surface_cache(
    ply_path: str,
    ply_scale: float,
    shape_params: dict[str, Any],
    batch_size: int,
    n_dom: int,
    device: str,
    cache_dir: Path | None,
    *,
    near_surface_fraction: float = 0.50,
    domain_fraction: float = 0.50,
    near_surface_tight_sigma: float | None = None,
    near_surface_wide_sigma: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Load surface pts + GT SDF from disk cache, or build and save them.

    Cache key is derived from all parameters that affect the point distribution.
    Cache file: ``cache_dir/<hash>.pt``.  On a cache miss the file is written
    so that subsequent calls return immediately.

    Args:
        ply_path: Absolute path to the PLY file (used for cache key only; mesh
            must already be loaded into *shape_params*).
        ply_scale: Multiplicative scale applied when the PLY was loaded (for key).
        shape_params: Dict with ``"ply_verts"``, ``"ply_faces"``,
            ``"bbox_extent"``, ``"surface_noise_sigma_factor"`` as produced by
            ``_build_shape_params``.
        batch_size: Total batch size passed to ``_sample_points_for_shape``.
        n_dom: Number of domain (eikonal) pts per slot.  Surface pts =
            ``batch_size - n_dom``.
        device: PyTorch device string.
        cache_dir: Directory for ``.pt`` cache files.  ``None`` = disabled
            (always build fresh, never write).
        near_surface_fraction: Forwarded to ``_sample_points_for_shape``.
        domain_fraction: Forwarded to ``_sample_points_for_shape``.
        near_surface_tight_sigma: Forwarded to ``_sample_points_for_shape``.
        near_surface_wide_sigma: Forwarded to ``_sample_points_for_shape``.

    Returns:
        ``(surf_pts, surf_gt)`` — the surface half of the sampled batch
        (``batch_size - n_dom`` rows).
    """
    n_surf = batch_size - n_dom
    sigma_key = float(shape_params.get("surface_noise_sigma_factor", 0.05))
    tight_key = near_surface_tight_sigma if near_surface_tight_sigma is not None else -1.0
    wide_key = near_surface_wide_sigma if near_surface_wide_sigma is not None else -1.0

    if cache_dir is not None:
        # Include PLY file content hash so that an in-place file update (same path,
        # new bytes) automatically produces a new cache key and forces a rebuild.
        # Cost: one extra file read at startup, negligible vs BVH computation.
        ply_content_hash = hashlib.sha256(Path(ply_path).read_bytes()).hexdigest()[:8]
        key_str = (
            f"{Path(ply_path).stem}|{ply_scale}|{sigma_key}|{tight_key}|{wide_key}"
            f"|{n_surf}|{near_surface_fraction}|{domain_fraction}|{ply_content_hash}"
        )
        cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:10]
        cache_path = cache_dir / f"{cache_hash}.pt"

        if cache_path.exists():
            payload: dict[str, Any] = torch.load(cache_path, weights_only=True)
            cached_pts: Tensor = payload["pts"]
            cached_gt: Tensor = payload["gt"]
            print(f"[latent_space] Loaded cached SDF surface pts for {Path(ply_path).stem}. Population size: {len(cached_pts)}")
            return cached_pts.to(device), cached_gt.to(device)

    # Cache miss (or caching disabled): build via sampling
    pts_all, gt_all = _sample_points_for_shape(
        "ply",
        shape_params,
        batch_size,
        device,
        near_surface_fraction=near_surface_fraction,
        domain_fraction=domain_fraction,
        near_surface_tight_sigma=near_surface_tight_sigma,
        near_surface_wide_sigma=near_surface_wide_sigma,
    )
    surf_pts = pts_all[n_dom:]
    surf_gt = gt_all[n_dom:]

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pts": surf_pts.cpu(),
                "gt": surf_gt.cpu(),
                "meta": {
                    "ply_path": ply_path,
                    "ply_scale": ply_scale,
                    "ply_content_hash": ply_content_hash,
                    "surface_noise_sigma_factor": sigma_key,
                    "near_surface_tight_sigma": tight_key,
                    "near_surface_wide_sigma": wide_key,
                    "n_surf": n_surf,
                    "near_surface_fraction": near_surface_fraction,
                    "domain_fraction": domain_fraction,
                    "date": datetime.now().isoformat(),
                },
            },
            cache_path,
        )

    return surf_pts, surf_gt


def _resolve_active_population(config: dict[str, Any]) -> None:
    """Resolve active_population + shape_library into config["latent_population"].

    Idempotent: if active_population is absent (legacy format with a direct
    latent_population list), config is left unchanged.

    Args:
        config: Pretraining config dict, modified in place.

    Raises:
        ValueError: If a name in active_population is not found in shape_library.
    """
    if "active_population" not in config:
        return

    library: dict[str, Any] = config.get("shape_library", {})
    if not library:
        raise ValueError(
            "active_population is set but shape_library is empty or missing. "
            "Add shape definitions to shape_library in pretraining_latent_config.yaml."
        )

    name_counts: dict[str, int] = {}
    resolved: list[dict[str, Any]] = []
    for name in config["active_population"]:
        if name not in library:
            raise ValueError(
                f"Shape '{name}' listed in active_population not found in shape_library. "
                f"Available shapes: {sorted(library.keys())}."
            )
        entry = dict(library[name])  # shallow copy — do not mutate the library
        count = name_counts.get(name, 0)
        name_counts[name] = count + 1
        entry["name"] = name if count == 0 else f"{name}_{count + 1}"
        resolved.append(entry)

    config["latent_population"] = resolved


def _compute_circle_validity_fraction(
    model: SupportsLatentConcat,
    z_radius: float,
    z_dim: int,
    outer_prism: dict[str, Any],
    device: str,
    n_probe_angles: int = 36,
    n_pts: int = 4096,
) -> tuple[float, int, int]:
    """Probe what fraction of equally-spaced angles on the z-circle have a valid zero crossing.

    A zero crossing is possible iff the model's SDF minimum over a uniform prism
    sample is negative.  Uses no marching cubes — very fast.

    Args:
        model: Pretrained model with set_latent_vector interface.
        z_radius: Radius of the latent circle to probe.
        z_dim: Latent dimensionality (must be ≥ 2).
        outer_prism: Dict with x/y/z bound lists, e.g. {"x": [lo, hi], ...}.
        device: PyTorch device string.
        n_probe_angles: Number of equally-spaced angles to test (default 36 = 10° spacing).
        n_pts: Number of uniform prism samples per angle probe.

    Returns:
        (fraction_valid, n_valid, n_probe_angles)
    """
    lo = torch.tensor([outer_prism["x"][0], outer_prism["y"][0], outer_prism["z"][0]], device=device)
    hi = torch.tensor([outer_prism["x"][1], outer_prism["y"][1], outer_prism["z"][1]], device=device)
    pts = lo + torch.rand(n_pts, 3, device=device) * (hi - lo)

    model.eval()
    n_valid = 0
    with torch.no_grad():
        for k in range(n_probe_angles):
            angle = 2.0 * math.pi * k / n_probe_angles
            z_vec = torch.zeros(z_dim, device=device)
            z_vec[0] = z_radius * math.cos(angle)
            z_vec[1] = z_radius * math.sin(angle)
            model.set_latent_vector(z_vec)
            sdf = model(pts).squeeze()
            if float(sdf.min().item()) < 0.0:
                n_valid += 1
    model.train()

    return n_valid / n_probe_angles, n_valid, n_probe_angles


def _compute_latent_pretrain_hash(config: dict[str, Any], model: SupportsLatentConcat) -> str:
    """Compute a stable hash over model architecture and all training-significant parameters.

    The hash covers: full model architecture (via get_model_info_dict), z_placement,
    z_radius, the complete training_loop dict (including bbox_scale_factor), and
    per-shape specs (name, type, ply_path_stem, ply_scale, sigma_factor, radius,
    side_length).  Two runs with identical parameters produce the same hash;
    any significant parameter change produces a different hash.

    Note: Only the PLY stem (not the full path) is hashed, so moving a PLY file
    to a different directory without renaming it does NOT force a rehash.

    Args:
        config: Dict matching the pretraining_latent_config.yaml schema.
        model: Model whose architecture is included in the hash.

    Returns:
        10-character hex hash string.
    """
    _resolve_active_population(config)
    shapes: list[dict[str, Any]] = config.get("latent_population", [])
    shape_entries = [
        {
            "name": s["name"],
            "type": s["shape_type"],
            "stem": Path(s["ply_path"]).stem if s.get("ply_path") else None,
            "scale": s.get("ply_scale"),
            "sigma": s.get("surface_noise_sigma_factor"),
            "radius": s.get("radius"),
            "side_length": s.get("side_length"),
        }
        for s in shapes
    ]
    key: dict[str, Any] = {
        "model": model.get_model_info_dict(),
        "z_placement": config.get("z_placement", "circle"),
        "z_radius": config.get("z_radius", 2.0),
        "training_loop": config["training_loop"],
        "shapes": shape_entries,
    }
    return dict_hashing(key)


def _resolve_checkpoint_path(config: dict[str, Any], model: SupportsLatentConcat) -> Path:
    """Return the canonical checkpoint path for a given config and model.

    All outputs for a run go into a single folder inside the canonical base dir:
      <base_dir>/<epochs:06d>_<ModelClass>_<config_hash>/
        pretrained_latent_<config_hash>.pth
        model_info_dict.json
        sampling_vis/
        pretrained_shapes/

    Both standalone scripts and main_v2 call this function, so the same config
    always resolves to the same folder regardless of launch context.

    Args:
        config: Dict matching the pretraining_latent_config.yaml schema.
        model: Model whose architecture is included in the hash.

    Returns:
        Absolute Path to the checkpoint file inside the canonical folder.
    """
    config_hash = _compute_latent_pretrain_hash(config, model)
    epochs: int = int(config["training_loop"]["epochs"])
    model_class_name: str = model.__class__.__name__
    base_dir = Path(config["output"]["base_dir"])
    folder_name = f"{epochs:06d}_{model_class_name}_{config_hash}"
    return base_dir / folder_name / f"pretrained_latent_{config_hash}.pth"


def _compute_input_grad_norms(
    model: SupportsLatentConcat,
    pts: Tensor,
    z: Tensor,
    gt_sdf: Tensor,
    device: str,  # noqa: ARG001  # kept for API consistency; tensors must already be on the correct device
) -> tuple[float, float, bool]:
    """Probe ‖∂L/∂z_input‖ and ‖∂L/∂xyz_input‖ on a small representative batch.

    Uses ``torch.autograd.grad`` (not ``.backward()``) so that model parameter
    ``.grad`` attributes are never touched — no optimizer-state pollution.

    The scalar probe loss is intentionally MSE(pred/1000, gt_sdf/1000) rather
    than the training loss (L1-all).  MSE is a valid diagnostic probe: it is
    smooth everywhere and produces clean gradient signals for sensitivity
    analysis.  Using MSE here does NOT imply that training uses MSE — it is
    a deliberate choice for the gradient-sensitivity diagnostic.
    ``gt_sdf`` is detached; only ``pts`` and ``z`` carry the computation graph.

    Args:
        model: ConcatWIREV2 (or any SupportsLatentConcat).  Caller manages
            train/eval mode; this function does not switch mode.
        pts: (M, 3) surface query points in mm.  Will be detached internally.
        z: (M, z_dim) latent vectors aligned with pts.  Will be detached.
        gt_sdf: (M,) ground-truth SDF in mm.  Detached; used only for loss.
        device: PyTorch device string (unused; reserved for API symmetry).

    Returns:
        ``(z_grad_norm, xyz_grad_norm, z_detached_warn)`` where:

        - ``z_grad_norm``: ‖∂L/∂z_input‖ (0.0 if z not in graph).
        - ``xyz_grad_norm``: ‖∂L/∂xyz_input‖ (0.0 if xyz not in graph).
        - ``z_detached_warn``: True when z gradient is None (z not in graph).
    """
    pts_probe = pts.detach().requires_grad_(True)
    z_probe = z.detach().requires_grad_(True)

    pred = model(pts_probe, z=z_probe).squeeze()
    loss = ((pred / 1000.0 - gt_sdf.detach() / 1000.0) ** 2).mean()

    grads = torch.autograd.grad(
        outputs=loss,
        inputs=[z_probe, pts_probe],
        allow_unused=True,
    )
    z_grad, pts_grad = grads

    z_detached = z_grad is None
    z_norm = 0.0 if z_detached else float(z_grad.norm().item())
    xyz_norm = float(pts_grad.norm().item()) if pts_grad is not None else 0.0
    return z_norm, xyz_norm, z_detached


def _check_state_dict_compatibility(
    model: SupportsLatentConcat,
    ckpt_state_dict: dict[str, Any],
    ckpt_path: Path,
) -> None:
    """Raise ValueError if ckpt_state_dict is structurally incompatible with model.

    Checks key sets and per-key tensor shapes.  Called before load_state_dict
    so the error is actionable rather than a raw PyTorch RuntimeError.

    Args:
        model: The model instance to load weights into.
        ckpt_state_dict: The ``model_state_dict`` loaded from a checkpoint.
        ckpt_path: Path of the checkpoint file (used in error messages only).

    Raises:
        ValueError: If keys or tensor shapes do not match.
    """
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(ckpt_state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    if missing or unexpected:
        raise ValueError(
            f"Architecture mismatch: checkpoint at '{ckpt_path}' has incompatible keys.\n"
            f"  Missing in checkpoint : {sorted(missing) if missing else 'none'}\n"
            f"  Unexpected in checkpoint: {sorted(unexpected) if unexpected else 'none'}\n"
            "The checkpoint was likely trained with a different model architecture. "
            "Update specific_pretraining_path in pretraining_latent_config.yaml to a "
            "checkpoint that matches the current model."
        )

    shape_mismatches = [
        f"  '{k}': model={tuple(model.state_dict()[k].shape)}  ckpt={tuple(ckpt_state_dict[k].shape)}"
        for k in model_keys
        if model.state_dict()[k].shape != ckpt_state_dict[k].shape
    ]
    if shape_mismatches:
        raise ValueError(
            f"Architecture mismatch: checkpoint at '{ckpt_path}' has incompatible tensor shapes "
            f"({len(shape_mismatches)} parameter(s)):\n"
            + "\n".join(shape_mismatches)
            + "\nThe checkpoint was likely trained with different hidden_dim or z_dim. "
            "Update specific_pretraining_path in pretraining_latent_config.yaml to a "
            "compatible checkpoint."
        )


def pretraining_latent(  # noqa: PLR0912, PLR0915  # training loop is intrinsically sequential; decomposition would obscure control flow
    model: SupportsLatentConcat,
    config: dict[str, Any],
    device: str,
    *,
    use_wandb: bool = False,
    vis_sampling: bool = False,
    vis_output_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    vis_manager: CentralVisManager | None = None,
) -> tuple[SupportsLatentConcat, Tensor]:
    """Run latent-population pretraining for a SupportsLatentConcat model.

    Trains model weights so that each latent slot z_i (placed on a circle)
    approximates a corresponding target SDF (sphere / cube / PLY mesh).

    Fast-resume: if a valid checkpoint exists at the resolved checkpoint path
    and ``config["force_pretraining"]`` is False, the checkpoint is loaded and returned
    without training.

    Args:
        model: ConcatWIREV2 (or any SupportsLatentConcat model).
        config: Dict matching the pretraining_latent_config.yaml schema.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        checkpoint_path: Optional override for the checkpoint file path.  When
            provided (e.g. by main_v2 to co-locate with standard pretrained
            models), ``_resolve_checkpoint_path`` is skipped and this path is
            used directly.  When ``None`` (standalone / script use), the path is
            derived from ``config["output"]["checkpoint_path"]`` plus a config
            hash subdirectory.

    Returns:
        Tuple ``(model, Z)`` where ``Z`` has shape ``(N, z_dim)`` and
        ``model`` has been updated with pretrained weights.

    Raises:
        TypeError: If model does not implement SupportsLatentConcat.
        FileNotFoundError: If a PLY shape_type is configured but the file is missing.
        ValueError: If an existing checkpoint uses the old single-tensor format.
        SystemExit: If NaN/Inf loss is detected during training.
    """
    if not isinstance(model, SupportsLatentConcat):
        raise TypeError(f"Model {type(model).__name__} does not implement SupportsLatentConcat. ")

    _resolve_active_population(config)

    # --- Specific pretraining path override ---
    if bool(config.get("use_specific_pretraining", False)):
        specific_path_raw = config.get("specific_pretraining_path")
        if not specific_path_raw:
            raise ValueError(
                "use_specific_pretraining is true but specific_pretraining_path is not set. "
                "Set specific_pretraining_path to the absolute path of the .pth file in "
                "pretraining_latent_config.yaml."
            )
        specific_path = Path(specific_path_raw)
        if not specific_path.exists():
            raise FileNotFoundError(
                f"specific_pretraining_path does not exist: {specific_path}. Check the path in pretraining_latent_config.yaml."
            )
        ckpt = torch.load(specific_path, map_location=device, weights_only=True)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "z_pretrained" not in ckpt:
            raise ValueError(
                f"Checkpoint at '{specific_path}' is missing 'model_state_dict' or 'z_pretrained' keys. "
                "Ensure the file is a valid latent pretraining checkpoint."
            )
        _check_state_dict_compatibility(model, ckpt["model_state_dict"], specific_path)
        print(f"[latent_pretrain] OK  architecture_compatible  model={model.__class__.__name__}  path={specific_path}")
        model.load_state_dict(ckpt["model_state_dict"])
        Z: Tensor = ckpt["z_pretrained"].to(device)
        print(f"[latent_pretrain] LOAD  path={specific_path}  status=specific_pretraining_override")
        return model, Z

    ckpt_path = checkpoint_path if checkpoint_path is not None else _resolve_checkpoint_path(config, model)
    config_hash = _compute_latent_pretrain_hash(config, model)
    force: bool = bool(config.get("force_pretraining", True))

    # --- Fast resume ---
    if not force and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "z_pretrained" not in ckpt:
            raise ValueError(
                f"Checkpoint at '{ckpt_path}' appears to use the old single-tensor format "
                "(missing 'model_state_dict' or 'z_pretrained' keys). "
                "Re-run pretraining with force_pretraining=true in "
                "pretraining_latent_config.yaml to generate a new combined checkpoint."
            )
        # --- Hash safety check (M5): detect manual copies or hash collisions ---
        loaded_hash = ckpt.get("meta", {}).get("config_hash")
        if loaded_hash is not None and loaded_hash != config_hash:
            print(f"[latent_pretrain] WARN  hash_mismatch  loaded={loaded_hash}  current={config_hash}  path={ckpt_path}")
        model.load_state_dict(ckpt["model_state_dict"])
        Z = ckpt["z_pretrained"].to(device)
        print(f"[latent_pretrain] LOAD  path={ckpt_path}  hash={config_hash}  status=fast_resume")
        return model, Z

    shapes: list[dict[str, Any]] = config["latent_population"]
    num_shapes = len(shapes)

    # --- PLY validation before training ---
    for slot_idx, shape in enumerate(shapes):
        if shape["shape_type"] == "ply":
            ply_path = shape.get("ply_path")
            if not ply_path or not Path(ply_path).exists():
                raise FileNotFoundError(
                    f"PLY file not found for shape slot {slot_idx} "
                    f"('{shape.get('name', slot_idx)}'): {ply_path!r}. "
                    "Check the ply_path field in pretraining_latent_config.yaml."
                )

    # --- Pre-load PLY meshes ---
    ply_cache: dict[int, tuple[Tensor, Tensor, float]] = {}
    for slot_idx, shape in enumerate(shapes):
        if shape["shape_type"] == "ply":
            ply_scale: float = float(shape.get("ply_scale", 1.0))
            ply_cache[slot_idx] = _load_ply_tensors(str(shape["ply_path"]), ply_scale)
            print(f"[latent_pretrain] Slot {slot_idx}: loaded PLY '{shape['ply_path']}' scale={ply_scale}")

    # --- Build circle Z ---
    z_dim: int = model.get_model_info_dict()["z_dim"]
    z_radius: float = float(config.get("z_radius", 1.0))
    Z = build_unit_circle_Z(num_shapes, z_dim, z_radius, device)

    # --- Training hyper-parameters ---
    loop_cfg: dict[str, Any] = config["training_loop"]
    epochs: int = int(loop_cfg["epochs"])
    lr: float = float(loop_cfg["learning_rate"])
    batch_size_per_shape: int = int(loop_cfg["batch_size_per_shape"])
    logging_interval: int = int(loop_cfg["logging_interval"])
    # 0 = static surface cache (DeepSDF-style: sample once, reuse forever)
    # N > 0 = refresh surface cache every N epochs
    surface_resample_cadence: int = int(loop_cfg.get("surface_resample_cadence", 0))
    track_band_losses: bool = bool(loop_cfg.get("track_band_losses", False))
    eikonal_weight: float = float(loop_cfg.get("eikonal_weight", 0.0))

    # --- M2: Dual-band near-surface sampling params ---
    # Defaults reproduce the legacy single-band PLY behaviour (50/50 domain/surface, one band).
    near_surface_fraction: float = float(loop_cfg.get("near_surface_fraction", 0.50))
    domain_fraction: float = float(loop_cfg.get("domain_fraction", 0.50))
    # None → single-band fallback (uses surface_noise_sigma_factor from shape_params)
    near_surface_tight_sigma: float | None = (
        float(loop_cfg["near_surface_tight_sigma"]) if loop_cfg.get("near_surface_tight_sigma") is not None else None
    )
    near_surface_wide_sigma: float | None = (
        float(loop_cfg["near_surface_wide_sigma"]) if loop_cfg.get("near_surface_wide_sigma") is not None else None
    )
    interior_boost_fraction: float = float(loop_cfg.get("interior_boost_fraction", 0.0))

    # --- Output / visualisation config (read once, shared between loop and post-training) ---
    output_cfg: dict[str, Any] = config.get("output", {})
    run_dir: Path = ckpt_path.parent  # Canonical folder containing all outputs
    visualisation_directory: str = str(output_cfg.get("visualisation_directory", "sampling_vis"))
    vis_dir: Path = vis_output_dir if vis_output_dir is not None else run_dir / visualisation_directory
    outer_prism_raw = get_outer_prism_bounds_mm()  # Single source of truth for bounding box
    bbox_scale_factor: float = float(loop_cfg.get("bbox_scale_factor", 1.0))  # Scales outer_prism for domain sampling
    outer_prism = scale_outer_prism(outer_prism_raw, bbox_scale_factor)

    vis_cadence: int = int(output_cfg.get("vis_cadence", 0))  # 0 = disabled
    vis_mesh_res: int = int(output_cfg.get("mesh_resolution", 64))

    # --- M3: Precomputed SDF cache ---
    _sdf_cache_dir_raw = output_cfg.get("sdf_cache_dir")
    sdf_cache_dir: Path | None = Path(_sdf_cache_dir_raw) if _sdf_cache_dir_raw is not None else None

    vis_show_domain_pts: bool = bool(
        output_cfg.get("vis_show_domain_pts", output_cfg.get("vis_show_eikonal_pts", True))
    )  # GREY: domain pts used for L1-all loss
    vis_show_target_pts: str = str(output_cfg.get("vis_show_target_pts", "pts"))  # "pts" or "mesh"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    l1_fn = torch.nn.L1Loss()  # L1 on all pts (surface + domain) replaces MSE-surface + eikonal

    # --- Build initial surface sample cache (near-surface Gaussian-perturbed pts for MSE) ---
    # Split convention: domain = pts[:n_dom], surface = pts[n_dom:].
    # n_dom = int(batch * domain_fraction) — matches _sample_points_for_shape PLY branch.
    # Sphere/cube hardcode 50 % domain; domain_fraction should be kept at 0.50 when mixing
    # PLY with sphere/cube shapes to maintain consistent surface-cache sizes.
    n_dom: int = int(batch_size_per_shape * domain_fraction)
    n_surf: int = batch_size_per_shape - n_dom
    resample_label = (
        "static (DeepSDF-style: sampled once, reused all epochs)"
        if surface_resample_cadence == 0
        else f"refreshed every {surface_resample_cadence} epochs"
    )
    print(f"[latent_pretrain] Surface cache: {resample_label}. L1-all uses surface cache + fresh domain pts.", flush=True)
    if bbox_scale_factor != 1.0:
        print(
            f"[latent_pretrain] Mesh bbox scale factor: {bbox_scale_factor}x outer_prism (expanded from wp2)",
            flush=True,
        )
    cached_surface: list[tuple[Tensor, Tensor]] = []
    for slot_idx, shape in enumerate(shapes):
        shape_params = _build_shape_params(shape, ply_cache, slot_idx)
        if shape["shape_type"] == "ply":
            surf_pts_i, surf_gt_i = _load_or_build_surface_cache(
                ply_path=str(shape["ply_path"]),
                ply_scale=float(shape.get("ply_scale", 1.0)),
                shape_params=shape_params,
                batch_size=batch_size_per_shape,
                n_dom=n_dom,
                device=device,
                cache_dir=sdf_cache_dir,
                near_surface_fraction=near_surface_fraction,
                domain_fraction=domain_fraction,
                near_surface_tight_sigma=near_surface_tight_sigma,
                near_surface_wide_sigma=near_surface_wide_sigma,
            )
            cached_surface.append((surf_pts_i, surf_gt_i))
        else:
            pts_i, gt_i = _sample_points_for_shape(shape["shape_type"], shape_params, batch_size_per_shape, device)
            cached_surface.append((pts_i[n_dom:], gt_i[n_dom:]))  # surface = second half

    print(
        f"[latent_pretrain] Starting: N={num_shapes} shapes, epochs={epochs}, "
        f"lr={lr}, batch_size_per_shape={batch_size_per_shape} (n_surf={n_surf}, n_dom={n_dom}), device={device}",
        flush=True,
    )
    if interior_boost_fraction > 0.0:
        print(
            f"[latent_pretrain] Interior boost ENABLED: target ≥{interior_boost_fraction:.0%} of domain pts inside mesh "
            f"(affects PLY shapes only; rejection-samples {interior_boost_fraction * n_dom:.0f} interior pts per slot per epoch).",
            flush=True,
        )

    # --- Per-epoch visualizer (b): current trained mesh + loss sample points ---
    epoch_vis: SamplingVisualizer | None = None
    if vis_sampling and vis_cadence > 0:
        epoch_vis = SamplingVisualizer(output_dir=vis_dir, is_static=False)
        epoch_vis.write_legend(_make_vis_legend(config_hash, vis_cadence, vis_show_domain_pts, surface_resample_cadence))
        print(f"[latent_pretrain] Per-epoch vis enabled: cadence={vis_cadence}, mesh_res={vis_mesh_res} → {vis_dir}")

    # --- Register run metadata in W&B config (keys sourced from config when present) ---
    if use_wandb:
        import wandb  # noqa: PLC0415

        wandb.config.update(
            {
                "loss_type": loop_cfg.get("loss_type", "l1-all"),
                "sampling_strategy": loop_cfg.get("sampling_strategy", "all-pts-split"),
                "run_group": loop_cfg.get("run_group", ""),
            },
            allow_val_change=True,
        )

    # --- Training loop ---
    for epoch in tqdm(range(epochs), desc="latent_pretrain", unit="ep"):
        model.train()
        optimizer.zero_grad()

        # Optionally refresh surface cache at the requested cadence
        if surface_resample_cadence > 0 and epoch > 0 and epoch % surface_resample_cadence == 0:
            for slot_idx, shape in enumerate(shapes):
                shape_params = _build_shape_params(shape, ply_cache, slot_idx)
                pts_i, gt_i = _sample_points_for_shape(
                    shape["shape_type"],
                    shape_params,
                    batch_size_per_shape,
                    device,
                    near_surface_fraction=near_surface_fraction,
                    domain_fraction=domain_fraction,
                    near_surface_tight_sigma=near_surface_tight_sigma,
                    near_surface_wide_sigma=near_surface_wide_sigma,
                )
                cached_surface[slot_idx] = (pts_i[n_dom:], gt_i[n_dom:])

        # Sample fresh domain pts (uniform bbox) for L1-all loss — every epoch
        all_domain_pts: list[Tensor] = []  # Domain points evaluated in the model
        all_domain_gt: list[Tensor] = []  # Ground truth SDF for L1-all loss
        all_domain_z: list[Tensor] = []
        all_surface_pts: list[Tensor] = []  # Surface points evaluated in the model
        all_surface_gt: list[Tensor] = []  # Ground truth SDF for surface points
        all_surface_z: list[Tensor] = []  # Latent vectors for surface points

        for slot_idx, shape in enumerate(shapes):
            shape_params = _build_shape_params(shape, ply_cache, slot_idx)
            # Domain pts: always fresh; GT SDF retained for L1-all loss
            pts_i, domain_gt_i = _sample_points_for_shape(
                shape["shape_type"],
                shape_params,
                batch_size_per_shape,
                device,
                near_surface_fraction=near_surface_fraction,
                domain_fraction=domain_fraction,
                near_surface_tight_sigma=near_surface_tight_sigma,
                near_surface_wide_sigma=near_surface_wide_sigma,
                interior_min_fraction=interior_boost_fraction,
            )
            domain_pts_i = pts_i[:n_dom]
            # Surface pts: from cache (or refreshed above)
            surf_pts_i, surf_gt_i = cached_surface[slot_idx]
            all_domain_pts.append(domain_pts_i)
            all_domain_gt.append(domain_gt_i[:n_dom])
            all_domain_z.append(Z[slot_idx].unsqueeze(0).expand(n_dom, -1))
            all_surface_pts.append(surf_pts_i)
            all_surface_gt.append(surf_gt_i)
            all_surface_z.append(Z[slot_idx].unsqueeze(0).expand(n_surf, -1))

        # Concatenate: surface first, then domain (so pred[:N*n_surf] = surface predictions)
        surface_pts = torch.cat(all_surface_pts, dim=0)  # (N*n_surf, 3)
        surface_gt = torch.cat(all_surface_gt, dim=0)  # (N*n_surf,)
        surface_z = torch.cat(all_surface_z, dim=0)  # (N*n_surf, z_dim)
        # Keep per-slot domain pts for epoch visualization; also concatenate for forward pass
        domain_pts_list = all_domain_pts  # list of per-slot domain tensors (visualised as grey)
        domain_pts = torch.cat(all_domain_pts, dim=0)  # (N*n_dom, 3)
        domain_gt = torch.cat(all_domain_gt, dim=0)  # (N*n_dom,) GT SDF for L1-all loss
        domain_z = torch.cat(all_domain_z, dim=0)  # (N*n_dom, z_dim)

        all_pts_combined = torch.cat([surface_pts, domain_pts], dim=0)
        all_z_combined = torch.cat([surface_z, domain_z], dim=0)

        # Single parallel forward pass (surface + domain together for NaN detection)
        pred = model(all_pts_combined, z=all_z_combined).squeeze()  # (N*(n_surf+n_dom),)
        pred_surface = pred[: num_shapes * n_surf]  # surface slice for MSE

        # Early NaN check — exit before trying autograd.grad on a broken graph
        if not math.isfinite(pred.sum().item() if pred.numel() > 0 else 0.0):
            nan_slots = [
                i for i in range(num_shapes) if not math.isfinite(pred_surface[i * n_surf : (i + 1) * n_surf].sum().item())
            ]
            print(
                f"[latent_pretrain] NaN/Inf in model output at epoch {epoch}, "
                f"shape slots {nan_slots}. "
                "No checkpoint written. "
                "Try reducing learning_rate in "
                "pretraining_latent_config.yaml.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Per-slot L1 on surface pts (diagnostic tracking — comparable to baseline per-slot losses)
        slot_losses: list[float] = []
        for slot_idx in range(num_shapes):
            start = slot_idx * n_surf
            end = start + n_surf
            slot_l1 = l1_fn(
                pred_surface[start:end] / 1000.0,
                surface_gt[start:end] / 1000.0,
            )
            slot_losses.append(slot_l1.item())

        # L1 on ALL pts (surface + domain) — surface first, domain second (mirrors all_pts_combined layout)
        gt_all = torch.cat([surface_gt, domain_gt], dim=0)
        total_loss = l1_fn(pred / 1000.0, gt_all / 1000.0)
        l1_all = total_loss.item()

        # --- Per-band loss instrumentation (monitoring only; backprop still uses total_loss) ---
        # Gated by training_loop.track_band_losses: true in config (default: false).
        loss_tight_item = None
        loss_wide_item = None
        loss_domain_item = None

        if track_band_losses:
            if near_surface_wide_sigma is None:
                # Single-band: all surface points are tight-band
                loss_tight_item = l1_fn(pred_surface / 1000.0, surface_gt / 1000.0).item()
            else:
                # Dual-band: split surface points 50/50 between tight and wide
                n_tight_per_shape = n_surf // 2
                tight_indices = torch.tensor(
                    [s * n_surf + i for s in range(num_shapes) for i in range(n_tight_per_shape)],
                    device=device,
                )
                wide_indices = torch.tensor(
                    [s * n_surf + i for s in range(num_shapes) for i in range(n_tight_per_shape, n_surf)],
                    device=device,
                )
                loss_tight_item = l1_fn(pred_surface[tight_indices] / 1000.0, surface_gt[tight_indices] / 1000.0).item()
                loss_wide_item = l1_fn(pred_surface[wide_indices] / 1000.0, surface_gt[wide_indices] / 1000.0).item()

            if num_shapes * n_dom > 0:
                loss_domain_item = l1_fn(pred[num_shapes * n_surf :] / 1000.0, domain_gt / 1000.0).item()

        # --- Eikonal regularization: ||∇SDF(x)|| = 1 on domain points ---
        # Gated by training_loop.eikonal_weight > 0 (default: 0.0 = disabled).
        # Runs a second forward pass on domain pts with create_graph=True.
        eikonal_loss_item: float | None = None
        if eikonal_weight > 0.0:
            eik_pts = domain_pts.detach().requires_grad_(True)
            eik_pred = model(eik_pts, z=domain_z.detach()).squeeze()
            (eik_grads,) = torch.autograd.grad(eik_pred.sum(), eik_pts, create_graph=True)
            eikonal_loss = ((eik_grads.norm(dim=-1) - 1.0) ** 2).mean()
            eikonal_loss_item = eikonal_loss.item()
            total_loss = total_loss + eikonal_weight * eikonal_loss

        # NaN guard — halt before writing checkpoint
        if not math.isfinite(total_loss.item()):
            nan_slot: int | str = next(
                (i for i, v in enumerate(slot_losses) if not math.isfinite(v)),
                "unknown",
            )
            shape_name = shapes[nan_slot]["shape_type"] if isinstance(nan_slot, int) else "unknown"
            print(
                f"[latent_pretrain] NaN/Inf detected at epoch {epoch}, "
                f"shape slot {nan_slot} ({shape_name}). "
                "No checkpoint written. "
                "Try reducing learning_rate in "
                "pretraining_latent_config.yaml.",
                file=sys.stderr,
            )
            sys.exit(1)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Per-epoch snapshot (b): current trained mesh + current sampled points ---
        if vis_sampling and epoch % vis_cadence == 0 and (epoch_vis is not None or vis_manager is not None):
            _write_epoch_vis_snapshot(
                model=model,
                Z=Z,
                shapes=shapes,
                cached_surface=cached_surface,
                domain_pts_per_slot=domain_pts_list,
                ply_cache=ply_cache,
                epoch=epoch,
                mesh_res=vis_mesh_res,
                outer_prism=outer_prism,
                device=device,
                vis=epoch_vis,
                vis_manager=vis_manager,
                show_target_pts=vis_show_target_pts,
                show_domain_pts=vis_show_domain_pts,
            )
            model.train()  # restore after eval inside snapshot

        if epoch % logging_interval == 0:
            lr_now = scheduler.get_last_lr()[0]

            # --- Gradient probe: ‖∂L/∂z_input‖ vs ‖∂L/∂xyz_input‖ on slot-0 surface pts ---
            model.eval()
            surf_pts_0, surf_gt_0 = cached_surface[0]
            z0_expanded = Z[0].unsqueeze(0).expand(surf_pts_0.size(0), -1)
            z_grad_norm, xyz_grad_norm, z_warn = _compute_input_grad_norms(model, surf_pts_0, z0_expanded, surf_gt_0, device)
            model.train()

            grad_ratio = z_grad_norm / (xyz_grad_norm + 1e-8)
            if z_warn:
                warn_suffix = " [WARN: z not in graph]"
            elif grad_ratio < 0.01:
                warn_suffix = " [WARN: ratio<0.01]"
            else:
                warn_suffix = ""

            eik_suffix = f" | eikonal={eikonal_loss_item:.4f}" if eikonal_loss_item is not None else ""
            tqdm.write(
                f"[latent_pretrain] Epoch {epoch:5d} | total={total_loss.item():.6f} "
                f"| slot_losses={[f'{v:.4f}' for v in slot_losses]} "
                f"| l1_all={l1_all:.6f} | lr={lr_now:.2e}"
                f" | z_grad={z_grad_norm:.2e} | xyz_grad={xyz_grad_norm:.2e}"
                f" | ratio={grad_ratio:.2e}{warn_suffix}{eik_suffix}"
            )
            if use_wandb:
                # Prepare per-band losses for logging
                wandb_dict = {
                    "latent_pretraining/total_loss": total_loss.item(),
                    "latent_pretraining/l1_all": l1_all,
                    "latent_pretraining/lr": lr_now,
                    "latent_pretraining/latent_grad_norm": z_grad_norm,
                    "latent_pretraining/spatial_grad_norm": xyz_grad_norm,
                    "latent_pretraining/grad_norm_ratio": grad_ratio,
                    **{f"latent_pretraining/slot_{i}_loss": v for i, v in enumerate(slot_losses)},
                }
                # Add per-band losses if available
                if loss_tight_item is not None:
                    wandb_dict["latent_pretraining/loss_tight_band"] = loss_tight_item
                if loss_wide_item is not None:
                    wandb_dict["latent_pretraining/loss_wide_band"] = loss_wide_item
                if loss_domain_item is not None:
                    wandb_dict["latent_pretraining/loss_domain"] = loss_domain_item
                if eikonal_loss_item is not None:
                    wandb_dict["latent_pretraining/eikonal_loss"] = eikonal_loss_item

                wandb.log(wandb_dict, step=epoch)

    # --- Save checkpoint ---
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    _shape_names = [s.get("name", str(i)) for i, s in enumerate(shapes)]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "z_pretrained": Z,
            "meta": {
                "config_hash": config_hash,
                "date": datetime.now().isoformat(),
                "epochs": epochs,
                "model_class": model.__class__.__name__,
                "shape_names": _shape_names,
            },
        },
        ckpt_path,
    )
    print(f"[latent_pretrain] SAVE  path={ckpt_path}  hash={config_hash}  epochs={epochs}  shapes={_shape_names}")
    print(f"[latent_pretrain] All outputs → {run_dir}/")
    # Write model_info_dict.json sidecar: model architecture + shape names for human-readable audit
    info_json_path = ckpt_path.parent / "model_info_dict.json"
    with open(info_json_path, "w") as _f:
        json.dump({**model.get_model_info_dict(), "shape_names": _shape_names}, _f, indent=2)
    print(f"[latent_pretrain] Wrote model_info_dict.json → {info_json_path}")

    # --- Log Z spread ---
    pairwise = [(Z[i] - Z[j]).norm().item() for i in range(num_shapes) for j in range(i + 1, num_shapes)]
    if pairwise:
        mean_dist = sum(pairwise) / len(pairwise)
        print(f"[latent_pretrain] Z pairwise distances: min={min(pairwise):.3f}, mean={mean_dist:.3f}, max={max(pairwise):.3f}")

    # --- Circle validity probe ---
    # Scans N equally-spaced angles on the z-circle. A position is "valid" when SDF min < 0
    # (zero crossing possible → marching cubes would produce a mesh). Fast: no marching cubes.
    n_probe_angles: int = int(loop_cfg.get("circle_probe_n_angles", 36))
    if n_probe_angles > 0:
        frac_valid, n_valid_angles, n_total_angles = _compute_circle_validity_fraction(
            model, z_radius, z_dim, outer_prism, device, n_probe_angles
        )
        print(
            f"[latent_pretrain] Circle validity: {n_valid_angles}/{n_total_angles} angles valid "
            f"({frac_valid:.1%})  — target ≥ 85%"
        )
        if use_wandb:
            import wandb  # noqa: PLC0415

            wandb.log({"latent_pretraining/circle_valid_fraction": frac_valid}, step=epochs)

    # --- Final Mesh Extraction + Topology Summary ---
    pretrained_shapes_dir = run_dir / "pretrained_shapes"
    pretrained_shapes_dir.mkdir(parents=True, exist_ok=True)
    topo_wandb: dict[str, Any] = {}
    print("[latent_pretrain] Mesh topology summary (Euler X = 2-2g; genus-0 closed surface -> X=2):")
    print(f"  {'slot':<5} {'shape':<35} {'verts':>7} {'faces':>7} {'comps':>6} {'euler':>6} {'watertight':>11}")
    for slot_idx in range(num_shapes):
        slot_name = shapes[slot_idx].get("name", str(slot_idx))
        slot_mesh_path = pretrained_shapes_dir / f"slot{slot_idx:02d}_{slot_name}.ply"
        try:
            clear_mesh_cache()
            model.set_latent_vector(Z[slot_idx].to(device))
            model.eval()
            _extract_mesh(model, slot_mesh_path, device, vis_mesh_res, outer_prism)
            # --- Topology analysis ---
            m = trimesh.load(str(slot_mesh_path))
            if isinstance(m, trimesh.Trimesh):
                n_comps = len(trimesh.graph.connected_components(m.edges, min_len=1))
                euler = m.euler_number
                wt = m.is_watertight
                print("!!!!Topology analysis:")
                print(
                    f"  {slot_idx:<5} {slot_name:<35} {m.vertices.shape[0]:>7} {m.faces.shape[0]:>7}"
                    f" {n_comps:>6} {euler:>6} {'yes' if wt else 'no':>11}"
                )
                topo_wandb[f"mesh_topo/slot{slot_idx}_{slot_name}/components"] = n_comps
                topo_wandb[f"mesh_topo/slot{slot_idx}_{slot_name}/euler"] = euler
                topo_wandb[f"mesh_topo/slot{slot_idx}_{slot_name}/watertight"] = int(wt)
            else:
                print(f"  {slot_idx:<5} {slot_name:<35} (not a single Trimesh — scene?)")
        except Exception as exc:  # BLE001 not enabled; mesh extraction is non-critical and must not abort the run
            print(
                f"[latent_pretrain] WARNING: mesh extraction for slot {slot_idx} failed: {exc}. "
                "Skipping mesh output — checkpoint is still valid."
            )
    if use_wandb and topo_wandb:
        wandb.log(topo_wandb, step=epochs)

    return model, Z


def _write_epoch_vis_snapshot(
    model: SupportsLatentConcat,
    Z: Tensor,
    shapes: list[dict[str, Any]],
    cached_surface: list[tuple[Tensor, Tensor]],
    domain_pts_per_slot: list[Tensor],
    ply_cache: dict[int, tuple[Tensor, Tensor, float]],
    epoch: int,
    mesh_res: int,
    outer_prism: dict[str, Any],
    device: str,
    vis: SamplingVisualizer | None,
    *,
    vis_manager: CentralVisManager | None = None,
    show_target_pts: str = "pts",
    show_domain_pts: bool = True,
) -> None:
    """Write one epoch snapshot: current trained mesh, target surface, and domain pts.

    For each slot the combined PLY shows training convergence at a glance.
    All tensors are moved to CPU before passing to the visualizer.

    Color coding:
    - **Forest green**: target (either surface-cache pts or target mesh geometry).
    - **Red**: current SDF model surface (marching-cubes vertices, always shown).
    - **Light grey**: domain pts (shown when ``show_domain_pts=True``).

    Args:
        show_target_pts: ``"pts"`` shows surface-cache pts in green (primary points).
            ``"mesh"`` shows the target PLY triangle mesh in green; for non-PLY
            slots falls back to ``"pts"`` behavior.
        show_domain_pts: When ``True``, domain pts are shown in light grey.

    The caller must call ``model.train()`` after this function returns.
    """
    for slot_idx, shape in enumerate(shapes):
        slot_name = shape.get("name", str(slot_idx))
        shape_type: str = shape["shape_type"]

        surf_pts_i, _ = cached_surface[slot_idx]  # target surface cache pts
        domain_pts_i: Tensor | None = domain_pts_per_slot[slot_idx] if show_domain_pts else None

        # Extract current model surface (red) as numpy arrays
        clear_mesh_cache()
        model.set_latent_vector(Z[slot_idx].to(device))
        model.eval()
        mesh_arrays = _extract_mesh(model, None, device, mesh_res, outer_prism, return_mesh=True)
        model_verts_np: np.ndarray | None = mesh_arrays[0] if mesh_arrays is not None else None

        # Determine whether to show the target as a proper triangle mesh (green geometry)
        use_mesh_mode = show_target_pts == "mesh" and shape_type == "ply" and slot_idx in ply_cache

        try:
            if use_mesh_mode:
                # "mesh" mode: target PLY mesh (green), model pts (red), domain pts (grey)
                ply_verts_t, ply_faces_t, _ = ply_cache[slot_idx]
                target_verts_np = ply_verts_t.cpu().numpy()
                target_faces_np = ply_faces_t.cpu().numpy()
                model_pts_tensor = torch.from_numpy(model_verts_np) if model_verts_np is not None else torch.zeros(0, 3)
                # CentralVisManager path (preferred when provided)
                if vis_manager is not None:
                    vis_manager.save_ply_points(
                        f"latent/sampling/slot{slot_idx:02d}_{slot_name}",
                        epoch,
                        model_pts_tensor,
                        secondary_points=domain_pts_i.cpu() if domain_pts_i is not None else None,
                        mesh_verts=target_verts_np,
                        mesh_faces=target_faces_np,
                    )
                if vis is not None:
                    vis.save_snapshot(
                        points_mm=model_pts_tensor,
                        mesh_verts=target_verts_np,
                        mesh_faces=target_faces_np,
                        name=f"slot{slot_idx:02d}_{slot_name}_epoch{epoch:05d}",
                        secondary_points_mm=domain_pts_i.cpu() if domain_pts_i is not None else None,
                        points_colour=_POINT_COLOUR_RED,
                        secondary_colour=_POINT_COLOUR_WHITE,
                        mesh_colour=_POINT_COLOUR_GREEN,
                    )
            else:
                # "pts" mode (or non-PLY slot): target cache pts (green), model pts (red), domain pts (grey)
                if vis_manager is not None:
                    vis_manager.save_ply_points(
                        f"latent/sampling/slot{slot_idx:02d}_{slot_name}",
                        epoch,
                        surf_pts_i.cpu(),
                        secondary_points=torch.from_numpy(model_verts_np) if model_verts_np is not None else None,
                        tertiary_points=domain_pts_i.cpu() if domain_pts_i is not None else None,
                    )
                if vis is not None:
                    vis.save_snapshot(
                        points_mm=surf_pts_i.cpu(),
                        mesh_verts=None,
                        mesh_faces=None,
                        name=f"slot{slot_idx:02d}_{slot_name}_epoch{epoch:05d}",
                        secondary_points_mm=torch.from_numpy(model_verts_np) if model_verts_np is not None else None,
                        tertiary_points_mm=domain_pts_i.cpu() if domain_pts_i is not None else None,
                        points_colour=_POINT_COLOUR_GREEN,
                        secondary_colour=_POINT_COLOUR_RED,
                        tertiary_colour=_POINT_COLOUR_WHITE,
                    )
        except Exception as exc:
            print(f"[latent_pretrain] WARNING: epoch vis save failed for slot {slot_idx} ({slot_name}): {exc}")


def _make_vis_legend(
    config_hash: str,
    vis_cadence: int,
    vis_show_domain_pts: bool,
    surface_resample_cadence: int,
) -> str:
    """Build the legend.txt content for a sampling_vis directory."""
    domain_note = (
        "  Light grey (220, 220, 220): Domain pts (bounding-box uniform, ~50% of batch).\n"
        "      Used for L1-all loss together with the surface pts.\n"
        "      Shown in epoch PLYs because vis_show_domain_pts: true."
        if vis_show_domain_pts
        else "  Domain pts (light grey): HIDDEN because vis_show_domain_pts: false in config."
    )
    cadence_note = f"every {vis_cadence} epochs" if vis_cadence > 0 else "disabled (vis_cadence: 0)"
    surface_note = (
        "static for all epochs (DeepSDF-style: sampled once at training start, reused every epoch)."
        "\n      Config: training_loop.surface_resample_cadence: 0"
        if surface_resample_cadence == 0
        else f"refreshed every {surface_resample_cadence} epochs.\n      Config: training_loop.surface_resample_cadence: {surface_resample_cadence}"
    )
    return f"""Sampling Visualization Legend
==============================
Config hash   : {config_hash}

File naming
-----------
  slot{{N:02d}}_{{name}}_epoch{{E:05d}}.ply
      Written {cadence_note}.
      When vis_show_target_pts: "pts":
        Green points = target surface-cache pts, Red points = current model surface.
      When vis_show_target_pts: "mesh" (PLY slots only):
        Green triangle mesh = target shape, Red points = current model surface.
      Light grey points = domain pts when vis_show_domain_pts: true.
      As training converges red points migrate toward the green target.

  pretrained_shapes/slot{{N:02d}}_{{name}}.ply
      Written once after training.  Final extracted mesh for each slot.

Colour coding
-------------
  Forest green (34, 139, 34): Target shape.
      - "pts" mode : surface-cache points near the target mesh (near-surface
        Gaussian-perturbed pts).  These are the surface half of the L1-all batch.
      - "mesh" mode: target triangle mesh (proper geometry).
      Surface cache : {surface_note}

  Red (220, 30, 30): Current SDF model surface — in EPOCH (*_epochNNNNN.ply)
      files.  Marching-cubes mesh vertices of the model's predicted zero level
      set at that epoch.  Always shown.

  Light grey (220, 220, 220): Domain pts — in EPOCH files when
      vis_show_domain_pts: true.  Bounding-box uniform points (~50% of batch),
      the domain half of the L1-all loss.
      Toggle: vis_show_domain_pts in config output section.

{domain_note}

Loss in pretraining
--------------------------
  L1-all loss: L1( predicted_SDF / 1000, gt_SDF / 1000 ).
     Applied to ALL points (surface + domain combined).
     GREEN (near-surface) + GREY (domain) = complete L1-all batch.
     GT SDF: mesh_sdf (GPU BVH) for PLY; analytic for sphere/cube.
     Surface pts are Gaussian-perturbed from the mesh surface so GT SDF != 0
     everywhere (avoids the degenerate all-zero solution).

  NO smoothness loss in pretraining.  Smoothness applies in the main
  aeroshell optimizer loop only.

DeepSDF vs AISCO sampling (reference)
---------------------------------------
  DeepSDF pre-samples TWO Gaussian bands from surface
  (sigma_1 = 0.005, sigma_2 = 0.0005 normalised [-1,1] space) and
  uniform sphere points, stored offline.  During training the same
  pre-computed dataset is reused every epoch (equivalent to
  surface_resample_cadence: 0 here).
  AISCO uses ONE Gaussian band (sigma = 0.05 x bbox_extent) + domain
  uniform, computed online.  The tighter DeepSDF sigma_2 gives higher
  accuracy very close to the surface; consider reducing noise_sigma in
  _sample_points_for_shape if fine surface detail is needed.
"""
