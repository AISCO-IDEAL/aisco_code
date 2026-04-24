"""Latent-population pretraining for ConcatWIREV2 (Plan 005 / v0.3.0).

Trains model weights so that each latent slot z_i approximates a target SDF
(sphere, cube, or PLY mesh).  All N shapes are batched into a single forward
pass for efficiency.

Entry point: ``pretraining_latent(model, config, device)``
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh
from torch import Tensor
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _compute_eikonal_latent,
    _extract_mesh,
    _sample_points_for_shape,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization import SamplingVisualizer
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.latent_utils import build_unit_circle_Z


def _load_ply_tensors(ply_path: str, device: str) -> tuple[Tensor, Tensor, float]:
    """Load a PLY mesh and return (verts, faces, bbox_extent).

    Args:
        ply_path: Absolute path to the PLY file.
        device: Target device string.

    Returns:
        Tuple of (ply_verts, ply_faces, bbox_extent) where bbox_extent is the
        diagonal of the mesh bounding box in mm.
    """
    raw_mesh = trimesh.load(ply_path)
    if not isinstance(raw_mesh, trimesh.Trimesh):
        raise TypeError(
            f"Expected a Trimesh object when loading '{ply_path}', got {type(raw_mesh)}. "
            "Ensure the PLY file contains a single solid mesh."
        )
    verts_np = np.array(raw_mesh.vertices, dtype="float32")
    faces_np = np.array(raw_mesh.faces, dtype="int64")
    bbox_diag = float(np.linalg.norm(verts_np.max(axis=0) - verts_np.min(axis=0)))
    ply_verts = torch.tensor(verts_np, device=device)
    ply_faces = torch.tensor(faces_np, device=device)
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
        return {"ply_verts": ply_verts, "ply_faces": ply_faces, "bbox_extent": bbox_extent}
    raise ValueError(
        f"Unknown shape_type '{shape_type}' in latent_population slot {slot_idx}. Valid values: 'sphere', 'cube', 'ply'."
    )


def _compute_pretrain_config_hash(config: dict[str, Any], model: SupportsLatentConcat) -> str:
    """Compute a stable hash over model architecture and training config.

    The hash covers: model architecture (via get_model_info_dict), latent_population,
    training_loop, z_placement, and z_radius. Two runs with identical parameters
    produce the same hash; any parameter change produces a different hash.

    Args:
        config: Dict matching the pretraining_latent_config.yaml schema.
        model: Model whose architecture is included in the hash.

    Returns:
        10-character hex hash string.
    """
    key: dict[str, Any] = {
        "model": dict_hashing(model.get_model_info_dict()),
        "latent_population": config["latent_population"],
        "training_loop": config["training_loop"],
        "z_placement": config.get("z_placement", "unit_circle"),
        "z_radius": config.get("z_radius", 1.0),
    }
    return dict_hashing(key)


def _resolve_checkpoint_path(config: dict[str, Any], model: SupportsLatentConcat) -> Path:
    """Return a hash-tagged checkpoint path derived from the base path in config.

    For example, if config specifies ``pretrained_latent.pth``, the resolved path
    is ``pretrained_latent_<hash>.pth`` in the same directory. Different configs
    resolve to different files; identical configs resolve to the same file.

    Args:
        config: Dict matching the pretraining_latent_config.yaml schema.
        model: Model whose architecture is included in the hash.

    Returns:
        Path with the config hash embedded in the stem.
    """
    base = Path(config["output"]["checkpoint_path"])
    config_hash = _compute_pretrain_config_hash(config, model)
    return base.parent / f"{base.stem}_{config_hash}{base.suffix}"


def pretraining_latent(  # noqa: PLR0912, PLR0915  # training loop is intrinsically sequential; decomposition would obscure control flow
    model: SupportsLatentConcat,
    config: dict[str, Any],
    device: str,
    *,
    use_wandb: bool = False,
    vis_sampling: bool = False,
) -> tuple[SupportsLatentConcat, Tensor]:
    """Run latent-population pretraining for a SupportsLatentConcat model.

    Trains model weights so that each latent slot z_i (placed on a unit circle)
    approximates a corresponding target SDF (sphere / cube / PLY mesh).

    Fast-resume: if a valid checkpoint exists at ``config["output"]["checkpoint_path"]``
    and ``config["force_pretraining"]`` is False, the checkpoint is loaded and returned
    without training.

    Args:
        model: ConcatWIREV2 (or any SupportsLatentConcat model).
        config: Dict matching the pretraining_latent_config.yaml schema.
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).

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

    ckpt_path = _resolve_checkpoint_path(config, model)
    config_hash = _compute_pretrain_config_hash(config, model)
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
        model.load_state_dict(ckpt["model_state_dict"])
        Z: Tensor = ckpt["z_pretrained"].to(device)
        print(f"[latent_pretrain] Fast resume: loaded checkpoint from '{ckpt_path}'")
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
            ply_cache[slot_idx] = _load_ply_tensors(str(shape["ply_path"]), device)
            print(f"[latent_pretrain] Slot {slot_idx}: loaded PLY '{shape['ply_path']}'")

    # --- Build unit-circle Z ---
    z_dim: int = model.get_model_info_dict()["z_dim"]
    z_radius: float = float(config.get("z_radius", 1.0))
    Z = build_unit_circle_Z(num_shapes, z_dim, z_radius, device)

    # --- Training hyper-parameters ---
    loop_cfg: dict[str, Any] = config["training_loop"]
    epochs: int = int(loop_cfg["epochs"])
    lr: float = float(loop_cfg["learning_rate"])
    batch_size_per_shape: int = int(loop_cfg["batch_size_per_shape"])
    print_interval: int = int(loop_cfg["print_interval"])
    eikonal_weight: float = float(loop_cfg["eikonal_weight"])

    # --- Output / visualisation config (read once, shared between loop and post-training) ---
    output_cfg: dict[str, Any] = config.get("output", {})
    vis_dir: Path = ckpt_path.parent / f"sampling_vis_{config_hash}"
    vis_cadence: int = int(output_cfg.get("vis_cadence", 0))  # 0 = disabled
    vis_mesh_res: int = int(output_cfg.get("mesh_resolution", 64))
    vis_mesh_box: int = int(output_cfg.get("mesh_box_mm", 2000))
    vis_eikonal_pts: bool = bool(output_cfg.get("vis_eikonal_pts", True))

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    mse_fn = torch.nn.MSELoss()

    print(
        f"[latent_pretrain] Starting: N={num_shapes} shapes, epochs={epochs}, "
        f"lr={lr}, batch_size_per_shape={batch_size_per_shape}, device={device}",
        flush=True,
    )

    # --- Per-epoch visualizer (b): current trained mesh + loss sample points ---
    epoch_vis: SamplingVisualizer | None = None
    if vis_sampling and vis_cadence > 0:
        epoch_vis = SamplingVisualizer(output_dir=vis_dir, is_static=False)
        print(
            f"[latent_pretrain] Per-epoch vis enabled: cadence={vis_cadence}, mesh_res={vis_mesh_res}, mesh_box=±{vis_mesh_box} mm → {vis_dir}"
        )

    # --- Training loop ---
    for epoch in tqdm(range(epochs), desc="latent_pretrain", unit="ep"):
        model.train()
        optimizer.zero_grad()

        # Assemble parallel batch: all N shapes in one forward pass
        all_pts: list[Tensor] = []
        all_gt: list[Tensor] = []
        all_z: list[Tensor] = []

        for slot_idx, shape in enumerate(shapes):
            shape_params = _build_shape_params(shape, ply_cache, slot_idx)
            pts_i, gt_i = _sample_points_for_shape(shape["shape_type"], shape_params, batch_size_per_shape, device)
            all_pts.append(pts_i)
            all_gt.append(gt_i)
            all_z.append(Z[slot_idx].unsqueeze(0).expand(batch_size_per_shape, -1))

        pts = torch.cat(all_pts, dim=0)  # (N*B, 3)
        gt = torch.cat(all_gt, dim=0)  # (N*B,)
        z_expanded = torch.cat(all_z, dim=0)  # (N*B, z_dim)

        # Single parallel forward pass
        pred = model(pts, z=z_expanded).squeeze()  # (N*B,)

        # Early NaN check — exit before trying autograd.grad on a broken graph
        if not math.isfinite(pred.sum().item() if pred.numel() > 0 else 0.0):
            nan_slots = [
                i
                for i in range(num_shapes)
                if not math.isfinite(pred[i * batch_size_per_shape : (i + 1) * batch_size_per_shape].sum().item())
            ]
            print(
                f"[latent_pretrain] NaN/Inf in model output at epoch {epoch}, "
                f"shape slots {nan_slots}. "
                "No checkpoint written. "
                "Try reducing learning_rate or eikonal_weight in "
                "pretraining_latent_config.yaml.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Per-slot MSE (normalised to metres to avoid scale issues)
        slot_losses: list[float] = []
        total_mse = torch.zeros(1, device=device).squeeze()
        for slot_idx in range(num_shapes):
            start = slot_idx * batch_size_per_shape
            end = start + batch_size_per_shape
            slot_mse = mse_fn(
                pred[start:end] / 1000.0,
                gt[start:end] / 1000.0,
            )
            slot_losses.append(slot_mse.item())
            total_mse = total_mse + slot_mse

        eikonal = _compute_eikonal_latent(model, pts, z_expanded, device)
        total_loss = total_mse / num_shapes + eikonal_weight * eikonal

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
                "Try reducing learning_rate or eikonal_weight in "
                "pretraining_latent_config.yaml.",
                file=sys.stderr,
            )
            sys.exit(1)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # --- Per-epoch snapshot (b): current trained mesh + current sampled points ---
        if vis_sampling and epoch_vis is not None and epoch % vis_cadence == 0:
            _write_epoch_vis_snapshot(
                model=model,
                Z=Z,
                shapes=shapes,
                ply_cache=ply_cache,
                epoch=epoch,
                mesh_res=vis_mesh_res,
                mesh_box=vis_mesh_box,
                device=device,
                vis=epoch_vis,
                batch_size_per_shape=batch_size_per_shape,
                show_domain_pts=vis_eikonal_pts,
            )
            model.train()  # restore after eval inside snapshot

        if epoch % print_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            tqdm.write(
                f"[latent_pretrain] Epoch {epoch:5d} | total={total_loss.item():.6f} "
                f"| slot_losses={[f'{v:.4f}' for v in slot_losses]} "
                f"| eikonal={eikonal.item():.6f} | lr={lr_now:.2e}"
            )
            if use_wandb:
                import wandb  # noqa: PLC0415

                wandb.log(
                    {
                        "total_loss": total_loss.item(),
                        "eikonal": eikonal.item(),
                        "lr": lr_now,
                        **{f"slot_{i}_loss": v for i, v in enumerate(slot_losses)},
                    },
                    step=epoch,
                )

    # --- Save checkpoint ---
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "z_pretrained": Z}, ckpt_path)
    print(f"[latent_pretrain] Saved checkpoint to '{ckpt_path}' (model_state_dict + z_pretrained shape={tuple(Z.shape)})")

    # --- Log Z spread ---
    pairwise = [(Z[i] - Z[j]).norm().item() for i in range(num_shapes) for j in range(i + 1, num_shapes)]
    if pairwise:
        mean_dist = sum(pairwise) / len(pairwise)
        print(f"[latent_pretrain] Z pairwise distances: min={min(pairwise):.3f}, mean={mean_dist:.3f}, max={max(pairwise):.3f}")

    # --- Optional per-slot mesh extraction ---
    mesh_res = output_cfg.get("mesh_resolution")
    mesh_box = output_cfg.get("mesh_box_mm")
    if mesh_res and mesh_box:
        for slot_idx in range(num_shapes):
            slot_name = shapes[slot_idx].get("name", str(slot_idx))
            slot_ckpt = ckpt_path.parent / f"{ckpt_path.stem}_slot{slot_idx}_{slot_name}.pth"
            try:
                model.set_latent_vector(Z[slot_idx].to(device))
                model.eval()
                _extract_mesh(model, slot_ckpt, device, int(mesh_res), int(mesh_box))
            except Exception as exc:  # BLE001 not enabled; mesh extraction is non-critical and must not abort the run
                print(
                    f"[latent_pretrain] WARNING: mesh extraction for slot {slot_idx} failed: {exc}. "
                    "Skipping mesh output — checkpoint is still valid."
                )

    # --- Optional sampling visualization (a): target mesh + representative sample ---
    if vis_sampling:
        if epoch_vis is None:  # no per-epoch vis created yet (vis_cadence == 0)
            static_vis = SamplingVisualizer(output_dir=vis_dir, is_static=False)
            static_vis.write_legend(_make_vis_legend(config_hash, vis_cadence, vis_eikonal_pts))
        else:
            static_vis = epoch_vis  # re-use same dir and legend already written
        _visualize_pretraining_sampling(
            shapes=shapes,
            ply_cache=ply_cache,
            device=device,
            vis=static_vis,
            batch_size_per_shape=batch_size_per_shape,
            show_domain_pts=vis_eikonal_pts,
        )

    return model, Z


def _visualize_pretraining_sampling(
    shapes: list[dict[str, Any]],
    ply_cache: dict[int, tuple[Tensor, Tensor, float]],
    device: str,
    vis: SamplingVisualizer,
    batch_size_per_shape: int,
    *,
    show_domain_pts: bool = True,
) -> None:
    """Write target-shape visualizations for each pretraining slot after training.

    For each slot, produces a PLY file showing:
    - The **target mesh** (dark blue) — original PLY vertices, or absent for sphere/cube.
    - The **surface sample points** (forest green) — surface-concentrated sub-population.
    - The **domain sample points** (light grey) — uniform bounding-box sub-population,
      only written when ``show_domain_pts=True``.
    """
    print(f"[latent_pretrain] Writing target sampling visualization → {vis.output_dir}")

    for slot_idx, shape in enumerate(shapes):
        slot_name = shape.get("name", str(slot_idx))
        shape_type: str = shape["shape_type"]
        shape_params = _build_shape_params(shape, ply_cache, slot_idx)

        try:
            surface_pts, domain_pts = _vis_split_sample(shape_type, shape_params, batch_size_per_shape, device)
        except Exception as exc:  # visualization is non-critical
            print(f"[latent_pretrain] WARNING: sampling failed for slot {slot_idx} ({slot_name}): {exc}")
            continue

        # --- Target mesh (PLY slots only) ---
        mesh_verts: np.ndarray | None = None
        mesh_faces: np.ndarray | None = None
        if shape_type == "ply" and slot_idx in ply_cache:
            ply_verts_t, ply_faces_t, _ = ply_cache[slot_idx]
            mesh_verts = ply_verts_t.cpu().numpy()
            mesh_faces = ply_faces_t.cpu().numpy()

        try:
            vis.save_snapshot(
                points_mm=surface_pts,
                mesh_verts=mesh_verts,
                mesh_faces=mesh_faces,
                name=f"slot{slot_idx:02d}_{slot_name}_{shape_type}_target",
                secondary_points_mm=domain_pts if show_domain_pts else None,
            )
        except Exception as exc:  # visualization is non-critical
            print(f"[latent_pretrain] WARNING: visualization failed for slot {slot_idx} ({slot_name}): {exc}")


def _extract_current_mesh_arrays(
    model: SupportsLatentConcat,
    z_slot: Tensor,
    mesh_res: int,
    mesh_box: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the current SDF mesh for a single latent slot as numpy arrays.

    Sets the model's active latent vector to ``z_slot``, runs marching cubes via
    kaolin, and returns the mesh.  The model is set to eval mode internally;
    the caller is responsible for restoring train mode afterwards.

    Returns:
        ``(verts_np, faces_np)`` as float32 / int64 numpy arrays, or ``None``
        if kaolin is unavailable or mesh extraction fails.
    """
    try:
        from subprojects.wp4_aeroshell_optimizer.v2.src.utils.kaolin_conversions import (  # noqa: PLC0415
            get_diff_mesh,
        )
    except ImportError:
        return None

    try:
        model.set_latent_vector(z_slot.to(device))
        model.eval()
        box_dim = [-mesh_box, mesh_box, -mesh_box, mesh_box, -mesh_box, mesh_box]
        verts, faces = get_diff_mesh(model, grid_res=mesh_res, box_dim=box_dim, device=device)
        return verts.detach().cpu().numpy().astype("float32"), faces.detach().cpu().numpy().astype("int64")
    except Exception as exc:
        print(f"[latent_pretrain] WARNING: per-epoch mesh extraction failed: {exc}")
        return None


def _write_epoch_vis_snapshot(
    model: SupportsLatentConcat,
    Z: Tensor,
    shapes: list[dict[str, Any]],
    ply_cache: dict[int, tuple[Tensor, Tensor, float]],
    epoch: int,
    mesh_res: int,
    mesh_box: int,
    device: str,
    vis: SamplingVisualizer,
    batch_size_per_shape: int,
    *,
    show_domain_pts: bool = True,
) -> None:
    """Write one epoch snapshot: current trained mesh + current loss sample points.

    For each slot, extracts the current SDF mesh via marching cubes and pairs it
    with a fresh sample batch split by sub-population:
    - **Forest green** (primary): surface-concentrated samples (quality-of-fit region).
    - **Light grey** (secondary, optional): domain / bounding-box uniform samples
      that also feed the eikonal term; omitted when ``show_domain_pts=False``.

    The caller must call ``model.train()`` after this function returns.
    """
    for slot_idx, shape in enumerate(shapes):
        slot_name = shape.get("name", str(slot_idx))
        shape_type: str = shape["shape_type"]
        shape_params = _build_shape_params(shape, ply_cache, slot_idx)

        try:
            surface_pts, domain_pts = _vis_split_sample(shape_type, shape_params, batch_size_per_shape, device)
        except Exception as exc:
            print(f"[latent_pretrain] WARNING: epoch vis sampling failed for slot {slot_idx} ({slot_name}): {exc}")
            continue

        mesh_arrays = _extract_current_mesh_arrays(model, Z[slot_idx], mesh_res, mesh_box, device)
        mesh_verts_ep = mesh_arrays[0] if mesh_arrays is not None else None
        mesh_faces_ep = mesh_arrays[1] if mesh_arrays is not None else None

        try:
            vis.save_snapshot(
                points_mm=surface_pts,
                mesh_verts=mesh_verts_ep,
                mesh_faces=mesh_faces_ep,
                name=f"slot{slot_idx:02d}_{slot_name}_epoch{epoch:05d}",
                secondary_points_mm=domain_pts if show_domain_pts else None,
            )
        except Exception as exc:
            print(f"[latent_pretrain] WARNING: epoch vis save failed for slot {slot_idx} ({slot_name}): {exc}")


def _vis_split_sample(
    shape_type: str,
    shape_params: dict[str, Any],
    batch_size: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """Sample for visualization, returning ``(surface_pts, domain_pts)`` separately.

    Calls ``_sample_points_for_shape`` and splits by the known sub-population
    boundary: the first ``n_domain`` rows are always uniform domain points; the
    remainder are surface-concentrated (surface ± jitter for PLY, face-uniform +
    near-surface for sphere/cube).

    Returns:
        ``(surface_pts, domain_pts)`` each as ``(K, 3)`` Tensor.
    """
    pts, _ = _sample_points_for_shape(shape_type, shape_params, batch_size, device)
    n_domain = batch_size // 2 if shape_type == "ply" else int(batch_size * 0.50)
    domain_pts = pts[:n_domain]
    surface_pts = pts[n_domain:]
    return surface_pts, domain_pts


def _make_vis_legend(config_hash: str, vis_cadence: int, vis_eikonal_pts: bool) -> str:
    """Build the legend.txt content for a sampling_vis directory."""
    eikonal_note = (
        "  Light grey (220, 220, 220): Domain sample points (bounding-box uniform, 50 % of batch).\n"
        "      Contribute to BOTH the MSE loss and the eikonal loss.\n"
        "      Shown because vis_eikonal_pts: true in config."
        if vis_eikonal_pts
        else "  Domain points (light grey): HIDDEN because vis_eikonal_pts: false in config."
    )
    cadence_note = f"every {vis_cadence} epochs" if vis_cadence > 0 else "disabled (vis_cadence: 0)"
    return f"""Sampling Visualization Legend
==============================
Config hash : {config_hash}

File naming
-----------
  slot{{N:02d}}_{{name}}_{{shape_type}}_target.ply
      Written once after training.
      Shows the original target shape and a representative training sample.

  slot{{N:02d}}_{{name}}_epoch{{E:05d}}.ply
      Written {cadence_note}.
      Shows the CURRENT trained SDF mesh and a fresh training sample at that epoch.

Companion files (*_mesh_only.ply)
---------------------------------
  Each entry above has a companion *_mesh_only.ply (pure triangle mesh, no
  point cloud).  Open this in MeshLab / CloudCompare to see full mesh geometry.
  The combined .ply is a point cloud; mesh vertices appear as blue dots.

Colour coding
-------------
  Dark blue  (0,   0,   139): Mesh vertices.
      - target PLYs : vertices of the original target PLY file.
      - epoch PLYs  : current trained SDF mesh (marching cubes at that epoch).
      NOTE: mesh vertices are rendered as POINTS in the combined .ply file.
            Open the companion *_mesh_only.ply for the full triangle mesh view.
            Blue dots only at front/back of a car model is CORRECT: those
            regions have denser triangulation (more curvature), hence more vertices.

  Forest green (34, 139, 34): Surface-concentrated sample points.
      - For PLY targets : 50 % of batch — trimesh.sample.sample_surface() +
                          Gaussian jitter (sigma = 0.05 x bbox_extent).
      - For sphere/cube : 50 % of batch — exact surface (30 %) +
                          near-surface band (20 %, |SDF| < 0.15-0.3 x scale).
      Role: primary quality-of-fit region for the MSE loss.

{eikonal_note}

Loss terms in pretraining
--------------------------
  1. MSE loss (quality of fit): L2 between predicted SDF and ground-truth SDF,
     normalised to metres.  All sample points (green + grey) contribute.
     GT SDF computed by: mesh_sdf (GPU) for PLY; analytic for sphere/cube.

  2. Eikonal loss: penalises |grad SDF| != 1.  Uses the SAME sample batch as
     the MSE loss (green + grey points together).  There is no separate
     eikonal-specific point set in the current implementation.
     Toggle grey points off with vis_eikonal_pts: false in config.output.

  NO smoothness loss in pretraining.  Smoothness is applied only in the
  main aeroshell optimizer loop, not here.

DeepSDF vs AISCO sampling (reference)
---------------------------------------
  DeepSDF uses TWO near-surface Gaussian scales (sigma_1=0.0025,
  sigma_2=0.00025 in normalised [-1,1] space) + unit-sphere uniform samples,
  pre-computed offline.  AISCO uses ONE scale (0.05 x bbox_extent) + uniform
  domain, computed online per epoch.  The tighter DeepSDF sigma_2 band
  provides higher SDF accuracy very close to the surface.
"""
