#!/usr/bin/env python3
"""Render high-quality PNGs of the aeroshell assembly with optional solar layers.

Supports two input modes:

1. **Checkpoint mode** (``--checkpoint path/to/epoch_1400.pth``):
   - Extracts the aeroshell mesh via marching cubes at the given polar latent
     coordinates (``--radius``, ``--angle_deg``).
   - Optionally loads the inner assembly from co-located PLY artifacts.
   - Computes solar binary mask and shading in-process (heavy; requires GPU).

2. **Artifact mode** (``--run_dir path/to/run_dir --epoch 1400``):
   - Loads pre-built PLY files from ``mesh_assembly_v4/``, ``solar/binary_mask/``,
     and ``solar/shading/`` next to a training run.
   - No model required; purely post-process file transforms.

One PNG per camera view defined in ``postpro_vis_config.yaml`` is written to the
configured output directory.

Missing layers emit ``warnings.warn`` and are silently skipped (warning-first,
not hard fail).

Usage
-----
    # Checkpoint mode:
    python aeroshell_png.py \\
        --checkpoint /cluster/work/.../models/run_epoch1400.pth \\
        --radius 2.0 --angle_deg 45.0

    # Artifact mode:
    python aeroshell_png.py \\
        --run_dir /cluster/work/.../my_run \\
        --epoch 1400

    # Override config:
    python aeroshell_png.py --checkpoint ... --config /path/to/my_config.yaml
"""

from __future__ import annotations

import glob as _glob
import importlib.util
import json
import math
import os
import re
import sys
import types
import warnings
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

# MUST be set before any OpenGL / pyrender import.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse

import numpy as np
import pyrender
import torch
import trimesh
import yaml
from PIL import Image
from torch import Tensor

_DEFAULT_CONFIG = Path(__file__).parent / "postpro_vis_config.yaml"

_MODELS_DIR = Path(__file__).parents[1] / "wp4_aeroshell_optimizer" / "v2" / "src" / "models"

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    """Load YAML config from *path*.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Pass --config to specify a config path or create postpro_vis_config.yaml next to the script."
        )
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Navigate a nested dict via *keys* with a safe fallback to *default*.

    Args:
        cfg: Nested config dict.
        *keys: Sequence of string keys to traverse.
        default: Value returned when any key is missing or the node is None.

    Returns:
        The value at cfg[keys[0]][keys[1]]…, or *default* if any node is missing.
    """
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Color helpers (same as latent_rollout_gif.py)
# ---------------------------------------------------------------------------


def _parse_color_rgb(c: str | list) -> tuple[int, int, int]:
    """Parse '#RRGGBB' hex string or int list [R, G, B] → (R, G, B) ints 0-255.

    Args:
        c: Hex color string or list of three ints.

    Returns:
        Tuple of (R, G, B) integers in 0-255 range.
    """
    if isinstance(c, (list, tuple)):
        return (int(c[0]), int(c[1]), int(c[2]))
    h = str(c).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _parse_color_rgba_float(c: str | list, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Parse '#RRGGBB' hex string or int list → (r, g, b, a) floats 0.0-1.0.

    Args:
        c: Hex color string or list of three ints.
        alpha: Alpha override (0.0–1.0).

    Returns:
        Tuple of (r, g, b, a) floats.
    """
    r, g, b = _parse_color_rgb(c)
    return (r / 255.0, g / 255.0, b / 255.0, alpha)


# ---------------------------------------------------------------------------
# Polar latent helpers
# ---------------------------------------------------------------------------


def _polar_to_z(radius: float, angle_deg: float, z_dim: int, device: torch.device | None = None) -> Tensor:
    """Convert polar latent coordinates to a z-vector.

    Convention (confirmed in analysis doc 003):
      z₀ = r · cos(θ),  z₁ = r · sin(θ),  z₂…z_{d-1} = 0
    where θ = 0° is +z₀ axis and increases CCW in the (z₀, z₁) plane.

    Args:
        radius: Latent norm (distance from origin).
        angle_deg: Angle in degrees from +z₀ axis, CCW.
        z_dim: Total latent dimension.
        device: Torch device for the output tensor (default: CPU).

    Returns:
        Float32 tensor of shape (z_dim,).
    """
    dev = device or torch.device("cpu")
    z = torch.zeros(z_dim, dtype=torch.float32, device=dev)
    angle_rad = math.radians(angle_deg)
    z[0] = radius * math.cos(angle_rad)
    z[1] = radius * math.sin(angle_rad)
    return z


# ---------------------------------------------------------------------------
# Model loading (identical pattern to latent_rollout_gif.py)
# ---------------------------------------------------------------------------


def _load_module_direct(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a .py file directly by path, registering it in sys.modules.

    Bypasses the package __init__.py chain so no CUDA extensions are triggered.
    Idempotent: returns the cached module if already loaded.

    Args:
        module_name: Fully-qualified module name used as sys.modules key.
        file_path: Absolute path to the .py file.

    Returns:
        The loaded module.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]  # type: ignore[return-value]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader, f"Cannot load {file_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _ensure_wp4_model_imports() -> None:
    """Pre-populate sys.modules so that importing concat_WIRE_v5 never triggers
    the v2 package __init__.py (which requires the uncompiled fixed_renderer
    CUDA extension).

    Strategy
    --------
    1. Register stub Modules for every parent package so Python finds them in
       sys.modules and does NOT re-run any __init__.py.
    2. Register a minimal BaseModel stub for model_baseclass.
    3. Directly load concat_baseclass, WIRE_v5, and concat_WIRE_v5 from their
       source files, bypassing the package import machinery entirely.
    """
    # Stub parent packages for both models and sampling_visualization paths.
    # _MODELS_DIR is already the canonical path (subprojects/wp4.../models); use
    # its parent so we never have to reconstruct the path independently.
    _src_dir = _MODELS_DIR.parent  # subprojects/wp4_aeroshell_optimizer/v2/src
    _real_paths: dict[str, str] = {
        "subprojects.wp4_aeroshell_optimizer.v2.src": str(_src_dir),
        "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization": str(
            _src_dir / "sampling_visualization"
        ),
    }
    for pkg in [
        "subprojects.wp4_aeroshell_optimizer",
        "subprojects.wp4_aeroshell_optimizer.v2",
        "subprojects.wp4_aeroshell_optimizer.v2.src",
        "subprojects.wp4_aeroshell_optimizer.v2.src.models",
        "subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization",
    ]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            if pkg in _real_paths:
                # Set __path__ to the real directory so Python can discover submodules.
                m.__path__ = [_real_paths[pkg]]  # type: ignore[attr-defined]
            else:
                m.__path__ = []  # type: ignore[attr-defined]
            sys.modules[pkg] = m

    mb_key = "subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass"
    if mb_key not in sys.modules:
        from torch import nn  # noqa: PLC0415

        mb_stub = types.ModuleType(mb_key)

        class _BaseModel(nn.Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002  # stub absorbs all args
                super().__init__()

        mb_stub.BaseModel = _BaseModel  # type: ignore[attr-defined]
        sys.modules[mb_key] = mb_stub

    for mod_name, fname in [
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass", "concat_baseclass.py"),
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5", "WIRE_v5.py"),
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5", "concat_WIRE_v5.py"),
    ]:
        _load_module_direct(mod_name, _MODELS_DIR / fname)


def _build_model(info: dict[str, Any], device: torch.device) -> Any:
    root = str(Path(__file__).parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)

    _ensure_wp4_model_imports()

    ConcatWIREV5 = sys.modules["subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5"].ConcatWIREV5  # type: ignore[attr-defined]

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


def _find_model_info(ckpt_path: Path) -> Path:
    """Walk up the directory tree to find model_info_dict.json or Architecture_*.json.

    Args:
        ckpt_path: Path to the .pth checkpoint file.

    Returns:
        Path to the model info JSON file.
    """
    search = ckpt_path.parent
    for _ in range(6):
        candidate = search / "model_info_dict.json"
        if candidate.exists():
            return candidate
        arch_matches = _glob.glob(str(search / "Architecture_*.json"))
        if arch_matches:
            return Path(arch_matches[0])
        search = search.parent
    sys.exit(
        f"ERROR: could not find model_info_dict.json or Architecture_*.json in any ancestor of {ckpt_path.parent}\n"
        "Make sure the checkpoint path is inside a valid WP4 run directory."
    )


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[Any, int, float, Tensor | None]:
    """Load a ConcatWIREV5 model from a .pth checkpoint.

    Args:
        ckpt_path: Path to the checkpoint file.
        device: Torch device to load onto.

    Returns:
        (model, z_dim, z_radius, Z_pretrained) where Z_pretrained is a tensor of
        pre-trained latent vectors or None if not stored in the checkpoint.
    """
    info_path = _find_model_info(ckpt_path)
    print(f"[info] using model config: {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    print(f"[load] {info['name']}  z_dim={info['z_dim']}  hidden={info['hidden_features']}x{info['hidden_layers']}")

    model = _build_model(info, device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state_dict)
    model.eval()

    Z: Tensor | None = None
    if isinstance(raw, dict) and "z_pretrained" in raw:
        Z = raw["z_pretrained"].to(device)
        for i, z in enumerate(Z):
            r = float(torch.norm(z).item())
            print(f"  slot {i}: z=[{z[0].item():+.4f}, {z[1].item():+.4f}]  r={r:.4f}")

    z_dim: int = info["z_dim"]
    z_radius = 2.0
    if Z is not None and Z.shape[0] > 1:
        radii = [float(torch.norm(Z[i]).item()) for i in range(1, Z.shape[0])]
        z_radius = float(sum(radii) / len(radii))
    print(f"[load] z_radius = {z_radius:.4f}")

    return model, z_dim, z_radius, Z


# ---------------------------------------------------------------------------
# Epoch parsing
# ---------------------------------------------------------------------------


def _parse_epoch_from_stem(stem: str) -> int | None:
    """Extract epoch number from a checkpoint stem like 'run_name_epoch1400'.

    Args:
        stem: Filename stem (without extension).

    Returns:
        Epoch integer, or None if not found.
    """
    m = re.search(r"_epoch(\d+)$", stem)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------


def _load_prism(scale: float = 1.0) -> dict[str, tuple[float, float]]:
    """Return outer-prism bounds in mm, optionally scaled around each axis centre.

    Single source of truth is WP2 ``get_outer_prism_bounds_mm()``.  The
    *scale* factor is applied via ``scale_outer_prism()`` so each axis
    half-span is multiplied while the centre is preserved.

    Args:
        scale: Scale factor for the prism half-spans (1.0 = no change).

    Returns:
        Dict with keys ``'x'``, ``'y'``, ``'z'``, each a ``(lo, hi)`` tuple in mm.
    """
    from subprojects.wp2_assembler.version4.utils.reg_imp import (  # noqa: PLC0415
        get_outer_prism_bounds_mm,
        scale_outer_prism,
    )

    prism = get_outer_prism_bounds_mm()
    if scale != 1.0:
        prism = scale_outer_prism(prism, scale)
    return prism


def build_extractor(
    device: torch.device,
    resolution: int,
    prism: dict[str, tuple[float, float]],
    bbox_padding: float = 0.0,
) -> Any:
    """Build a MeshSDFExtractor configured for the WP4 aeroshell prism.

    Args:
        device: Torch device for grid evaluation.
        resolution: Marching-cubes grid resolution per axis.
        prism: Outer-prism bounds dict from ``_load_prism()``.
        bbox_padding: Expand each lateral/top face by this many mm (Z floor
            is never padded — car rests on the ground).

    Returns:
        Configured MeshSDFExtractor instance.
    """
    from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config  # noqa: PLC0415

    cfg = load_mesh_extraction_config()
    cfg = dc_replace(cfg, device=str(device))
    extractor = MeshSDFExtractor(config=cfg)
    extractor.set_grid(
        bbox=[
            [prism["x"][0] - bbox_padding, prism["x"][1] + bbox_padding],
            [prism["y"][0] - bbox_padding, prism["y"][1] + bbox_padding],
            [prism["z"][0], prism["z"][1] + bbox_padding],
        ],
        resolution=resolution,
    )
    return extractor


def extract_mesh(
    model: Any,
    extractor: Any,
    z_vec: Tensor,
    z_dim: int,
    device: torch.device,
) -> trimesh.Trimesh | None:
    """Extract an aeroshell mesh via marching cubes at the given latent vector.

    Args:
        model: ConcatWIREV5 model with ``set_latent_vector`` method.
        extractor: MeshSDFExtractor instance.
        z_vec: Latent vector of shape (z_dim,).
        z_dim: Latent dimension (used for validation).
        device: Compute device.

    Returns:
        Extracted trimesh.Trimesh, or None if extraction fails / mesh is empty.
    """
    from subprojects.wp5.unified_mesh_extraction import clear_mesh_cache  # noqa: PLC0415

    assert z_vec.shape == (z_dim,), f"z_vec shape {z_vec.shape} does not match z_dim={z_dim}"
    model.set_latent_vector(z_vec.to(device))
    clear_mesh_cache()

    def _sdf(pts: Tensor) -> Tensor:
        return model(pts)

    try:
        verts, faces = extractor.extract_mesh(_sdf)
    except Exception as exc:
        print(f"  WARN: mesh extraction failed: {exc}")
        return None

    if verts is None or verts.numel() == 0:
        return None

    return trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())


# ---------------------------------------------------------------------------
# Assembly loading
# ---------------------------------------------------------------------------


def load_inner_assembly(run_dir: Path, epoch_num: int) -> trimesh.Trimesh | None:
    """Load the inner assembly from slot_00 of mesh_assembly_v4.

    Follows the same split-and-filter logic as latent_rollout_gif.py:
      - Large component (aeroshell) → skipped
      - Small components > 12 verts  → inner assembly
      - 12-vert bbox corner markers  → skipped

    Args:
        run_dir: WP4 run directory (parent of ``mesh_assembly_v4/``).
        epoch_num: Epoch number used to select the correct PLY epoch directory.

    Returns:
        Concatenated inner assembly mesh, or None if not found.
    """
    slot_path = run_dir / "mesh_assembly_v4" / f"epoch_{epoch_num:05d}" / "slot_00_mesh.ply"
    if not slot_path.exists():
        warnings.warn(
            f"[aeroshell_png] inner_assembly: PLY not found — {slot_path}. Skipping inner_assembly layer.",
            UserWarning,
            stacklevel=2,
        )
        return None

    full_mesh: trimesh.Trimesh = trimesh.load(str(slot_path), force="mesh")  # type: ignore[assignment]
    components = full_mesh.split(only_watertight=False)
    components_sorted = sorted(components, key=lambda c: -len(c.vertices))
    inner_parts = [c for c in components_sorted[1:] if len(c.vertices) > 12]
    if not inner_parts:
        warnings.warn(
            "[aeroshell_png] inner_assembly: no inner components found in mesh_assembly_v4.",
            UserWarning,
            stacklevel=2,
        )
        return None

    inner = trimesh.util.concatenate(inner_parts)
    print(f"[assembly] loaded {len(inner.vertices)} verts, {len(inner.faces)} faces  ({slot_path})")
    return inner


# ---------------------------------------------------------------------------
# Solar layer loading — artifact mode
# ---------------------------------------------------------------------------


def load_solar_ply_layer(ply_path: Path) -> trimesh.Trimesh | None:
    """Load a solar PLY artifact (binary_mask or shading).

    Warns and returns None if the file does not exist rather than raising,
    so callers can skip disabled layers gracefully.

    Args:
        ply_path: Path to the PLY file.

    Returns:
        Loaded trimesh.Trimesh, or None if the file is missing.
    """
    if not ply_path.exists():
        warnings.warn(
            f"[aeroshell_png] solar layer PLY not found — {ply_path}. Skipping this solar layer.",
            UserWarning,
            stacklevel=2,
        )
        return None
    mesh = trimesh.load(str(ply_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        warnings.warn(
            f"[aeroshell_png] solar PLY at {ply_path} is not a triangle mesh. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return mesh


def _find_solar_ply(run_dir: Path, layer_type: str, epoch_num: int, slot: int = 0) -> Path | None:
    """Locate a solar PLY artifact for a given epoch/slot.

    Tries slotted path first (epoch_dir/slot_XX_mesh.ply), then flat
    epoch_XXXXX.ply.

    Args:
        run_dir: WP4 run directory.
        layer_type: One of 'binary_mask' or 'shading'.
        epoch_num: Epoch number.
        slot: Slot index (default 0).

    Returns:
        Path to the PLY file if found, else None.
    """
    base = run_dir / "solar" / layer_type
    epoch_str = f"epoch_{epoch_num:05d}"

    slotted = base / epoch_str / f"slot_{slot:02d}_mesh.ply"
    if slotted.exists():
        return slotted

    flat = base / f"{epoch_str}.ply"
    if flat.exists():
        return flat

    return None


# ---------------------------------------------------------------------------
# Solar computation — checkpoint (compute) mode
# ---------------------------------------------------------------------------


def compute_solar_fields(
    vertices: Tensor,
    faces: Tensor,
    sun_dir: Tensor,
    area_budget_mm2: float,
    effective_irradiance_W_m2: float = 250.0,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Compute solar binary mask and illumination-quality (shading) for a mesh.

    Calls ``solar_loss_sr`` with a NullVisManager so no PLY files are written.
    Both outputs are per-face float32 tensors in the original face order.

    Args:
        vertices: (V, 3) mesh vertices in mm, float32.
        faces: (F, 3) triangle face indices, int64 or int32.
        sun_dir: (3,) sun direction vector (normalised internally).
        area_budget_mm2: Maximum selectable solar area in mm².
        effective_irradiance_W_m2: Electrical output per m² of aligned area.
        sigmoid_sharpness: Steepness of the soft budget boundary sigmoid.
        soft_mask_selection_threshold: Threshold for binary mask binarisation.

    Returns:
        (binary_mask, shading) — both shape (F,), float32, values in [0, 1].
        binary_mask: 1.0 for faces inside the solar budget, 0.0 outside.
        shading: cos(sun_angle) clamped to [0, 1] (face illumination quality).
    """
    from subprojects.wp3_solar_estimator.rollout_losses.SR_SolarLoss import solar_loss_sr  # noqa: PLC0415
    from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import (  # noqa: PLC0415
        compute_face_areas_and_normals,
    )
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.central_vis_manager_null import (  # noqa: PLC0415
        NullVisManager,
    )

    verts_f32 = vertices.float()
    faces_i64 = faces.long()

    _, _, face_normals_unit = compute_face_areas_and_normals(verts_f32, faces_i64)
    sun_dir_unit = sun_dir.float().to(verts_f32.device)
    sun_dir_unit = sun_dir_unit / torch.linalg.norm(sun_dir_unit).clamp_min(1e-12)
    illumination = (face_normals_unit * sun_dir_unit).sum(-1).clamp(0.0, 1.0)

    # Run full SR pipeline to get soft mask (handles area budget constraint)
    _power, _diag = solar_loss_sr(
        verts_f32,
        faces_i64,
        sun_dir_unit,
        area_budget_mm2=area_budget_mm2,
        effective_irradiance_W_m2=effective_irradiance_W_m2,
        sigmoid_sharpness=sigmoid_sharpness,
        soft_mask_selection_threshold=soft_mask_selection_threshold,
        vis_manager=NullVisManager(),
        epoch_name=None,
    )

    # Recompute soft mask locally (solar_loss_sr doesn't return it directly)
    from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import (  # noqa: PLC0415
        compute_face_areas_and_normals,
    )

    _, face_area_m2, _ = compute_face_areas_and_normals(verts_f32, faces_i64)
    quality_order = torch.argsort(illumination, descending=True)
    sorted_areas = face_area_m2[quality_order]
    cumulative_area = torch.cumsum(sorted_areas.cpu(), dim=0).to(sorted_areas.device)
    area_budget_m2 = area_budget_mm2 * 1e-6
    distance_to_budget = (area_budget_m2 - cumulative_area) / max(area_budget_m2, 1e-12)
    soft_mask_sorted = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    # Restore original face order
    original_order = torch.argsort(quality_order)
    soft_mask = soft_mask_sorted[original_order]

    binary_mask = (soft_mask > soft_mask_selection_threshold).float()
    shading = illumination.detach()

    return binary_mask.detach(), shading


# ---------------------------------------------------------------------------
# Color-map helper for per-face scalar → RGBA per vertex (flat mesh)
# ---------------------------------------------------------------------------


def _scalar_to_rgb_mpl(values: np.ndarray, colormap: str, vmin: float, vmax: float) -> np.ndarray:
    """Map per-face scalar values to uint8 RGB via a matplotlib colormap.

    Args:
        values: (F,) float array.
        colormap: Matplotlib colormap name (e.g. 'Blues').
        vmin: Value mapped to the colormap's minimum.
        vmax: Value mapped to the colormap's maximum.

    Returns:
        (F, 3) uint8 RGB array.
    """
    from matplotlib import cm  # noqa: PLC0415

    norm_vals = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    cmap = cm.get_cmap(colormap)
    rgba = cmap(norm_vals)  # (F, 4) float
    return (rgba[:, :3] * 255).astype(np.uint8)


def _build_flat_mesh_with_colors(
    base_mesh: trimesh.Trimesh,
    face_colors: np.ndarray,
) -> trimesh.Trimesh:
    """Build a flat-shaded copy of *base_mesh* with per-face colors.

    Each triangle is duplicated into its own 3 vertices so face colors can
    be encoded as vertex colors without interpolation across face boundaries.

    Args:
        base_mesh: Source triangle mesh.
        face_colors: (F, 3) or (F, 4) uint8 color array.

    Returns:
        A new trimesh.Trimesh with per-vertex colors matching per-face input.
    """
    verts = base_mesh.vertices
    faces = base_mesh.faces
    n_faces = len(faces)

    flat_verts = verts[faces.reshape(-1)]  # (F*3, 3)
    flat_faces = np.arange(n_faces * 3, dtype=np.int64).reshape(-1, 3)

    if face_colors.ndim == 2 and face_colors.shape[1] in (3, 4):
        vertex_colors = np.repeat(face_colors, 3, axis=0)
    else:
        raise ValueError(f"face_colors must be (F, 3) or (F, 4), got {face_colors.shape}")

    flat = trimesh.Trimesh(vertices=flat_verts, faces=flat_faces, process=False)
    flat.visual = trimesh.visual.ColorVisuals(mesh=flat, vertex_colors=vertex_colors)
    return flat


# ---------------------------------------------------------------------------
# Camera pose
# ---------------------------------------------------------------------------


def compute_camera_pose(
    vertices: np.ndarray,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    distance_mult: float = 1.6,
) -> np.ndarray:
    """Compute a 4x4 camera-to-world pose (OpenGL convention, Z-up world).

    Args:
        vertices: (V, 3) mesh vertices used to derive centroid and scale.
        azim_deg: Camera azimuth angle in degrees.
        elev_deg: Camera elevation angle in degrees.
        distance_mult: Multiplier on the mesh diagonal used for camera distance.

    Returns:
        4x4 float64 pose matrix (camera-to-world, OpenGL convention).
    """
    centroid = vertices.mean(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    dist = diag * distance_mult

    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    cam_pos = centroid + dist * np.array(
        [
            np.cos(elev) * np.cos(azim),
            np.cos(elev) * np.sin(azim),
            np.sin(elev),
        ]
    )

    forward = centroid - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)

    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = cam_pos
    return pose


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _make_material(
    color_rgba: tuple[float, float, float, float],
    roughness: float = 0.65,
) -> pyrender.MetallicRoughnessMaterial:
    alpha_mode = "BLEND" if color_rgba[3] < 1.0 else "OPAQUE"
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=list(color_rgba),
        metallicFactor=0.0,
        roughnessFactor=roughness,
        alphaMode=alpha_mode,
        doubleSided=True,
    )


def render_scene(
    renderer: pyrender.OffscreenRenderer,
    cam_pose: np.ndarray,
    bg_color: tuple[float, float, float, float],
    meshes: list[tuple[trimesh.Trimesh, tuple[float, float, float, float] | None]],
) -> np.ndarray:
    """Render a list of (mesh, color_rgba) pairs to a (H, W, 3) uint8 image.

    Pass ``None`` as the color to use the mesh's embedded vertex colors (for
    flat-shaded solar layers).  Pass an RGBA tuple to apply a uniform material.

    Meshes are added in list order — render opaque layers before transparent
    ones to get correct depth compositing.

    Args:
        renderer: pyrender.OffscreenRenderer already sized to the desired output.
        cam_pose: 4x4 camera-to-world pose from compute_camera_pose.
        bg_color: Background RGBA float tuple.
        meshes: List of (trimesh_mesh, color_rgba_or_None) pairs.

    Returns:
        (H, W, 3) uint8 RGB image.
    """
    scene = pyrender.Scene(bg_color=list(bg_color), ambient_light=[0.35, 0.35, 0.35, 1.0])

    for mesh, color_rgba in meshes:
        if color_rgba is not None:
            mat = _make_material(color_rgba)
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
        else:
            # Use embedded vertex colors (flat solar overlay)
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(pr_mesh)

    aspect = renderer.viewport_width / renderer.viewport_height
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=aspect)
    scene.add(camera, pose=cam_pose)

    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.5), pose=cam_pose)

    fill_pose = np.eye(4, dtype=np.float64)
    fill_pose[:3, 3] = cam_pose[:3, 3] + np.array([0.0, 0.0, 800.0])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=fill_pose)

    color, _ = renderer.render(scene)
    return color[:, :, :3]


# ---------------------------------------------------------------------------
# Layer assembly helper
# ---------------------------------------------------------------------------


def _build_render_list(
    aeroshell_mesh: trimesh.Trimesh | None,
    inner_assembly: trimesh.Trimesh | None,
    outer_bbox: trimesh.Trimesh | None,
    solar_binary_mesh: trimesh.Trimesh | None,
    solar_shading_mesh: trimesh.Trimesh | None,
    cfg: dict[str, Any],
) -> list[tuple[trimesh.Trimesh, tuple[float, float, float, float] | None]]:
    """Assemble render-list from active layers ordered for correct compositing.

    Opaque layers (inner_assembly, outer_bbox, solar layers) are added first
    so they are correctly depth-composited behind the transparent aeroshell.
    Disabled layers are excluded entirely (not just made invisible).

    Args:
        aeroshell_mesh: Aeroshell surface mesh or None.
        inner_assembly: Inner assembly mesh or None.
        outer_bbox: Outer bounding-box mesh or None.
        solar_binary_mesh: Flat-shaded binary-mask mesh or None.
        solar_shading_mesh: Flat-shaded shading mesh or None.
        cfg: Parsed YAML config dict.

    Returns:
        List of (mesh, rgba_or_None) pairs ready for render_scene.
    """
    render_list: list[tuple[trimesh.Trimesh, tuple[float, float, float, float] | None]] = []

    def _layer_enabled(layer_name: str) -> bool:
        return bool(_get(cfg, "layers", layer_name, "enabled", default=True))

    def _layer_rgba(layer_name: str, default_hex: str, default_alpha: float) -> tuple[float, float, float, float]:
        color = _get(cfg, "layers", layer_name, "color", default=default_hex)
        alpha = float(_get(cfg, "layers", layer_name, "alpha", default=default_alpha))
        return _parse_color_rgba_float(color, alpha=alpha)

    # --- Inner assembly (opaque, rendered first) ---
    if _layer_enabled("inner_assembly") and inner_assembly is not None:
        rgba = _layer_rgba("inner_assembly", "#F39C12", 1.0)
        render_list.append((inner_assembly, rgba))

    # --- Outer bbox (transparent, rendered before aeroshell) ---
    if _layer_enabled("outer_bbox") and outer_bbox is not None:
        rgba = _layer_rgba("outer_bbox", "#FF504D", 0.12)
        render_list.append((outer_bbox, rgba))

    # --- Solar binary mask (flat-shaded overlay, no material) ---
    if _layer_enabled("solar_binary_mask") and solar_binary_mesh is not None:
        render_list.append((solar_binary_mesh, None))

    # --- Solar shading (flat-shaded overlay) ---
    if _layer_enabled("solar_shading") and solar_shading_mesh is not None:
        render_list.append((solar_shading_mesh, None))

    # --- Aeroshell (transparent, rendered last so inner layers show through) ---
    if _layer_enabled("aeroshell") and aeroshell_mesh is not None:
        rgba = _layer_rgba("aeroshell", "#33FFFF", 0.82)
        render_list.append((aeroshell_mesh, rgba))

    return render_list


# ---------------------------------------------------------------------------
# Solar mesh builders (from tensor data computed in checkpoint mode)
# ---------------------------------------------------------------------------


def _build_solar_binary_mesh(
    base_mesh: trimesh.Trimesh,
    binary_mask: Tensor,
    cfg: dict[str, Any],
) -> trimesh.Trimesh:
    """Build a flat-shaded mesh colored by the binary solar selection mask.

    Args:
        base_mesh: Aeroshell mesh (faces match binary_mask length).
        binary_mask: (F,) float tensor with 0.0 / 1.0 values.
        cfg: Parsed YAML config dict.

    Returns:
        Flat-shaded trimesh.Trimesh with per-face colors.
    """
    color_on_hex = _get(cfg, "layers", "solar_binary_mask", "color_on", default="#FFD700")
    color_off_hex = _get(cfg, "layers", "solar_binary_mask", "color_off", default="#2A2F45")
    alpha = float(_get(cfg, "layers", "solar_binary_mask", "alpha", default=1.0))

    color_on = np.array([*_parse_color_rgb(color_on_hex), int(alpha * 255)], dtype=np.uint8)
    color_off = np.array([*_parse_color_rgb(color_off_hex), int(alpha * 255)], dtype=np.uint8)

    mask_np = binary_mask.detach().cpu().numpy().reshape(-1)
    face_colors = np.where(mask_np[:, None] > 0.5, color_on, color_off).astype(np.uint8)

    return _build_flat_mesh_with_colors(base_mesh, face_colors)


def _build_solar_shading_mesh(
    base_mesh: trimesh.Trimesh,
    shading: Tensor,
    cfg: dict[str, Any],
) -> trimesh.Trimesh:
    """Build a flat-shaded mesh colored by the per-face illumination quality.

    Args:
        base_mesh: Aeroshell mesh (faces match shading length).
        shading: (F,) float tensor with values in [0, 1].
        cfg: Parsed YAML config dict.

    Returns:
        Flat-shaded trimesh.Trimesh with per-face colors.
    """
    colormap = str(_get(cfg, "layers", "solar_shading", "colormap", default="Blues"))
    alpha = float(_get(cfg, "layers", "solar_shading", "alpha", default=0.90))
    vmin = float(_get(cfg, "layers", "solar_shading", "vmin", default=0.0))
    vmax = float(_get(cfg, "layers", "solar_shading", "vmax", default=1.0))

    shading_np = shading.detach().cpu().numpy().reshape(-1)
    rgb = _scalar_to_rgb_mpl(shading_np, colormap, vmin, vmax)
    alpha_col = np.full((len(rgb), 1), int(alpha * 255), dtype=np.uint8)
    face_colors = np.concatenate([rgb, alpha_col], axis=1)

    return _build_flat_mesh_with_colors(base_mesh, face_colors)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915  # orchestrator — large by design; refactoring would lose clarity
    """CLI entry point for aeroshell PNG rendering."""
    # ------------------------------------------------------------------
    # Step 1: pre-parse --config so YAML can supply defaults
    # ------------------------------------------------------------------
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None)
    pre_args, _ = pre.parse_known_args()

    cfg: dict[str, Any] = {}
    if pre_args.config:
        cfg = _load_config(pre_args.config)

    def _c(*keys: str, default: Any = None) -> Any:
        return _get(cfg, *keys, default=default)

    # ------------------------------------------------------------------
    # Step 2: full argument parsing — CLI overrides YAML
    # ------------------------------------------------------------------
    _dev_default = _c("device", default="auto")
    if _dev_default == "auto":
        _dev_default = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(
        description="Render high-quality aeroshell PNGs (one per camera view).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG if _DEFAULT_CONFIG.exists() else None,
        help="YAML config file (default: postpro_vis_config.yaml next to script)",
    )

    # --- Input mode: checkpoint ---
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--radius", type=float, default=None, help="Latent norm (inferred from checkpoint if omitted)")
    parser.add_argument("--angle_deg", type=float, default=0.0, help="Polar angle in degrees (0° = +z₀, CCW)")

    # --- Input mode: artifact ---
    parser.add_argument("--run_dir", type=Path, default=None, help="WP4 run directory containing PLY artifacts")
    parser.add_argument("--epoch", type=int, default=None, help="Epoch number to load (artifact mode)")
    parser.add_argument("--slot", type=int, default=0, help="Population slot index for artifact lookup (default 0)")

    # --- Misc ---
    parser.add_argument("--device", default=_dev_default, help="cuda | cpu | auto")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(_c("output", "output_dir", default="/cluster/work/fuge/AISCO/wp9/aeroshell_png")),
        help="Output directory for PNGs",
    )
    args = parser.parse_args()

    if args.checkpoint is None and args.run_dir is None:
        sys.exit("ERROR: provide either --checkpoint (compute mode) or --run_dir --epoch (artifact mode)")

    device = torch.device(args.device)
    print(f"[aeroshell_png] device: {device}")

    w = int(_c("image_width", default=1920))
    h = int(_c("image_height", default=1080))
    bg_hex = _c("background_color", default="#23283E")
    bg_rgba = _parse_color_rgba_float(bg_hex)

    # ------------------------------------------------------------------
    # Mode A: checkpoint — extract mesh + compute solar
    # ------------------------------------------------------------------
    aeroshell_mesh: trimesh.Trimesh | None = None
    inner_assembly: trimesh.Trimesh | None = None
    binary_mask_tensor: Tensor | None = None
    shading_tensor: Tensor | None = None
    run_dir_resolved: Path | None = None
    epoch_num: int | None = None

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint.resolve()
        if not ckpt_path.exists():
            sys.exit(f"ERROR: checkpoint not found: {ckpt_path}")

        model, z_dim, z_radius_ckpt, _Z = load_checkpoint(ckpt_path, device)
        z_radius = args.radius if args.radius is not None else z_radius_ckpt

        z_vec = _polar_to_z(z_radius, args.angle_deg, z_dim, device=device)
        print(f"[aeroshell_png] z=[{z_vec[0].item():+.4f}, {z_vec[1].item():+.4f}]  r={z_radius:.4f}  θ={args.angle_deg}°")

        resolution = int(_c("compute", "resolution", default=64))
        bbox_padding = float(_c("compute", "bbox_padding", default=1.25))
        prism_scale = float(_c("compute", "prism_scale", default=1.0))
        prism = _load_prism(prism_scale)
        extractor = build_extractor(device, resolution, prism, bbox_padding)

        aeroshell_mesh = extract_mesh(model, extractor, z_vec, z_dim, device)
        if aeroshell_mesh is None or len(aeroshell_mesh.vertices) == 0:
            sys.exit("ERROR: mesh extraction returned empty mesh — check checkpoint and latent coordinates")

        print(f"[aeroshell_png] aeroshell: {len(aeroshell_mesh.vertices):,} verts, {len(aeroshell_mesh.faces):,} faces")

        # Attempt to load inner assembly from co-located PLY artifacts
        epoch_num = _parse_epoch_from_stem(ckpt_path.stem)
        run_dir_resolved = ckpt_path.parent.parent
        if epoch_num is not None and _get(cfg, "layers", "inner_assembly", "enabled", default=True):
            inner_assembly = load_inner_assembly(run_dir_resolved, epoch_num)

        # Compute solar fields if any solar layer is enabled
        solar_mask_enabled = bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True))
        solar_shading_enabled = bool(_get(cfg, "layers", "solar_shading", "enabled", default=True))
        if solar_mask_enabled or solar_shading_enabled:
            print("[aeroshell_png] computing solar fields (compute mode)…")
            verts_t = torch.tensor(aeroshell_mesh.vertices, dtype=torch.float32, device=device)
            faces_t = torch.tensor(aeroshell_mesh.faces, dtype=torch.long, device=device)
            sun_dir_list = _c("solar", "sun_dir", default=[0.0, -0.5, 1.0])
            sun_dir = torch.tensor(sun_dir_list, dtype=torch.float32, device=device)
            area_budget = float(_c("solar", "area_budget_mm2", default=4_200_000.0))
            irradiance = float(_c("solar", "effective_irradiance_W_m2", default=250.0))
            sharpness = float(_c("solar", "sigmoid_sharpness", default=10.0))
            threshold = float(_c("solar", "soft_mask_selection_threshold", default=0.5))

            binary_mask_tensor, shading_tensor = compute_solar_fields(
                vertices=verts_t,
                faces=faces_t,
                sun_dir=sun_dir,
                area_budget_mm2=area_budget,
                effective_irradiance_W_m2=irradiance,
                sigmoid_sharpness=sharpness,
                soft_mask_selection_threshold=threshold,
            )
            print(f"[aeroshell_png] solar: {int(binary_mask_tensor.sum().item())} / {len(aeroshell_mesh.faces)} faces selected")

    # ------------------------------------------------------------------
    # Mode B: artifact — load everything from PLYs
    # ------------------------------------------------------------------
    else:
        run_dir_resolved = args.run_dir.resolve()
        if not run_dir_resolved.exists():
            sys.exit(f"ERROR: run_dir not found: {run_dir_resolved}")

        if args.epoch is None:
            sys.exit("ERROR: --epoch is required in artifact mode")

        epoch_num = args.epoch

        # Load aeroshell from slot_00 assembly
        slot_path = run_dir_resolved / "mesh_assembly_v4" / f"epoch_{epoch_num:05d}" / f"slot_{args.slot:02d}_mesh.ply"
        if not slot_path.exists():
            # Fallback: flat epoch ply
            slot_path = run_dir_resolved / "mesh_assembly_v4" / f"epoch_{epoch_num:05d}.ply"
        if not slot_path.exists():
            sys.exit(
                f"ERROR: aeroshell PLY not found at {slot_path}\n"
                "Check --run_dir and --epoch, or use --checkpoint for compute mode."
            )
        full_mesh: trimesh.Trimesh = trimesh.load(str(slot_path), force="mesh")  # type: ignore[assignment]
        components = full_mesh.split(only_watertight=False)
        components_sorted = sorted(components, key=lambda c: -len(c.vertices))
        aeroshell_mesh = components_sorted[0] if components_sorted else None
        if aeroshell_mesh is None or len(aeroshell_mesh.vertices) == 0:
            sys.exit("ERROR: aeroshell component not found in mesh_assembly_v4 PLY")

        inner_parts = [c for c in components_sorted[1:] if len(c.vertices) > 12]
        if inner_parts and bool(_get(cfg, "layers", "inner_assembly", "enabled", default=True)):
            inner_assembly = trimesh.util.concatenate(inner_parts)

        # Solar layers from pre-built PLYs
        if bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True)):
            ply_path = _find_solar_ply(run_dir_resolved, "binary_mask", epoch_num, slot=args.slot)
            if ply_path is None:
                warnings.warn(
                    f"[aeroshell_png] solar/binary_mask PLY not found for epoch {epoch_num}. Skipping.",
                    UserWarning,
                    stacklevel=1,
                )
            else:
                bm_mesh = load_solar_ply_layer(ply_path)
                if bm_mesh is not None:
                    # Extract face scalars from vertex colors
                    vc = np.array(bm_mesh.visual.vertex_colors)[:, 0].astype(np.float32) / 255.0
                    binary_mask_tensor = torch.tensor(vc[bm_mesh.faces[:, 0]])

        if bool(_get(cfg, "layers", "solar_shading", "enabled", default=True)):
            ply_path = _find_solar_ply(run_dir_resolved, "shading", epoch_num, slot=args.slot)
            if ply_path is None:
                warnings.warn(
                    f"[aeroshell_png] solar/shading PLY not found for epoch {epoch_num}. Skipping.",
                    UserWarning,
                    stacklevel=1,
                )
            else:
                sh_mesh = load_solar_ply_layer(ply_path)
                if sh_mesh is not None:
                    vc = np.array(sh_mesh.visual.vertex_colors)[:, 0].astype(np.float32) / 255.0
                    shading_tensor = torch.tensor(vc[sh_mesh.faces[:, 0]])

    assert aeroshell_mesh is not None  # guaranteed by both branches above

    # ------------------------------------------------------------------
    # Build solar overlay meshes (shared by both modes)
    # ------------------------------------------------------------------
    solar_binary_mesh: trimesh.Trimesh | None = None
    solar_shading_mesh: trimesh.Trimesh | None = None

    if binary_mask_tensor is not None and bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True)):
        if len(binary_mask_tensor) == len(aeroshell_mesh.faces):
            solar_binary_mesh = _build_solar_binary_mesh(aeroshell_mesh, binary_mask_tensor, cfg)
        else:
            warnings.warn(
                f"[aeroshell_png] binary_mask face count ({len(binary_mask_tensor)}) "
                f"does not match aeroshell ({len(aeroshell_mesh.faces)}). Skipping.",
                UserWarning,
                stacklevel=1,
            )

    if shading_tensor is not None and bool(_get(cfg, "layers", "solar_shading", "enabled", default=True)):
        if len(shading_tensor) == len(aeroshell_mesh.faces):
            solar_shading_mesh = _build_solar_shading_mesh(aeroshell_mesh, shading_tensor, cfg)
        else:
            warnings.warn(
                f"[aeroshell_png] shading face count ({len(shading_tensor)}) "
                f"does not match aeroshell ({len(aeroshell_mesh.faces)}). Skipping.",
                UserWarning,
                stacklevel=1,
            )

    # ------------------------------------------------------------------
    # Build outer bbox mesh from WP2 prism (shared by both modes)
    # ------------------------------------------------------------------
    outer_bbox: trimesh.Trimesh | None = None
    if bool(_get(cfg, "layers", "outer_bbox", "enabled", default=False)):
        bbox_prism_scale = float(_get(cfg, "layers", "outer_bbox", "prism_scale", default=1.0))
        bp = _load_prism(bbox_prism_scale)
        extents = [
            bp["x"][1] - bp["x"][0],
            bp["y"][1] - bp["y"][0],
            bp["z"][1] - bp["z"][0],
        ]
        centre = np.array(
            [(bp["x"][0] + bp["x"][1]) / 2.0, (bp["y"][0] + bp["y"][1]) / 2.0, (bp["z"][0] + bp["z"][1]) / 2.0]
        )
        outer_bbox = trimesh.creation.box(extents=extents)
        outer_bbox.apply_translation(centre)
        print(f"[aeroshell_png] outer_bbox: extents={extents}  centre={centre}  (scale={bbox_prism_scale})")

    # ------------------------------------------------------------------
    # Render one PNG per camera view
    # ------------------------------------------------------------------
    renderer = pyrender.OffscreenRenderer(w, h)
    camera_cfgs: list[dict[str, Any]] = _c(
        "cameras", default=[{"name": "front_iso", "azim_deg": -55.0, "elev_deg": 22.0, "distance_mult": 1.6}]
    )

    # Pin camera from aeroshell vertices (inner assembly may have different bounds)
    ref_verts = aeroshell_mesh.vertices

    # Build output directory name from run info
    name_stem = args.checkpoint.stem if args.checkpoint is not None else f"run_epoch{epoch_num:05d}"

    prefix = _c("output", "filename_prefix", default="aeroshell")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[aeroshell_png] rendering {len(camera_cfgs)} view(s) → {output_dir}")

    for cam_def in camera_cfgs:
        cam_name = str(cam_def.get("name", "view"))
        azim = float(cam_def.get("azim_deg", -55.0))
        elev = float(cam_def.get("elev_deg", 22.0))
        dist_mult = float(cam_def.get("distance_mult", 1.6))

        cam_pose = compute_camera_pose(ref_verts, azim_deg=azim, elev_deg=elev, distance_mult=dist_mult)

        render_list = _build_render_list(
            aeroshell_mesh=aeroshell_mesh,
            inner_assembly=inner_assembly,
            outer_bbox=outer_bbox,
            solar_binary_mesh=solar_binary_mesh,
            solar_shading_mesh=solar_shading_mesh,
            cfg=cfg,
        )

        if not render_list:
            warnings.warn(
                f"[aeroshell_png] no enabled layers for view '{cam_name}'. Skipping.",
                UserWarning,
                stacklevel=1,
            )
            continue

        rgb = render_scene(renderer, cam_pose, bg_rgba, render_list)
        img = Image.fromarray(rgb)

        out_path = output_dir / f"{prefix}_{name_stem}_{cam_name}.png"
        img.save(str(out_path))
        print(f"  [{cam_name}] saved → {out_path}")

    renderer.delete()
    print("\n[aeroshell_png] done.")


if __name__ == "__main__":
    main()
