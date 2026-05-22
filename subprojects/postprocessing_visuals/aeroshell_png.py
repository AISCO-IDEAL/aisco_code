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
from typing import Any, cast

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
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
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

    full_mesh = cast(trimesh.Trimesh, trimesh.load(str(slot_path), force="mesh"))
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




# ---------------------------------------------------------------------------
# Solar computation — checkpoint (compute) mode
# ---------------------------------------------------------------------------


def compute_solar_fields(
    vertices: Tensor,
    faces: Tensor,
    sun_dir: Tensor,
    area_budget_mm2: float,
    cone_face_mask: Tensor | None = None,
    sigmoid_sharpness: float = 10.0,
    soft_mask_selection_threshold: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Compute solar binary mask and illumination quality (shading) for a mesh.

    Mirrors the SR_SolarLoss area-budget selection.  When ``cone_face_mask`` is
    provided, those faces receive quality=-1 before sorting so they are never
    included in the area-budget selection (matching the training-time behaviour).

    Args:
        vertices: (V, 3) mesh vertices in mm, float32.
        faces: (F, 3) triangle face indices, int64 or int32.
        sun_dir: (3,) sun direction vector (normalised internally).
        area_budget_mm2: Maximum selectable solar area in mm².
        cone_face_mask: (F,) bool tensor marking windshield faces to exclude.
        sigmoid_sharpness: Steepness of the soft budget boundary sigmoid.
        soft_mask_selection_threshold: Threshold for binary mask binarisation.

    Returns:
        (binary_mask, shading) — both shape (F,), float32.
        binary_mask: 1.0 for faces inside the solar budget, 0.0 outside.
        shading: illumination quality; -1.0 for windshield faces.
    """
    from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import (  # noqa: PLC0415
        compute_face_areas_and_normals,
    )

    epsilon = 1e-12
    verts_f32 = vertices.float()
    faces_i64 = faces.long()

    _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(verts_f32, faces_i64)
    sun_dir_unit = sun_dir.float().to(verts_f32.device)
    sun_dir_unit = sun_dir_unit / torch.linalg.norm(sun_dir_unit).clamp_min(epsilon)
    illumination = (face_normals_unit * sun_dir_unit).sum(-1)

    # Windshield faces get quality=-1: sorted last, never selected (matches SR_SolarLoss)
    if cone_face_mask is not None:
        illumination = illumination.masked_fill(cone_face_mask.to(illumination.device), -1.0)

    quality_order = torch.argsort(illumination, descending=True)
    sorted_areas = face_area_m2[quality_order]
    cumulative_area = torch.cumsum(sorted_areas.cpu(), dim=0).to(sorted_areas.device)
    area_budget_m2 = area_budget_mm2 * 1e-6
    distance_to_budget = (area_budget_m2 - cumulative_area) / max(area_budget_m2, epsilon)
    soft_mask_sorted = torch.sigmoid(sigmoid_sharpness * distance_to_budget)

    original_order = torch.argsort(quality_order)
    soft_mask = soft_mask_sorted[original_order]

    binary_mask = (soft_mask > soft_mask_selection_threshold).float()
    return binary_mask.detach(), illumination.detach()


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
    import matplotlib  # noqa: PLC0415

    norm_vals = np.clip((values - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    cmap = matplotlib.colormaps[colormap]
    rgba = cmap(norm_vals)  # (F, 4) float
    return (rgba[:, :3] * 255).astype(np.uint8)


def _bbox_wireframe_mesh(
    box_mesh: trimesh.Trimesh,
    color_rgba: tuple[float, float, float, float],
    tube_radius: float = 3.0,
) -> trimesh.Trimesh:
    """Build a solid mesh representing the wireframe edges of *box_mesh*.

    pyrender has no line-primitive support, so edges are tessellated into thin
    square-cross-section tubes (12 faces each).  The result renders as a
    wireframe cage.

    Args:
        box_mesh: Source box mesh whose unique edges are extracted.
        color_rgba: (R, G, B, A) float tuple in [0, 1].
        tube_radius: Half-width of each edge tube in mm.

    Returns:
        Flat-colored trimesh.Trimesh suitable for ``render_list``.
    """
    r_u, g_u, b_u, a_u = (int(v * 255) for v in color_rgba)
    tubes: list[trimesh.Trimesh] = []
    for edge in box_mesh.edges_unique:
        p0 = box_mesh.vertices[edge[0]]
        p1 = box_mesh.vertices[edge[1]]
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-9:
            continue
        tube = trimesh.creation.cylinder(radius=tube_radius, height=length, sections=4)
        # Align tube (initially along Z) to the edge direction
        z_axis = np.array([0.0, 0.0, 1.0])
        edge_dir = direction / length
        rotation = trimesh.geometry.align_vectors(z_axis, edge_dir)
        tube.apply_transform(rotation)
        tube.apply_translation((p0 + p1) / 2.0)
        tubes.append(tube)
    if not tubes:
        return box_mesh.copy()
    result = trimesh.util.concatenate(tubes)
    fc = np.full((len(result.faces), 4), [r_u, g_u, b_u, a_u], dtype=np.uint8)
    result.visual = trimesh.visual.ColorVisuals(mesh=result, face_colors=fc)
    return result


def _build_flat_mesh_with_colors(
    base_mesh: trimesh.Trimesh,
    face_colors: np.ndarray,
) -> trimesh.Trimesh:
    """Build a flat-shaded copy of *base_mesh* with per-face colors.

    Each triangle is duplicated into its own 3 vertices so face colors can
    be encoded as vertex colors without interpolation across face boundaries.
    Back-faces (flipped winding) are appended with the same color so the
    inside of the shell shows the same baked color as the outside.

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
) -> tuple[np.ndarray, np.ndarray]:
    """Render a list of (mesh, color_rgba) pairs to a (H, W, 3) uint8 image.

    Pass ``None`` as the color to use the mesh's embedded vertex colors (for
    flat-shaded solar-baked aeroshell or solar layers).  Pass an RGBA tuple to
    apply a uniform material.

    Args:
        renderer: pyrender.OffscreenRenderer already sized to the desired output.
        cam_pose: 4x4 camera-to-world pose from compute_camera_pose.
        bg_color: Background RGBA float tuple.
        meshes: List of (trimesh_mesh, color_rgba_or_None) pairs.

    Returns:
        Tuple of ``(color_rgb, depth)`` where ``color_rgb`` is (H, W, 3) uint8
        and ``depth`` is (H, W) float32 (0 for background pixels).
    """
    scene = pyrender.Scene(bg_color=list(bg_color), ambient_light=[0.35, 0.35, 0.35, 1.0])

    for mesh, color_rgba in meshes:
        if color_rgba is not None:
            mat = _make_material(color_rgba)
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
        else:
            # Use embedded vertex/face colors. Patch alphaMode to BLEND so the
            # alpha channel in each face color is honoured and inner layers show through.
            # Do NOT pass material= here — it would override vertex colors.
            pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            for prim in pr_mesh.primitives:
                prim.material.alphaMode = "BLEND"
                prim.material.doubleSided = True
        scene.add(pr_mesh)

    aspect = renderer.viewport_width / renderer.viewport_height
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=aspect)
    scene.add(camera, pose=cam_pose)

    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.5), pose=cam_pose)

    fill_pose = np.eye(4, dtype=np.float64)
    fill_pose[:3, 3] = cam_pose[:3, 3] + np.array([0.0, 0.0, 800.0])
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=fill_pose)

    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return color[:, :, :3], depth


# ---------------------------------------------------------------------------
# Layer assembly helper
# ---------------------------------------------------------------------------


def _build_render_list(
    aeroshell_mesh: trimesh.Trimesh | None,
    inner_assembly: trimesh.Trimesh | None,
    outer_bbox: trimesh.Trimesh | None,
    cfg: dict[str, Any],
) -> list[tuple[trimesh.Trimesh, tuple[float, float, float, float] | None]]:
    """Assemble render-list from active layers ordered for correct compositing.

    Opaque layers are added first so they are correctly depth-composited
    behind the semi-transparent aeroshell.  Disabled layers are excluded
    entirely (not just made invisible).  The windshield zone is no longer a
    separate mesh — it is baked directly into the aeroshell face colors by
    ``_bake_solar_into_aeroshell`` (via ``cone_mask``).

    Args:
        aeroshell_mesh: Aeroshell surface mesh (possibly solar-baked) or None.
        inner_assembly: Inner assembly mesh or None.
        outer_bbox: Outer bounding-box mesh or None.
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


    # --- Outer bbox: solid face or wireframe depending on config ---
    if _layer_enabled("outer_bbox") and outer_bbox is not None:
        rgba = _layer_rgba("outer_bbox", "#FF504D", 0.12)
        if bool(_get(cfg, "layers", "outer_bbox", "wireframe", default=True)):
            render_list.append((_bbox_wireframe_mesh(outer_bbox, rgba), None))
        else:
            render_list.append((outer_bbox, rgba))

    # --- Aeroshell: vertex-colored (solar baked) or uniform material ---
    if _layer_enabled("aeroshell") and aeroshell_mesh is not None:
        is_vertex_colored = isinstance(getattr(aeroshell_mesh, "visual", None), trimesh.visual.ColorVisuals)
        if is_vertex_colored:
            render_list.append((aeroshell_mesh, None))
        else:
            rgba = _layer_rgba("aeroshell", "#33FFFF", 0.82)
            render_list.append((aeroshell_mesh, rgba))

    # --- Inner assembly last, scaled toward its centroid so it is geometrically
    # inside the aeroshell shell — eliminates Z-fighting regardless of GPU precision.
    if _layer_enabled("inner_assembly") and inner_assembly is not None:
        rgba = _layer_rgba("inner_assembly", "#F39C12", 1.0)
        scale = float(_get(cfg, "layers", "inner_assembly", "z_fight_scale", default=0.98))
        render_list.append((_shrink_mesh_inward(inner_assembly, scale), rgba))

    return render_list


def _shrink_mesh_inward(mesh: trimesh.Trimesh, scale: float = 0.98) -> trimesh.Trimesh:
    """Return a copy of *mesh* uniformly scaled toward its centroid by *scale*.

    A scale < 1.0 shrinks the assembly uniformly so every surface sits strictly
    inside the aeroshell shell, eliminating Z-fighting without distorting the
    shape (unlike per-vertex-normal displacement which breaks on non-convex parts).
    """
    m = mesh.copy()
    centroid = m.vertices.mean(axis=0)
    m.vertices = centroid + (m.vertices - centroid) * scale
    return m


# ---------------------------------------------------------------------------
# Solar baking — composites solar data directly into aeroshell face colors,
# eliminating the Z-fighting caused by co-planar duplicate overlay meshes.
# ---------------------------------------------------------------------------


def _bake_solar_into_aeroshell(
    base_mesh: trimesh.Trimesh,
    binary_mask: Tensor | None,
    shading: Tensor | None,
    cone_mask: Tensor | None,
    cfg: dict[str, Any],
) -> trimesh.Trimesh:
    """Build a flat-shaded aeroshell mesh with solar data baked into per-face colors.

    Layer order (later = on top):
      1. Aeroshell base color — all faces.
      2. Solar binary mask gold — selected faces only (off-faces keep base).
      3. Solar shading colormap alpha-blended over all faces — visible on top
         of both the gold selection and the aeroshell base color.
      4. Windshield cone mask color alpha-blended onto cone faces — always on
         top so the windshield zone is clearly distinguishable.

    Args:
        base_mesh: Aeroshell mesh.
        binary_mask: (F,) float tensor (0.0/1.0), or None to skip.
        shading: (F,) float tensor in [0, 1], or None to skip.
        cone_mask: (F,) bool tensor marking windshield faces, or None to skip.
        cfg: Parsed YAML config dict.

    Returns:
        Flat-shaded trimesh.Trimesh with per-face colors encoding all active
        solar layers.  The caller should render this with ``color_rgba=None``
        so pyrender uses the embedded vertex colors.
    """
    F = len(base_mesh.faces)

    # Resolve cone face indices once (used to suppress solar layers in windshield zone)
    windshield_enabled = cone_mask is not None and bool(_get(cfg, "layers", "windshield", "enabled", default=True))
    cone_np: np.ndarray | None = None
    if windshield_enabled and cone_mask is not None:
        cone_np = cone_mask.detach().cpu().numpy().reshape(-1).astype(bool)
        if len(cone_np) != F:
            cone_np = None

    # Layer 1: aeroshell base color for all faces
    ae_hex = _get(cfg, "layers", "aeroshell", "color", default="#33FFFF")
    ae_alpha = float(_get(cfg, "layers", "aeroshell", "alpha", default=0.82))
    r, g, b = _parse_color_rgb(ae_hex)
    face_colors = np.full((F, 4), [r, g, b, int(ae_alpha * 255)], dtype=np.uint8)

    # Layer 2: binary mask — HARD OVERRIDE on selected non-windshield faces.
    # Bottom layer of the two solar layers, so shading sits visibly on top.
    mask_enabled = bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True))
    if binary_mask is not None and mask_enabled and len(binary_mask) == F:
        color_on_hex = _get(cfg, "layers", "solar_binary_mask", "color_on", default="#FFD700")
        mask_alpha = float(_get(cfg, "layers", "solar_binary_mask", "alpha", default=1.0))
        r_on, g_on, b_on = _parse_color_rgb(color_on_hex)
        color_on = np.array([r_on, g_on, b_on, int(mask_alpha * 255)], dtype=np.uint8)
        mask_np = binary_mask.detach().cpu().numpy().reshape(-1)
        face_colors[mask_np > 0.5] = color_on

    # Layer 3: shading colormap alpha-blended over ALL faces.
    # Windshield faces already carry quality=-1 (set by compute_solar_fields), which
    # the colormap renders as the minimum shade; Layer 4 overrides them with the
    # windshield color regardless.
    shading_enabled = bool(_get(cfg, "layers", "solar_shading", "enabled", default=True))
    if shading is not None and shading_enabled and len(shading) == F:
        colormap = str(_get(cfg, "layers", "solar_shading", "colormap", default="Greys_r"))
        sh_alpha = float(_get(cfg, "layers", "solar_shading", "alpha", default=0.90))
        vmin = float(_get(cfg, "layers", "solar_shading", "vmin", default=0.0))
        vmax = float(_get(cfg, "layers", "solar_shading", "vmax", default=1.0))
        shading_np = shading.detach().cpu().numpy().reshape(-1).astype(np.float32)
        rgb_map = _scalar_to_rgb_mpl(shading_np, colormap, vmin, vmax).astype(np.float32) / 255.0
        existing_rgb = face_colors[:, :3].astype(np.float32) / 255.0
        blended = rgb_map * sh_alpha + existing_rgb * (1.0 - sh_alpha)
        face_colors[:, :3] = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    # Layer 4: windshield cone mask — HARD SET, suppresses all layers above.
    # alpha=0 → transparent cut-through; alpha>0 → opaque-to-semi-transparent gray.
    if windshield_enabled and cone_np is not None:
        ws_hex = str(_get(cfg, "layers", "windshield", "color", default="#AAAAAA"))
        ws_alpha = float(_get(cfg, "layers", "windshield", "alpha", default=0.65))
        r_ws, g_ws, b_ws = _parse_color_rgb(ws_hex)
        face_colors[cone_np] = [r_ws, g_ws, b_ws, int(ws_alpha * 255)]

    return _build_flat_mesh_with_colors(base_mesh, face_colors)


# ---------------------------------------------------------------------------
# WP2 assembly fallback loaders (cadquery tessellation → trimesh)
# Used when epoch PLY artifacts are absent.
# ---------------------------------------------------------------------------


def _load_inner_assembly_from_glb(run_dir: Path) -> trimesh.Trimesh | None:
    """Load ``inner_assembly`` geometry from any GLB in the run's mesh_assembly_v4 dir.

    All slots share the same WP2 inner assembly geometry, so the first GLB
    found is used regardless of epoch or slot.

    Args:
        run_dir: WP4 run root (parent of ``mesh_assembly_v4/``).

    Returns:
        The ``inner_assembly`` sub-mesh, or None if not found / failed.
    """
    glb_files = sorted((run_dir / "mesh_assembly_v4").glob("**/*.glb"))
    if not glb_files:
        return None
    try:
        scene = trimesh.load(str(glb_files[0]), force="scene")
        if not hasattr(scene, "geometry") or "inner_assembly" not in scene.geometry:
            return None
        mesh = scene.geometry["inner_assembly"]
        print(f"[aeroshell_png] inner_assembly: {len(mesh.vertices):,} verts from GLB ({glb_files[0].name})")
    except Exception as exc:
        warnings.warn(
            f"[aeroshell_png] GLB inner_assembly load failed: {exc}. Falling back to WP2.",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return mesh


def _load_assembly_from_wp2(assembly_id: int, tol: float = 2.0, atol: float = 0.4) -> trimesh.Trimesh | None:
    """Tessellate inner assembly (seat + wheels) from WP2 cadquery.

    Used as a fallback when the epoch PLY is not available.  Each solid in
    the inner assembly is tessellated independently and concatenated.

    Args:
        assembly_id: WP2 assembly index (see assembly_map.yaml).
        tol: Tessellation linear tolerance in mm.
        atol: Tessellation angular tolerance in radians.

    Returns:
        Concatenated trimesh.Trimesh, or None on any failure.
    """
    try:
        from subprojects.wp2_assembler.version4.components.nbz import call_assembly  # noqa: PLC0415

        inner_assy, _, _ = call_assembly(id_number=assembly_id)
        inner_solids = list(inner_assy.toCompound().Solids())
        parts: list[trimesh.Trimesh] = []
        for solid in inner_solids:
            verts_cq, faces_cq = solid.tessellate(tol, atol)
            verts = np.array([[v.x, v.y, v.z] for v in verts_cq], dtype=np.float32)
            faces = np.array(faces_cq, dtype=np.int64)
            if len(verts) > 0 and len(faces) > 0:
                parts.append(trimesh.Trimesh(vertices=verts, faces=faces))
        if not parts:
            return None
        return trimesh.util.concatenate(parts)
    except Exception as exc:
        warnings.warn(
            f"[aeroshell_png] WP2 inner_assembly load failed: {exc}. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None


def _load_windshield_from_wp2(assembly_id: int, tol: float = 2.0, atol: float = 0.4) -> trimesh.Trimesh | None:
    """Tessellate the driver vision NBZ (= windshield visibility cone) from WP2.

    The vision NBZ is the regulatory driver-sightline frustum, displayed as
    the 'windshield' layer to highlight the transparency zone on the aeroshell.

    Args:
        assembly_id: WP2 assembly index.
        tol: Tessellation linear tolerance in mm.
        atol: Tessellation angular tolerance in radians.

    Returns:
        trimesh.Trimesh of the vision cone, or None on any failure.
    """
    try:
        from subprojects.wp2_assembler.version4.components.nbz import vision_nbz  # noqa: PLC0415

        assy = vision_nbz(id_number=assembly_id)
        solids = list(assy.toCompound().Solids())
        parts: list[trimesh.Trimesh] = []
        for solid in solids:
            verts_cq, faces_cq = solid.tessellate(tol, atol)
            verts = np.array([[v.x, v.y, v.z] for v in verts_cq], dtype=np.float32)
            faces = np.array(faces_cq, dtype=np.int64)
            if len(verts) > 0 and len(faces) > 0:
                parts.append(trimesh.Trimesh(vertices=verts, faces=faces))
        if not parts:
            return None
        return trimesh.util.concatenate(parts)
    except Exception as exc:
        warnings.warn(
            f"[aeroshell_png] WP2 windshield (vision NBZ) load failed: {exc}. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None


# ---------------------------------------------------------------------------
# Windshield face-mask (aeroshell faces inside the WP2 vision NBZ)
# ---------------------------------------------------------------------------


def _compute_windshield_face_mask(
    aeroshell_mesh: trimesh.Trimesh,
    assembly_id: int,
    device: torch.device,
) -> Tensor | None:
    """Return a bool tensor (F,) marking aeroshell faces inside the vision cone.

    Uses ``get_vision_cone_mesh`` (WP2 cadquery tessellation) and
    ``get_vision_cone_face_mask`` (open3d occupancy test) from wp3_vision_cone.
    Falls back to None on any import or geometry failure.
    """
    try:
        from subprojects.wp3_solar_estimator.utils.wp3_vision_cone import (  # noqa: PLC0415
            get_vision_cone_face_mask,
            get_vision_cone_mesh,
        )

        cone_verts, cone_faces = get_vision_cone_mesh(assembly_id, device=device)
        aero_verts = torch.tensor(aeroshell_mesh.vertices, dtype=torch.float32, device=device)
        aero_faces = torch.tensor(aeroshell_mesh.faces, dtype=torch.long, device=device)
        mask = get_vision_cone_face_mask(cone_verts, cone_faces, aero_verts, aero_faces)
        n = int(mask.sum().item())
        print(f"[aeroshell_png] windshield cone_face_mask: {n} / {len(aeroshell_mesh.faces)} faces")
    except Exception as exc:
        warnings.warn(
            f"[aeroshell_png] windshield face mask failed: {exc}. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return mask


# ---------------------------------------------------------------------------
# Full-wheel replacement (imports add_wheels.py in-process)
# ---------------------------------------------------------------------------

_ADD_WHEELS_PATH = (
    Path(__file__).parents[1] / "wp2_assembler" / "version4" / "3D_printing_helpers" / "add_wheels.py"
)


def _apply_full_wheels(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Replace cut-off wheel geometry in *mesh* with full wheel cylinders.

    Delegates to ``add_wheels.py`` (infer_wheel_centers → remove_cut_wheel_components
    → make_full_wheels → concatenate).  Returns the original mesh unchanged on
    any failure (non-fatal).
    """
    try:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location("_aw", _ADD_WHEELS_PATH)
        if spec is None or spec.loader is None:
            warnings.warn("[aeroshell_png] add_wheels.py not found — skipping full-wheel replacement.", UserWarning, stacklevel=2)
            return mesh
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        centers = mod.infer_wheel_centers(mesh)
        if not centers:
            warnings.warn("[aeroshell_png] No wheel-like components found — skipping full-wheel replacement.", UserWarning, stacklevel=2)
            return mesh
        base = mod.remove_cut_wheel_components(mesh, centers)
        wheels = mod.make_full_wheels(centers)
        if wheels is None:
            return base
        result = trimesh.util.concatenate([base, wheels])
        print(f"[aeroshell_png] full_wheels: {len(centers)} wheel(s) replaced")
    except Exception as exc:
        warnings.warn(f"[aeroshell_png] full_wheels failed: {exc}. Using original mesh.", UserWarning, stacklevel=2)
        return mesh
    else:
        return result


# ---------------------------------------------------------------------------
# GLB export
# ---------------------------------------------------------------------------


def _build_offset_layer_mesh(
    base_mesh: trimesh.Trimesh,
    face_mask: np.ndarray | None,
    face_colors_F4: np.ndarray,
    normal_offset_mm: float,
) -> trimesh.Trimesh | None:
    """Build a flat-shaded submesh offset along face normals for GLB layer stacking.

    Each selected face becomes 3 unique unshared vertices (flat shading) offset
    by ``normal_offset_mm`` along the face normal.  This prevents Z-fighting when
    multiple layers cover the same surface.

    Args:
        base_mesh: Source aeroshell mesh.
        face_mask: Bool array (F,) — True selects a face. None = all faces.
        face_colors_F4: (F, 4) uint8 face colors indexed by base_mesh face index.
        normal_offset_mm: Distance to offset along face normal in mm.

    Returns:
        Flat-shaded trimesh.Trimesh with embedded vertex colors, or None if no
        faces are selected.
    """
    selected = np.arange(len(base_mesh.faces)) if face_mask is None else np.where(face_mask)[0]
    if len(selected) == 0:
        return None
    v = base_mesh.vertices[base_mesh.faces[selected]].astype(np.float64)  # (K, 3, 3)
    v_off = v + base_mesh.face_normals[selected, np.newaxis, :] * normal_offset_mm
    new_verts = v_off.reshape(-1, 3)
    new_faces = np.arange(len(selected) * 3, dtype=np.int64).reshape(-1, 3)
    per_vert = np.repeat(face_colors_F4[selected], 3, axis=0)  # (K*3, 4)
    m = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    m.visual = trimesh.visual.ColorVisuals(mesh=m, vertex_colors=per_vert)
    return m


def _export_glb_layers(  # noqa: PLR0912, PLR0915
    aeroshell_mesh: trimesh.Trimesh,
    binary_mask: Tensor | None,
    shading: Tensor | None,
    cone_mask: Tensor | None,
    inner_assembly: trimesh.Trimesh | None,
    outer_bbox: trimesh.Trimesh | None,
    cfg: dict[str, Any],
    out_path: Path,
) -> None:
    """Export a GLB with each layer as a separate, independently-toggleable named node.

    Aeroshell-surface layers are offset along face normals to prevent Z-fighting
    when multiple layers are visible at once:

      aeroshell_base    0.0 mm  — base color + windshield gray/cutout
      solar_shading     0.5 mm  — full mesh with illumination-quality colormap
      solar_selection   1.0 mm  — selected faces only, gold
      windshield_zone   1.5 mm  — vision-cone faces with windshield color

    Independent meshes (no offset):
      inner_assembly, outer_bbox
    """
    F = len(aeroshell_mesh.faces)
    scene = trimesh.Scene()

    def _fc(layer: str, def_hex: str, def_alpha: float) -> np.ndarray:
        """Single RGBA uint8 row for a uniform layer."""
        c = _get(cfg, "layers", layer, "color", default=def_hex)
        a = float(_get(cfg, "layers", layer, "alpha", default=def_alpha))
        r_c, g_c, b_c = _parse_color_rgb(c)
        return np.array([r_c, g_c, b_c, int(a * 255)], dtype=np.uint8)

    # --- inner_assembly ---
    if inner_assembly is not None and bool(_get(cfg, "layers", "inner_assembly", "enabled", default=True)):
        asm = inner_assembly.copy()
        fc = np.tile(_fc("inner_assembly", "#F39C12", 1.0), (len(asm.faces), 1))
        asm.visual = trimesh.visual.ColorVisuals(mesh=asm, face_colors=fc)
        scene.add_geometry(asm, node_name="inner_assembly")

    # --- outer_bbox ---
    if outer_bbox is not None and bool(_get(cfg, "layers", "outer_bbox", "enabled", default=False)):
        bbox_color = _fc("outer_bbox", "#FF504D", 0.80)
        r_f, g_f, b_f, a_f = (v / 255.0 for v in bbox_color)
        if bool(_get(cfg, "layers", "outer_bbox", "wireframe", default=True)):
            wf = _bbox_wireframe_mesh(outer_bbox, (r_f, g_f, b_f, a_f))
            scene.add_geometry(wf, node_name="outer_bbox")
        else:
            bbox = outer_bbox.copy()
            fc = np.tile(bbox_color, (len(bbox.faces), 1))
            bbox.visual = trimesh.visual.ColorVisuals(mesh=bbox, face_colors=fc)
            scene.add_geometry(bbox, node_name="outer_bbox")

    # --- aeroshell_base --- (base color only; windshield faces excluded entirely)
    ae_rgba = _fc("aeroshell", "#33FFFF", 0.82)
    base_fc = np.tile(ae_rgba, (F, 1))
    cone_np_glb: np.ndarray | None = None
    if cone_mask is not None and bool(_get(cfg, "layers", "windshield", "enabled", default=True)):
        _cn = cone_mask.detach().cpu().numpy().reshape(-1).astype(bool)
        if len(_cn) == F:
            cone_np_glb = _cn
    # Build aeroshell without windshield faces (they live in windshield_zone node)
    non_cone_mask = ~cone_np_glb if cone_np_glb is not None else None
    glb_base = _build_offset_layer_mesh(aeroshell_mesh, non_cone_mask, base_fc, 0.0)
    if glb_base is not None:
        scene.add_geometry(glb_base, node_name="aeroshell_base")

    # --- solar_shading --- (non-windshield faces, solar_quality colormap, +0.5 mm)
    if shading is not None and bool(_get(cfg, "layers", "solar_shading", "enabled", default=True)):
        colormap = str(_get(cfg, "layers", "solar_shading", "colormap", default="solar_quality"))
        sh_alpha = float(_get(cfg, "layers", "solar_shading", "alpha", default=0.90))
        vmin = float(_get(cfg, "layers", "solar_shading", "vmin", default=0.0))
        vmax = float(_get(cfg, "layers", "solar_shading", "vmax", default=1.0))
        sh_np = shading.detach().cpu().numpy().reshape(-1).astype(np.float32)
        rgb_map = _scalar_to_rgb_mpl(sh_np, colormap, vmin, vmax)
        sh_fc = np.concatenate([rgb_map, np.full((F, 1), int(sh_alpha * 255), dtype=np.uint8)], axis=1)
        shading_mask = ~cone_np_glb if cone_np_glb is not None else None
        glb_sh = _build_offset_layer_mesh(aeroshell_mesh, shading_mask, sh_fc, 0.5)
        if glb_sh is not None:
            scene.add_geometry(glb_sh, node_name="solar_shading")

    # --- solar_selection --- (selected non-windshield faces, gold, +1.0 mm)
    if binary_mask is not None and bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True)):
        color_on_hex = str(_get(cfg, "layers", "solar_binary_mask", "color_on", default="#FFD700"))
        mask_alpha = float(_get(cfg, "layers", "solar_binary_mask", "alpha", default=1.0))
        r_on, g_on, b_on = _parse_color_rgb(color_on_hex)
        sel_mask = binary_mask.detach().cpu().numpy().reshape(-1) > 0.5
        sel_fc = np.full((F, 4), [r_on, g_on, b_on, int(mask_alpha * 255)], dtype=np.uint8)
        glb_sel = _build_offset_layer_mesh(aeroshell_mesh, sel_mask, sel_fc, 1.0)
        if glb_sel is not None:
            scene.add_geometry(glb_sel, node_name="solar_selection")

    # --- windshield_zone --- (only cone faces, windshield color, +1.5 mm)
    if cone_mask is not None and bool(_get(cfg, "layers", "windshield", "enabled", default=True)):
        ws_fc_row = _fc("windshield", "#AAAAAA", 0.65)
        ws_fc = np.tile(ws_fc_row, (F, 1))
        cone_np = cone_mask.detach().cpu().numpy().reshape(-1).astype(bool)
        if len(cone_np) == F:
            glb_ws = _build_offset_layer_mesh(aeroshell_mesh, cone_np, ws_fc, 1.5)
            if glb_ws is not None:
                scene.add_geometry(glb_ws, node_name="windshield_zone")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))
    print(f"  [glb] saved \u2192 {out_path}")


# ---------------------------------------------------------------------------
# Legend overlay
# ---------------------------------------------------------------------------


def _add_legend(
    img: Image.Image,
    entries: list[tuple[str, tuple[int, int, int, int]]],
    cfg: dict[str, Any],
) -> Image.Image:
    """Draw a legend box onto *img* (in-place via PIL draw) and return it.

    Args:
        img: Source PIL Image (RGB or RGBA).
        entries: List of ``(label, (R, G, B, A))`` uint8 tuples for active layers.
        cfg: Parsed YAML config dict (reads ``legend.position``).

    Returns:
        The modified image (same object).
    """
    if not entries:
        return img

    from PIL import ImageDraw  # noqa: PLC0415

    # Ensure RGBA so semi-transparent legend box works
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    swatch = 18
    pad = 10
    line_h = 28
    text_x_off = swatch + 8
    font_color = (230, 230, 230, 255)
    box_bg = (25, 28, 45, 210)

    max_label_px = max(len(lbl) * 8 for lbl, _ in entries)
    box_w = pad + swatch + text_x_off + max_label_px + pad
    box_h = pad + len(entries) * line_h + pad

    position = str(_get(cfg, "legend", "position", default="bottom_left"))
    x0 = img.width - box_w - pad if position in ("bottom_right", "top_right") else pad
    y0 = pad if position.startswith("top") else img.height - box_h - pad

    # Draw on a temporary RGBA overlay so the box is composited transparently
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=box_bg)

    for i, (label, rgba) in enumerate(entries):
        y = y0 + pad + i * line_h
        r, g, b, a = rgba
        draw.rectangle([x0 + pad, y, x0 + pad + swatch, y + swatch], fill=(r, g, b, min(a, 220)))
        draw.text((x0 + pad + text_x_off, y + 3), label, fill=font_color)

    img = Image.alpha_composite(img, overlay)
    return img


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915
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
    parser.add_argument(
        "--no_background",
        action="store_true",
        default=bool(_c("no_background", default=False)),
        help="Render with transparent background (saves RGBA PNG)",
    )
    parser.add_argument(
        "--assembly_id",
        type=int,
        default=int(_c("assembly_id", default=1)),
        help="WP2 assembly ID used for inner_assembly fallback and windshield loading (default 1)",
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
            # GLB fallback (any slot from mesh_assembly_v4)
            if inner_assembly is None:
                inner_assembly = _load_inner_assembly_from_glb(run_dir_resolved)
            # WP2 cadquery fallback when GLB is also absent
            if inner_assembly is None and bool(_get(cfg, "layers", "inner_assembly", "fallback_wp2", default=False)):
                _asm_id = int(_get(cfg, "layers", "inner_assembly", "assembly_id", default=args.assembly_id))
                inner_assembly = _load_assembly_from_wp2(_asm_id)
                if inner_assembly is not None:
                    print(f"[aeroshell_png] inner_assembly: {len(inner_assembly.vertices):,} verts loaded from WP2 (id={_asm_id})")
            # Replace cut-off wheel geometry with full cylinders
            if inner_assembly is not None and bool(_get(cfg, "layers", "inner_assembly", "full_wheels", default=True)):
                inner_assembly = _apply_full_wheels(inner_assembly)

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
        full_mesh = cast(trimesh.Trimesh, trimesh.load(str(slot_path), force="mesh"))
        components = full_mesh.split(only_watertight=False)
        components_sorted = sorted(components, key=lambda c: -len(c.vertices))
        aeroshell_mesh = components_sorted[0] if components_sorted else None
        if aeroshell_mesh is None or len(aeroshell_mesh.vertices) == 0:
            sys.exit("ERROR: aeroshell component not found in mesh_assembly_v4 PLY")

        inner_parts = [c for c in components_sorted[1:] if len(c.vertices) > 12]
        if inner_parts and bool(_get(cfg, "layers", "inner_assembly", "enabled", default=True)):
            inner_assembly = trimesh.util.concatenate(inner_parts)
        elif not inner_parts and bool(_get(cfg, "layers", "inner_assembly", "enabled", default=True)):
            # GLB fallback
            inner_assembly = _load_inner_assembly_from_glb(run_dir_resolved)
            # WP2 cadquery fallback
            if inner_assembly is None and bool(_get(cfg, "layers", "inner_assembly", "fallback_wp2", default=False)) and bool(
                _get(cfg, "layers", "inner_assembly", "enabled", default=True)
            ):
                _asm_id = int(_get(cfg, "layers", "inner_assembly", "assembly_id", default=args.assembly_id))
                inner_assembly = _load_assembly_from_wp2(_asm_id)
                if inner_assembly is not None:
                    print(f"[aeroshell_png] inner_assembly: {len(inner_assembly.vertices):,} verts from WP2 fallback")
        # Replace cut-off wheel geometry with full cylinders
        if inner_assembly is not None and bool(_get(cfg, "layers", "inner_assembly", "full_wheels", default=True)):
            inner_assembly = _apply_full_wheels(inner_assembly)

    assert aeroshell_mesh is not None  # guaranteed by both branches above

    # ------------------------------------------------------------------
    # Windshield: compute cone_face_mask on aeroshell (baked in, not a
    # separate mesh) using WP2 vision NBZ + open3d occupancy test.
    # ------------------------------------------------------------------
    cone_mask: Tensor | None = None
    if bool(_get(cfg, "layers", "windshield", "enabled", default=True)):
        _ws_id = int(_get(cfg, "layers", "windshield", "assembly_id", default=args.assembly_id))
        cone_mask = _compute_windshield_face_mask(aeroshell_mesh, _ws_id, device)

    # ------------------------------------------------------------------
    # Solar (checkpoint mode only — artifact mode references a different mesh).
    # cone_mask is passed so windshield faces are excluded from the area-budget
    # selection at source, matching the training-time SR_SolarLoss behaviour.
    # ------------------------------------------------------------------
    binary_mask_tensor: Tensor | None = None
    shading_tensor: Tensor | None = None
    if args.checkpoint is not None:
        solar_mask_enabled = bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True))
        solar_shading_enabled = bool(_get(cfg, "layers", "solar_shading", "enabled", default=True))
        if solar_mask_enabled or solar_shading_enabled:
            print("[aeroshell_png] computing solar fields…")
            verts_t = torch.tensor(aeroshell_mesh.vertices, dtype=torch.float32, device=device)
            faces_t = torch.tensor(aeroshell_mesh.faces, dtype=torch.long, device=device)
            sun_dir_list = _c("solar", "sun_dir", default=[0.0, -0.5, 1.0])
            sun_dir = torch.tensor(sun_dir_list, dtype=torch.float32, device=device)
            area_budget = float(_c("solar", "area_budget_mm2", default=4_200_000.0))
            sharpness = float(_c("solar", "sigmoid_sharpness", default=10.0))
            threshold = float(_c("solar", "soft_mask_selection_threshold", default=0.5))
            binary_mask_tensor, shading_tensor = compute_solar_fields(
                vertices=verts_t,
                faces=faces_t,
                sun_dir=sun_dir,
                area_budget_mm2=area_budget,
                cone_face_mask=cone_mask,
                sigmoid_sharpness=sharpness,
                soft_mask_selection_threshold=threshold,
            )
            print(f"[aeroshell_png] solar: {int(binary_mask_tensor.sum().item())} / {len(aeroshell_mesh.faces)} faces selected")

    # ------------------------------------------------------------------
    # Bake solar data + windshield zone into aeroshell face colors
    # ------------------------------------------------------------------
    solar_any = bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True)) or bool(
        _get(cfg, "layers", "solar_shading", "enabled", default=True)
    )
    aeroshell_for_render: trimesh.Trimesh = aeroshell_mesh
    if (solar_any and (binary_mask_tensor is not None or shading_tensor is not None)) or cone_mask is not None:
        aeroshell_for_render = _bake_solar_into_aeroshell(
            aeroshell_mesh, binary_mask_tensor, shading_tensor, cone_mask, cfg
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

    no_background = args.no_background
    no_bg_str = " (transparent background)" if no_background else ""
    print(f"\n[aeroshell_png] rendering {len(camera_cfgs)} view(s) → {output_dir}{no_bg_str}")

    # Collect legend entries for active layers
    legend_enabled = bool(_get(cfg, "legend", "enabled", default=False))
    legend_entries: list[tuple[str, tuple[int, int, int, int]]] = []
    if legend_enabled:
        def _le(name: str, default_hex: str, default_alpha: float) -> tuple[int, int, int, int]:
            c = _get(cfg, "layers", name, "color", default=default_hex)
            a = float(_get(cfg, "layers", name, "alpha", default=default_alpha))
            r, g, b = _parse_color_rgb(c)
            return (r, g, b, int(a * 255))

        if inner_assembly is not None and bool(_get(cfg, "layers", "inner_assembly", "enabled", default=True)):
            legend_entries.append(("inner assembly", _le("inner_assembly", "#F39C12", 1.0)))
        if cone_mask is not None and bool(_get(cfg, "layers", "windshield", "enabled", default=True)):
            legend_entries.append(("windshield", _le("windshield", "#4FC3F7", 0.40)))
        if outer_bbox is not None and bool(_get(cfg, "layers", "outer_bbox", "enabled", default=False)):
            legend_entries.append(("outer bbox", _le("outer_bbox", "#FF504D", 0.12)))
        if shading_tensor is not None and bool(_get(cfg, "layers", "solar_shading", "enabled", default=True)):
            cmap = str(_get(cfg, "layers", "solar_shading", "colormap", default="Blues"))
            legend_entries.append((f"solar shading ({cmap})", (80, 130, 200, 230)))
        if binary_mask_tensor is not None and bool(_get(cfg, "layers", "solar_binary_mask", "enabled", default=True)):
            c_on = _get(cfg, "layers", "solar_binary_mask", "color_on", default="#FFD700")
            r_on, g_on, b_on = _parse_color_rgb(c_on)
            legend_entries.append(("solar selected", (r_on, g_on, b_on, 255)))
        if bool(_get(cfg, "layers", "aeroshell", "enabled", default=True)):
            legend_entries.append(("aeroshell", _le("aeroshell", "#33FFFF", 0.82)))

    # Build render list once (geometry doesn't change between camera views)
    render_list = _build_render_list(
        aeroshell_mesh=aeroshell_for_render,
        inner_assembly=inner_assembly,
        outer_bbox=outer_bbox,
        cfg=cfg,
    )

    for cam_def in camera_cfgs:
        cam_name = str(cam_def.get("name", "view"))
        azim = float(cam_def.get("azim_deg", -55.0))
        elev = float(cam_def.get("elev_deg", 22.0))
        dist_mult = float(cam_def.get("distance_mult", 1.6))

        cam_pose = compute_camera_pose(ref_verts, azim_deg=azim, elev_deg=elev, distance_mult=dist_mult)

        if not render_list:
            warnings.warn(
                f"[aeroshell_png] no enabled layers for view '{cam_name}'. Skipping.",
                UserWarning,
                stacklevel=1,
            )
            if cam_def is camera_cfgs[0]:
                break  # all views would be empty
            continue

        rgb, depth = render_scene(renderer, cam_pose, bg_rgba, render_list)

        if no_background:
            alpha = (depth > 0).astype(np.uint8) * 255
            rgba_arr = np.dstack([rgb, alpha])
            img: Image.Image = Image.fromarray(rgba_arr, "RGBA")
        else:
            img = Image.fromarray(rgb)

        if legend_enabled and legend_entries:
            img = _add_legend(img, legend_entries, cfg)

        out_path = output_dir / f"{prefix}_{name_stem}_{cam_name}.png"
        img.save(str(out_path))
        print(f"  [{cam_name}] saved → {out_path}")

    # GLB export — one file per run, each layer as a separate named node
    if bool(_get(cfg, "output", "save_glb", default=True)):
        glb_path = output_dir / f"{prefix}_{name_stem}.glb"
        _export_glb_layers(
            aeroshell_mesh=aeroshell_mesh,
            binary_mask=binary_mask_tensor,
            shading=shading_tensor,
            cone_mask=cone_mask,
            inner_assembly=inner_assembly,
            outer_bbox=outer_bbox,
            cfg=cfg,
            out_path=glb_path,
        )

    renderer.delete()
    print("\n[aeroshell_png] done.")


if __name__ == "__main__":
    main()
