#!/usr/bin/env python3
"""Generate a GIF traversing the 2D latent space of a pretrained ConcatWIREV5.

For each frame, a z-vector is sampled along a circular arc, the aeroshell mesh
is extracted via marching cubes, rendered headlessly via pyrender/EGL, and
composited with a 2D latent-space map panel.  The frames are assembled into a
looping GIF.

Usage:
    python /cluster/home/kdiehl/AISCO_code/subprojects/postprocessing_visuals/latent_rollout_gif.py \
        --checkpoint /cluster/work/fuge/AISCO/pretraining/latent_pretraining/\
005000_ConcatWIREV5_b08f8ced4e/pretrained_latent_b08f8ced4e.pth
    [--out output.gif] [--n_frames 48] [--resolution 40] [--radius 2.0]
    [--fps 8] [--device cuda] [--width 960] [--height 600] [--panel_width 300]
    [--azim -60] [--elev 25]
"""

from __future__ import annotations

import os

os.environ["PYVISTA_OFF_SCREEN"] = "true"

import contextlib
import datetime
import glob
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
import csv

import imageio
import numpy as np
import pyvista as pv
import torch
import trimesh
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _parse_color_rgb(c: str | list) -> tuple[int, int, int]:
    """Parse '#RRGGBB' hex string or int list [R, G, B] → (R, G, B) ints 0-255."""
    if isinstance(c, (list, tuple)):
        return (int(c[0]), int(c[1]), int(c[2]))
    h = str(c).lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _parse_color_rgba_float(c: str | list, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Parse '#RRGGBB' hex string or int list → (r, g, b, a) floats 0.0-1.0."""
    r, g, b = _parse_color_rgb(c)
    return (r / 255.0, g / 255.0, b / 255.0, alpha)


# ---------------------------------------------------------------------------
# Domain constants — prism bounds (mm) matching the wp2 assembler definition
# ---------------------------------------------------------------------------

_DEFAULT_OUTER_PRISM_SCALE = 1.3
_PRISM_X: tuple[float, float] = (-2900.0, 2900.0)
_PRISM_Y: tuple[float, float] = (-1150.0, 1150.0)
_PRISM_Z: tuple[float, float] = (0.0, 1650.0)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODELS_DIR = Path(__file__).parents[1] / "wp4_aeroshell_optimizer" / "v2" / "src" / "models"


def _load_module_direct(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a .py file directly by path, registering it in sys.modules under module_name.

    Bypasses the package __init__.py chain so no CUDA extensions are triggered.
    Idempotent: returns the cached module if already loaded.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]  # type: ignore[return-value]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec and spec.loader, f"Cannot load {file_path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod  # register before exec to handle circular refs
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
    2. Register a minimal BaseModel stub for model_baseclass (WIRE_v5 defines
       a WIREV5(BaseModel) class but we never instantiate it).
    3. Directly load concat_baseclass, WIRE_v5, and concat_WIRE_v5 from their
       source files, bypassing the package import machinery entirely.
    """
    base_dir = Path(__file__).parents[1]
    package_paths = {
        "subprojects.wp4_aeroshell_optimizer": base_dir / "wp4_aeroshell_optimizer",
        "subprojects.wp4_aeroshell_optimizer.v2": base_dir / "wp4_aeroshell_optimizer" / "v2",
        "subprojects.wp4_aeroshell_optimizer.v2.src": base_dir / "wp4_aeroshell_optimizer" / "v2" / "src",
        "subprojects.wp4_aeroshell_optimizer.v2.src.models": base_dir / "wp4_aeroshell_optimizer" / "v2" / "src" / "models",
    }
    # --- 1. Parent package stubs ---
    for pkg, pkg_path in package_paths.items():
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(pkg_path)]  # type: ignore[attr-defined]  # marks it as a package and keeps subpackage imports working
            sys.modules[pkg] = m

    # --- 2. Minimal model_baseclass stub (WIRE_v5 imports BaseModel from here) ---
    mb_key = "subprojects.wp4_aeroshell_optimizer.v2.src.models.model_baseclass"
    if mb_key not in sys.modules:
        from torch import nn  # noqa: PLC0415

        mb_stub = types.ModuleType(mb_key)

        class _BaseModel(nn.Module):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                super().__init__()

        mb_stub.BaseModel = _BaseModel  # type: ignore[attr-defined]
        sys.modules[mb_key] = mb_stub

    # --- 3. Load real model files directly ---
    for mod_name, fname in [
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass", "concat_baseclass.py"),
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v5", "WIRE_v5.py"),
        ("subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_WIRE_v5", "concat_WIRE_v5.py"),
    ]:
        _load_module_direct(mod_name, _MODELS_DIR / fname)


def _build_model(info: dict[str, Any], device: torch.device) -> Any:
    root = str(Path(__file__).parents[2])  # AISCO_code root
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
    """Walk up the directory tree from the checkpoint to find model_info_dict.json
    or any Architecture_*.json file (same schema, different name convention)."""
    search = ckpt_path.parent
    for _ in range(6):  # at most 6 levels up
        candidate = search / "model_info_dict.json"
        if candidate.exists():
            return candidate
        arch_matches = glob.glob(str(search / "Architecture_*.json"))
        if arch_matches:
            return Path(arch_matches[0])
        search = search.parent
    sys.exit(f"ERROR: could not find model_info_dict.json or Architecture_*.json in any ancestor of {ckpt_path.parent}")


def load_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[Any, int, float, Tensor | None]:
    """Return (model, z_dim, z_radius, Z_pretrained)."""
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
# Inner assembly
# ---------------------------------------------------------------------------


def _parse_epoch_from_stem(stem: str) -> int | None:
    """Extract epoch number from a checkpoint stem like 'run_name_epoch1400'."""
    m = re.search(r"_epoch(\d+)$", stem)
    return int(m.group(1)) if m else None


def _checkpoint_prefix_from_stem(stem: str) -> str:
    """Return the shared checkpoint prefix for stems shaped as '<prefix>_epoch<NNN>'."""
    m = re.match(r"^(.*)_epoch\d+$", stem)
    if m is None:
        sys.exit(
            "ERROR: checkpoint name must end with '_epoch<NNN>' for epoch_gif mode. "
            "Use a checkpoint like '<run_name>_epoch2000.pth'."
        )
    return m.group(1)


def _find_epoch_checkpoints(ckpt_path: Path) -> list[tuple[int, Path]]:
    """Find all epoch checkpoints in the same directory, sorted chronologically."""
    prefix = _checkpoint_prefix_from_stem(ckpt_path.stem)
    pattern = f"{prefix}_epoch*.pth"
    matches = sorted(ckpt_path.parent.glob(pattern))

    parsed: list[tuple[int, Path]] = []
    for candidate in matches:
        epoch = _parse_epoch_from_stem(candidate.stem)
        if epoch is not None:
            parsed.append((epoch, candidate))

    if not parsed:
        sys.exit(f"ERROR: no epoch checkpoints found next to {ckpt_path}. Expected files matching pattern '{pattern}'.")

    parsed.sort(key=lambda x: x[0])
    print(f"[epoch_gif] discovered {len(parsed)} checkpoints: epoch {parsed[0][0]} -> {parsed[-1][0]}")
    return parsed


def _load_model_state_from_checkpoint(model: Any, ckpt_path: Path, device: torch.device) -> None:
    """Load model weights from a single checkpoint into an already-built model."""
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = raw["model_state_dict"] if isinstance(raw, dict) and "model_state_dict" in raw else raw
    model.load_state_dict(state_dict)
    model.eval()


def load_inner_assembly(ckpt_path: Path, epoch_num: int) -> trimesh.Trimesh | None:
    """Load the inner_assembly mesh from the slot_00_mesh.glb next to the checkpoint.

    The GLB stores named geometry nodes:
      - sdf_mesh       : aeroshell zero level set  (skipped)
      - inner_assembly : fixed inner structure     <- returned
      - outer_bbox     : bounding box wireframe    (skipped)

    If the exact epoch directory has no GLB, the closest earlier epoch that does
    is used as a fallback (the inner assembly is fixed across epochs).
    """
    run_dir = ckpt_path.parent.parent  # .../models/foo.pth -> .../run_dir
    assembly_dir = run_dir / "mesh_assembly_v4"
    if not assembly_dir.exists():
        print(f"[assembly] not found: {assembly_dir}")
        return None

    # Prefer the exact epoch; fall back to closest earlier epoch.
    target_epoch_dir = assembly_dir / f"epoch_{epoch_num:05d}"
    glb_path = target_epoch_dir / "slot_00_mesh.glb"
    if not glb_path.exists():
        available = sorted(assembly_dir.glob("epoch_*/slot_00_mesh.glb"))
        if not available:
            print(f"[assembly] no GLB files found under {assembly_dir}")
            return None
        # Use the closest epoch that is <= epoch_num, else the earliest available.
        candidates = [p for p in available if int(p.parent.name.split("_")[1]) <= epoch_num]
        glb_path = candidates[-1] if candidates else available[0]
        print(f"[assembly] exact epoch not found, using fallback: {glb_path.parent.name}")

    scene = trimesh.load(str(glb_path))
    if not hasattr(scene, "geometry") or "inner_assembly" not in scene.geometry:
        print(f"[assembly] WARN: 'inner_assembly' node not found in {glb_path}")
        return None

    inner = scene.geometry["inner_assembly"]
    print(f"[assembly] loaded {len(inner.vertices)} verts, {len(inner.faces)} faces  ({glb_path})")
    return inner


# ---------------------------------------------------------------------------
# Optional visual layers (wheels, windshield, solar selection)
# ---------------------------------------------------------------------------

_ADD_WHEELS_PATH = Path(__file__).parents[1] / "wp2_assembler" / "version4" / "3D_printing_helpers" / "add_wheels.py"


def _apply_full_wheels(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Replace cut-off wheel geometry with full wheel cylinders.

    Returns the original mesh unchanged on any failure.
    """
    try:
        spec = importlib.util.spec_from_file_location("_aw", _ADD_WHEELS_PATH)
        if spec is None or spec.loader is None:
            warnings.warn(
                "[latent_rollout] add_wheels.py not found - skipping full-wheel replacement.",
                UserWarning,
                stacklevel=2,
            )
            return mesh
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        centers = mod.infer_wheel_centers(mesh)
        if not centers:
            warnings.warn(
                "[latent_rollout] no wheel-like components found - skipping full-wheel replacement.",
                UserWarning,
                stacklevel=2,
            )
            return mesh
        base = mod.remove_cut_wheel_components(mesh, centers)
        wheels = mod.make_full_wheels(centers)
        if wheels is None:
            return base
        out = trimesh.util.concatenate([base, wheels])
        print(f"[latent_rollout] full_wheels: {len(centers)} wheel(s) replaced")
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] full_wheels failed: {exc}. Using original mesh.",
            UserWarning,
            stacklevel=2,
        )
        return mesh
    else:
        return out


def _precompute_vision_cone_mesh(
    assembly_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor] | None:
    """Load and cache the WP3 vision cone mesh once for reuse across frames."""
    try:
        from subprojects.wp3_solar_estimator.utils.wp3_vision_cone import get_vision_cone_mesh  # noqa: PLC0415

        cone_verts, cone_faces = get_vision_cone_mesh(assembly_id, device=device)
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] failed to precompute vision cone mesh: {exc}. Proceeding without cache.",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return cone_verts, cone_faces


def _compute_windshield_face_mask(
    aeroshell_mesh: trimesh.Trimesh,
    assembly_id: int,
    device: torch.device,
    precomputed_cone: tuple[Tensor, Tensor] | None = None,
) -> Tensor | None:
    """Return a bool face mask marking aeroshell faces inside the vision cone."""
    try:
        from subprojects.wp3_solar_estimator.utils.wp3_vision_cone import get_vision_cone_face_mask  # noqa: PLC0415

        if precomputed_cone is not None:
            cone_verts, cone_faces = precomputed_cone
            cone_verts = cone_verts.to(device=device)
            cone_faces = cone_faces.to(device=device)
        else:
            from subprojects.wp3_solar_estimator.utils.wp3_vision_cone import get_vision_cone_mesh  # noqa: PLC0415

            cone_verts, cone_faces = get_vision_cone_mesh(assembly_id, device=device)

        aero_verts = torch.tensor(aeroshell_mesh.vertices, dtype=torch.float32, device=device)
        aero_faces = torch.tensor(aeroshell_mesh.faces, dtype=torch.long, device=device)
        mask = get_vision_cone_face_mask(cone_verts, cone_faces, aero_verts, aero_faces)
        n = int(mask.sum().item())
        print(f"[latent_rollout] windshield cone_face_mask: {n} / {len(aeroshell_mesh.faces)} faces")
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] windshield face mask failed: {exc}. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return mask


def _contour_postprocess_solar_mask(
    binary_mask: Tensor,
    valid_mask: Tensor,
    aeroshell_mesh: trimesh.Trimesh,
    contour_postprocess_iters: int,
    contour_hole_fill_iters: int,
    area_np: np.ndarray,
    illum_np: np.ndarray,
    area_budget_mm2: float,
    device: torch.device,
    adjacency: np.ndarray,
) -> Tensor:
    """Apply contour smoothing and hole-fill passes to a binary solar mask."""
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    if contour_postprocess_iters > 0:
        base_np = binary_mask.detach().cpu().numpy().astype(np.float32)
        contour_scores = _smooth_binary_contour_scores(
            mesh=aeroshell_mesh,
            base_scores=base_np,
            valid_mask=valid_np,
            iterations=contour_postprocess_iters,
            adjacency=adjacency,
        )
        valid_idx = np.flatnonzero(valid_np)
        if valid_idx.size > 0:
            target_area_m2_np = min(max(area_budget_mm2 * 1e-6, 0.0), float(area_np[valid_idx].sum()))
            order = valid_idx[np.lexsort((-illum_np[valid_idx], -contour_scores[valid_idx]))]
            cumulative = np.cumsum(area_np[order])
            if target_area_m2_np <= 0.0:
                keep_count = 0
            else:
                keep_count = int(np.searchsorted(cumulative, target_area_m2_np, side="left")) + 1
                keep_count = min(keep_count, int(order.size))
            selected_np = np.zeros(len(area_np), dtype=bool)
            if keep_count > 0:
                selected_np[order[:keep_count]] = True
            binary_mask = torch.tensor(selected_np, dtype=torch.bool, device=device)
    if contour_hole_fill_iters > 0:
        filled_np = _fill_binary_contour_holes(
            mesh=aeroshell_mesh,
            binary_mask=binary_mask.detach().cpu().numpy().astype(bool),
            valid_mask=valid_np,
            iterations=contour_hole_fill_iters,
            adjacency=adjacency,
        )
        binary_mask = torch.tensor(filled_np, dtype=torch.bool, device=device)
    return binary_mask


def _postop_reallocate_solar_budget_np(
    selected_np: np.ndarray,
    valid_np: np.ndarray,
    area_np: np.ndarray,
    illum_np: np.ndarray,
    target_area_m2_np: float,
) -> np.ndarray:
    """Vectorised trim/refill to align selected solar area with target budget."""
    selected_area_m2_np = float(area_np[selected_np].sum())
    if selected_area_m2_np > target_area_m2_np:
        selected_idx = np.flatnonzero(selected_np & valid_np)
        if selected_idx.size > 0:
            remove_order = selected_idx[np.argsort(illum_np[selected_idx], kind="stable")]
            excess = selected_area_m2_np - target_area_m2_np
            remove_cumulative = np.cumsum(area_np[remove_order])
            remove_count = int(np.searchsorted(remove_cumulative, excess, side="left")) + 1
            remove_count = min(remove_count, int(remove_order.size))
            if remove_count > 0:
                selected_np[remove_order[:remove_count]] = False
                selected_area_m2_np = float(area_np[selected_np].sum())
    if selected_area_m2_np < target_area_m2_np:
        candidate_idx = np.flatnonzero((~selected_np) & valid_np)
        if candidate_idx.size > 0:
            add_order = candidate_idx[np.argsort(-illum_np[candidate_idx], kind="stable")]
            needed = target_area_m2_np - selected_area_m2_np
            add_cumulative = np.cumsum(area_np[add_order])
            add_count = int(np.searchsorted(add_cumulative, needed, side="left")) + 1
            add_count = min(add_count, int(add_order.size))
            if add_count > 0:
                selected_np[add_order[:add_count]] = True
    return selected_np


def _compute_solar_selection_mask(
    aeroshell_mesh: trimesh.Trimesh,
    device: torch.device,
    sun_dir_xyz: tuple[float, float, float],
    area_budget_mm2: float,
    cone_face_mask: Tensor | None = None,
    binary_postprocess_iters: int = 2,
    contour_postprocess_iters: int = 6,
    contour_hole_fill_iters: int = 2,
    sigmoid_sharpness: float = 10.0,
    threshold: float = 0.5,
) -> Tensor | None:
    """Compute a binary solar-selection face mask using SR_SolarLoss-style budgeting."""
    try:
        from subprojects.wp3_solar_estimator.utils.rollout_utils.mesh_area_normals import (  # noqa: PLC0415
            compute_face_areas_and_normals,
        )

        epsilon = 1e-12
        verts = torch.tensor(aeroshell_mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(aeroshell_mesh.faces, dtype=torch.long, device=device)
        _, face_area_m2, face_normals_unit = compute_face_areas_and_normals(verts, faces)

        sun_dir = torch.tensor(sun_dir_xyz, dtype=torch.float32, device=device)
        sun_dir = sun_dir / torch.linalg.norm(sun_dir).clamp_min(epsilon)
        illumination = (face_normals_unit * sun_dir).sum(-1)
        valid_mask = torch.ones_like(illumination, dtype=torch.bool)
        if cone_face_mask is not None:
            cone_face_mask = cone_face_mask.to(device=device, dtype=torch.bool)
            valid_mask = ~cone_face_mask
            illumination = illumination.masked_fill(cone_face_mask, -1.0)

        illum_np = illumination.detach().cpu().numpy()
        area_np = face_area_m2.detach().cpu().numpy()
        valid_np = valid_mask.detach().cpu().numpy().astype(bool)
        cached_adjacency = _get_face_adjacency_indices(aeroshell_mesh)

        quality_order = torch.argsort(illumination, descending=True)
        sorted_areas = face_area_m2[quality_order]
        cumulative_area = torch.cumsum(sorted_areas.cpu(), dim=0).to(sorted_areas.device)
        area_budget_m2 = area_budget_mm2 * 1e-6
        distance_to_budget = (area_budget_m2 - cumulative_area) / max(area_budget_m2, epsilon)
        soft_mask_sorted = torch.sigmoid(sigmoid_sharpness * distance_to_budget)
        original_order = torch.argsort(quality_order)
        soft_mask = soft_mask_sorted[original_order]
        binary_mask = soft_mask > threshold

        if binary_postprocess_iters > 0:
            binary_np = binary_mask.detach().cpu().numpy().astype(np.float32)
            smoothed_scores = _smooth_binary_face_scores(
                mesh=aeroshell_mesh,
                base_scores=binary_np,
                valid_mask=valid_np,
                iterations=binary_postprocess_iters,
                adjacency=cached_adjacency,
            )
            binary_mask = torch.tensor(smoothed_scores > 0.5, dtype=torch.bool, device=device)
        binary_mask = binary_mask & valid_mask

        binary_mask = _contour_postprocess_solar_mask(
            binary_mask=binary_mask,
            valid_mask=valid_mask,
            aeroshell_mesh=aeroshell_mesh,
            contour_postprocess_iters=contour_postprocess_iters,
            contour_hole_fill_iters=contour_hole_fill_iters,
            area_np=area_np,
            illum_np=illum_np,
            area_budget_mm2=area_budget_mm2,
            device=device,
            adjacency=cached_adjacency,
        )

        target_area_m2_np = min(max(area_budget_mm2 * 1e-6, 0.0), float(area_np[valid_np].sum()))
        selected_np = binary_mask.detach().cpu().numpy().astype(bool)
        selected_np = _postop_reallocate_solar_budget_np(
            selected_np=selected_np,
            valid_np=valid_np,
            area_np=area_np,
            illum_np=illum_np,
            target_area_m2_np=target_area_m2_np,
        )
        binary_mask = torch.tensor(selected_np, dtype=torch.bool, device=device)
    except Exception as exc:
        warnings.warn(f"[latent_rollout] solar selection failed: {exc}. Skipping.", UserWarning, stacklevel=2)
        return None
    else:
        return binary_mask


def _build_face_overlay_mesh(
    base_mesh: trimesh.Trimesh,
    face_mask: np.ndarray,
    normal_offset_mm: float,
) -> trimesh.Trimesh | None:
    """Build a submesh for selected faces and offset it slightly to avoid z-fighting."""
    idx = np.flatnonzero(face_mask.astype(bool))
    if idx.size == 0:
        return None
    try:
        sub = base_mesh.submesh([idx], append=True, repair=False)
        if not isinstance(sub, trimesh.Trimesh) or len(sub.vertices) == 0:
            return None
        if normal_offset_mm != 0.0:
            vnorm = sub.vertex_normals
            sub.vertices = sub.vertices + (vnorm * float(normal_offset_mm))
    except Exception:
        return None
    else:
        return sub


def _cutout_faces_from_mesh(base_mesh: trimesh.Trimesh, face_mask: np.ndarray) -> trimesh.Trimesh | None:
    """Return a mesh with masked faces removed."""
    keep_idx = np.flatnonzero(~face_mask.astype(bool))
    if keep_idx.size == 0:
        return None
    try:
        sub = base_mesh.submesh([keep_idx], append=True, repair=False)
        if not isinstance(sub, trimesh.Trimesh) or len(sub.vertices) == 0:
            return None
    except Exception:
        return None
    else:
        return sub


def _subdivide_mesh_for_masks(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Refine mesh triangulation for cleaner mask/cutout boundaries without smoothing."""
    if iterations <= 0:
        return mesh
    out = mesh
    try:
        for _ in range(int(iterations)):
            out = out.subdivide()
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] mask subdivision failed: {exc}. Using original mesh.",
            UserWarning,
            stacklevel=2,
        )
        return mesh
    else:
        return out


def _smooth_binary_face_scores(
    mesh: trimesh.Trimesh,
    base_scores: np.ndarray,
    valid_mask: np.ndarray,
    iterations: int,
    adjacency: np.ndarray | None = None,
) -> np.ndarray:
    """Diffuse binary face scores across face adjacency while keeping a hard final mask possible."""
    if iterations <= 0:
        return base_scores

    adjacency = _get_face_adjacency_indices(mesh) if adjacency is None else np.asarray(adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return base_scores

    scores = base_scores.astype(np.float32, copy=True)
    valid_f32 = valid_mask.astype(np.float32)
    left = adjacency[:, 0]
    right = adjacency[:, 1]

    for _ in range(int(iterations)):
        accum = scores.copy()
        counts = np.ones_like(scores, dtype=np.float32)
        np.add.at(accum, left, scores[right] * valid_f32[right])
        np.add.at(accum, right, scores[left] * valid_f32[left])
        np.add.at(counts, left, valid_f32[right])
        np.add.at(counts, right, valid_f32[left])
        scores = (accum / np.clip(counts, 1.0, None)).astype(np.float32)
        scores[~valid_mask] = -1.0

    return scores


def _smooth_binary_contour_scores(
    mesh: trimesh.Trimesh,
    base_scores: np.ndarray,
    valid_mask: np.ndarray,
    iterations: int,
    adjacency: np.ndarray | None = None,
) -> np.ndarray:
    """Smooth only boundary faces so contour gets cleaner while interior stays stable."""
    if iterations <= 0:
        return base_scores

    adjacency = _get_face_adjacency_indices(mesh) if adjacency is None else np.asarray(adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return base_scores

    scores = base_scores.astype(np.float32, copy=True)
    left = adjacency[:, 0]
    right = adjacency[:, 1]
    valid_edges = valid_mask[left] & valid_mask[right]
    valid_f32 = valid_mask.astype(np.float32)

    for _ in range(int(iterations)):
        current_binary = scores > 0.5
        edge_cross = valid_edges & (current_binary[left] != current_binary[right])
        if not np.any(edge_cross):
            break

        boundary = np.zeros_like(current_binary, dtype=bool)
        np.logical_or.at(boundary, left[edge_cross], True)
        np.logical_or.at(boundary, right[edge_cross], True)
        boundary &= valid_mask

        accum = scores.copy()
        counts = np.ones_like(scores, dtype=np.float32)
        np.add.at(accum, left, scores[right] * valid_f32[right])
        np.add.at(accum, right, scores[left] * valid_f32[left])
        np.add.at(counts, left, valid_f32[right])
        np.add.at(counts, right, valid_f32[left])
        averaged = accum / np.clip(counts, 1.0, None)

        # Keep interior faces unchanged; only relax boundary values.
        scores[boundary] = 0.35 * scores[boundary] + 0.65 * averaged[boundary]
        scores[~valid_mask] = -1.0

    return scores


def _fill_binary_contour_holes(
    mesh: trimesh.Trimesh,
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    iterations: int,
    min_positive_ratio: float = 0.7,
    min_neighbor_count: int = 3,
    adjacency: np.ndarray | None = None,
) -> np.ndarray:
    """Fill small holes inside selected regions without blurring the outer contour."""
    if iterations <= 0:
        return binary_mask

    adjacency = _get_face_adjacency_indices(mesh) if adjacency is None else np.asarray(adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return binary_mask

    left = adjacency[:, 0]
    right = adjacency[:, 1]
    valid_edges = valid_mask[left] & valid_mask[right]
    if not np.any(valid_edges):
        return binary_mask

    filled = binary_mask.astype(bool, copy=True)
    for _ in range(int(iterations)):
        pos_counts = np.zeros_like(filled, dtype=np.float32)
        nbr_counts = np.zeros_like(filled, dtype=np.float32)

        np.add.at(pos_counts, left[valid_edges], filled[right[valid_edges]].astype(np.float32))
        np.add.at(pos_counts, right[valid_edges], filled[left[valid_edges]].astype(np.float32))
        np.add.at(nbr_counts, left[valid_edges], 1.0)
        np.add.at(nbr_counts, right[valid_edges], 1.0)

        with np.errstate(invalid="ignore", divide="ignore"):
            pos_ratio = np.divide(pos_counts, np.clip(nbr_counts, 1.0, None))

        fill_candidates = (
            (~filled) & valid_mask & (nbr_counts >= float(min_neighbor_count)) & (pos_ratio >= float(min_positive_ratio))
        )
        if not np.any(fill_candidates):
            break
        filled[fill_candidates] = True

    return filled


def _get_face_adjacency_indices(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return face adjacency as (E, 2) int64 array with a stable empty fallback."""
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if adjacency.ndim != 2 or adjacency.shape[1] != 2:
        warnings.warn(
            "[latent_rollout] unexpected face_adjacency shape; skipping adjacency-based smoothing.",
            UserWarning,
            stacklevel=2,
        )
        return np.empty((0, 2), dtype=np.int64)
    return adjacency


def _allocate_area_budget_binary_mask(
    mesh: trimesh.Trimesh,
    face_area_m2: Tensor,
    illumination: Tensor,
    area_budget_mm2: float,
    valid_mask: Tensor,
    postprocess_iters: int = 0,
) -> Tensor:
    """Allocate solar area budget only over valid faces, optionally smoothing the binary footprint."""
    area_budget_m2 = max(area_budget_mm2 * 1e-6, 0.0)
    valid_np = valid_mask.detach().cpu().numpy().astype(bool)
    illum_np = illumination.detach().cpu().numpy()
    area_np = face_area_m2.detach().cpu().numpy()
    binary_np = np.zeros(len(area_np), dtype=bool)

    valid_idx = np.flatnonzero(valid_np)
    if valid_idx.size == 0 or area_budget_m2 <= 0.0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)

    initial_order = valid_idx[np.argsort(-illum_np[valid_idx], kind="stable")]
    cumulative_initial = np.cumsum(area_np[initial_order])
    initial_count = int(np.searchsorted(cumulative_initial, area_budget_m2, side="right"))
    if initial_count <= 0:
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    binary_np[initial_order[:initial_count]] = True

    if postprocess_iters > 0:
        smoothed_scores = _smooth_binary_face_scores(mesh, binary_np.astype(np.float32), valid_np, postprocess_iters)
        refined_order = valid_idx[np.lexsort((-illum_np[valid_idx], -smoothed_scores[valid_idx]))]
        cumulative_refined = np.cumsum(area_np[refined_order])
        refined_count = int(np.searchsorted(cumulative_refined, area_budget_m2, side="right"))
        binary_np[:] = False
        if refined_count > 0:
            binary_np[refined_order[:refined_count]] = True

    return torch.tensor(binary_np, dtype=torch.bool, device=valid_mask.device)


def _post_smooth_overlay_mesh(mesh: trimesh.Trimesh | None, iterations: int) -> trimesh.Trimesh | None:
    """Apply post-op Taubin smoothing to overlay geometry edges only."""
    if mesh is None or iterations <= 0:
        return mesh
    try:
        out = mesh.copy()
        trimesh.smoothing.filter_taubin(out, lamb=0.5, nu=-0.53, iterations=int(iterations))
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] solar overlay post-smoothing failed: {exc}. Using unsmoothed overlay.",
            UserWarning,
            stacklevel=2,
        )
        return mesh
    else:
        return out


# ---------------------------------------------------------------------------
# Mesh extraction
# ---------------------------------------------------------------------------


def build_extractor(device: torch.device, resolution: int, bbox_padding: float = 0.0) -> Any:
    from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config  # noqa: PLC0415

    cfg = load_mesh_extraction_config()
    cfg = dc_replace(cfg, device=str(device))
    extractor = MeshSDFExtractor(config=cfg)
    extractor.set_grid(
        bbox=[
            [_PRISM_X[0] - bbox_padding, _PRISM_X[1] + bbox_padding],
            [_PRISM_Y[0] - bbox_padding, _PRISM_Y[1] + bbox_padding],
            [_PRISM_Z[0], _PRISM_Z[1] + bbox_padding],
        ],
        resolution=resolution,
    )
    return extractor


def build_extractor_with_wp2_prism(
    device: torch.device,
    resolution: int,
    bbox_padding: float = 0.0,
    prism_scale: float = _DEFAULT_OUTER_PRISM_SCALE,
) -> Any:
    """Build extractor using WP2 regulation-derived outer prism bounds.

    Falls back to legacy hardcoded bounds if WP2 utility import fails.
    """
    from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config  # noqa: PLC0415

    cfg = load_mesh_extraction_config()
    cfg = dc_replace(cfg, device=str(device))
    extractor = MeshSDFExtractor(config=cfg)

    try:
        from subprojects.wp2_assembler.version4.utils.reg_imp import (  # noqa: PLC0415
            get_outer_prism_bounds_mm,
            scale_outer_prism,
        )

        outer = get_outer_prism_bounds_mm()
        scaled = scale_outer_prism(outer, prism_scale)
        prism_x = scaled["x"]
        prism_y = scaled["y"]
        prism_z = scaled["z"]
        print(f"[extractor] WP2 outer prism scale={prism_scale:.3f} x={prism_x} y={prism_y} z={prism_z}")
    except Exception as exc:
        warnings.warn(
            f"[extractor] WP2 prism import failed ({exc}); using legacy hardcoded bounds.",
            UserWarning,
            stacklevel=2,
        )
        prism_x = (-2900.0, 2900.0)
        prism_y = (-1150.0, 1150.0)
        prism_z = (0.0, 1650.0)

    extractor.set_grid(
        bbox=[
            [prism_x[0] - bbox_padding, prism_x[1] + bbox_padding],
            [prism_y[0] - bbox_padding, prism_y[1] + bbox_padding],
            [prism_z[0], prism_z[1] + bbox_padding],
        ],
        resolution=resolution,
    )
    return extractor


def extract_mesh(
    model: Any,
    extractor: Any,
    z0: float,
    z1: float,
    z_dim: int,
    device: torch.device,
) -> trimesh.Trimesh | None:
    from subprojects.wp5.unified_mesh_extraction import clear_mesh_cache  # noqa: PLC0415

    z_vec = torch.zeros(z_dim, device=device)
    z_vec[0] = z0
    z_vec[1] = z1
    model.set_latent_vector(z_vec)
    clear_mesh_cache()

    def _sdf(pts: Tensor) -> Tensor:
        return model(pts)

    try:
        verts, faces = extractor.extract_mesh(_sdf)
    except Exception as exc:
        print(f"  WARN: extraction failed: {exc}")
        return None

    if verts is None or verts.numel() == 0:
        return None

    return trimesh.Trimesh(vertices=verts.cpu().numpy(), faces=faces.cpu().numpy())


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_mesh(
    mesh: trimesh.Trimesh,
    bg_color: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0),
    mesh_color: tuple[float, float, float, float] = (0.2, 1.0, 1.0, 1.0),
    mesh_alpha: float = 0.75,
    assembly_mesh: trimesh.Trimesh | None = None,
    assembly_color: tuple[float, float, float, float] = (0.953, 0.612, 0.071, 1.0),
    width: int = 960,
    height: int = 600,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    look_at_bounds: list[float] | None = None,
    camera_position: np.ndarray | None = None,
    camera_focal_point: np.ndarray | None = None,
    camera_view_up: np.ndarray | None = None,
    solar_selection_mesh: trimesh.Trimesh | None = None,
    solar_selection_color: tuple[float, float, float, float] = (1.0, 0.843, 0.0, 1.0),
    solar_selection_texture: pv.Texture | None = None,
    windshield_mesh: trimesh.Trimesh | None = None,
    windshield_color: tuple[float, float, float, float] = (0.31, 0.76, 0.97, 0.65),
    assembly_smooth_shading: bool = True,
    pressure_field: np.ndarray | None = None,
    pressure_field_enabled: bool = False,
) -> np.ndarray:
    """Render mesh and inner assembly using PyVista with Order-Independent Transparency."""
    # 1. Initialize offscreen plotter (removing the multi_samples keyword argument)
    plotter = pv.Plotter(off_screen=True, window_size=[width, height])
    plotter.background_color = bg_color[:3]

    # Configure Anti-Aliasing explicitly via its method instead of the constructor
    plotter.enable_anti_aliasing(aa_type="fxaa")

    # 2. Enable Depth Peeling to fix transparency artifacts & Z-fighting
    plotter.enable_depth_peeling(number_of_peels=4, occlusion_ratio=0.0)

    # 3. Add Inner Assembly (Opaque)
    if assembly_mesh is not None:
        pv_asm = pv.wrap(assembly_mesh)
        plotter.add_mesh(
            pv_asm,
            color=assembly_color[:3],
            ambient=0.3,
            diffuse=0.7,
            smooth_shading=assembly_smooth_shading,
            specular=0.2,
            specular_power=50,
        )

    # 4. Add aeroshell OR pressure-field map on aeroshell surface
    pv_mesh = pv.wrap(mesh)
    if pressure_field_enabled and pressure_field is not None and len(pressure_field) == len(mesh.vertices):
        plotter.add_mesh(
            pv_mesh,
            scalars=np.asarray(pressure_field, dtype=np.float32),
            cmap="turbo",
            opacity=1.0,
            ambient=0.3,
            diffuse=0.7,
            specular=0.1,
            smooth_shading=True,
            show_scalar_bar=False,
        )
    else:
        plotter.add_mesh(
            pv_mesh, color=mesh_color[:3], opacity=mesh_alpha, ambient=0.3, diffuse=0.7, specular=0.1, smooth_shading=True
        )

    if solar_selection_mesh is not None:
        pv_sol = pv.wrap(solar_selection_mesh)
        if solar_selection_texture is not None:
            xmin, xmax, ymin, ymax, zmin, _zmax = pv_sol.bounds
            pv_sol = pv_sol.texture_map_to_plane(
                origin=(xmin, ymin, zmin),
                point_u=(xmax, ymin, zmin),
                point_v=(xmin, ymax, zmin),
                inplace=False,
            )
            plotter.add_mesh(
                pv_sol,
                texture=solar_selection_texture,
                color=(1.0, 1.0, 1.0),
                opacity=solar_selection_color[3],
                ambient=1.0,
                diffuse=0.0,
                specular=0.0,
                smooth_shading=True,
            )
        else:
            plotter.add_mesh(
                pv_sol,
                color=solar_selection_color[:3],
                opacity=solar_selection_color[3],
                ambient=0.3,
                diffuse=0.7,
                smooth_shading=True,
            )

    if windshield_mesh is not None:
        pv_ws = pv.wrap(windshield_mesh)
        plotter.add_mesh(
            pv_ws,
            color=windshield_color[:3],
            opacity=windshield_color[3],
            ambient=0.3,
            diffuse=0.7,
            smooth_shading=True,
        )

    # 5. Position camera
    if camera_position is not None and camera_focal_point is not None and camera_view_up is not None:
        # Match camera lock behavior in latent_rollout_gif.py by reusing one fixed pose.
        plotter.camera.position = tuple(float(v) for v in camera_position)
        plotter.camera.focal_point = tuple(float(v) for v in camera_focal_point)
        plotter.camera.up = tuple(float(v) for v in camera_view_up)
        plotter.reset_camera_clipping_range()
    else:
        plotter.camera_position = "iso"  # Initial isometric view orientation
        plotter.camera.azimuth = azim_deg
        plotter.camera.elevation = elev_deg

        if look_at_bounds is not None:
            plotter.reset_camera(bounds=look_at_bounds)
        else:
            plotter.reset_camera()

    img = plotter.screenshot(return_img=True)
    plotter.close()

    return img[:, :, :3]  # (H, W, 3) uint8 RGB


def _compute_camera_from_center(
    center: np.ndarray,
    diag: float,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    distance_mult: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute camera position/focal-point/up from a scene center and diagonal."""
    dist = diag * distance_mult
    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    cam_pos = center + dist * np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    forward = center - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return cam_pos, center.copy(), up


def _get_prism_center_diag(
    prism_scale: float = _DEFAULT_OUTER_PRISM_SCALE,
    bbox_padding: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Return (center_xyz, diagonal_mm) for the WP2 outer prism extraction bounds."""
    try:
        from subprojects.wp2_assembler.version4.utils.reg_imp import (  # noqa: PLC0415
            get_outer_prism_bounds_mm,
            scale_outer_prism,
        )

        outer = get_outer_prism_bounds_mm()
        scaled = scale_outer_prism(outer, prism_scale)
        prism_x, prism_y, prism_z = scaled["x"], scaled["y"], scaled["z"]
    except Exception:
        prism_x, prism_y, prism_z = (-2900.0, 2900.0), (-1150.0, 1150.0), (0.0, 1650.0)
    vmin = np.array([prism_x[0] - bbox_padding, prism_y[0] - bbox_padding, prism_z[0]], dtype=np.float64)
    vmax = np.array([prism_x[1] + bbox_padding, prism_y[1] + bbox_padding, prism_z[1] + bbox_padding], dtype=np.float64)
    return 0.5 * (vmin + vmax), float(np.linalg.norm(vmax - vmin))


def compute_camera_pose(
    mesh: trimesh.Trimesh,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    distance_mult: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute camera position/focus/up from a mesh bounding box."""
    vertices = mesh.vertices
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    diag = float(np.linalg.norm(vmax - vmin))
    return _compute_camera_from_center(center, diag, azim_deg, elev_deg, distance_mult)


# ---------------------------------------------------------------------------
# 2D latent space panel
# ---------------------------------------------------------------------------

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/etc/X11/fonts/TTF/DejaVuSans.ttf",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in _FONT_PATHS:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _build_solar_cell_texture(
    texture_size: int = 1536,
    cells_x: int = 8,
    cells_y: int = 5,
) -> pv.Texture:
    """Create a procedural texture matching a rectangular industrial PV array."""
    cell_a = (9, 31, 58)
    cell_b = (11, 44, 78)
    seam_color = (150, 160, 170)
    finger_color = (146, 166, 184)
    busbar_color = (202, 206, 210)

    image = Image.new("RGB", (texture_size, texture_size), color=cell_a)
    draw = ImageDraw.Draw(image)

    panel_left = 0
    panel_top = 0
    panel_right = texture_size
    panel_bottom = texture_size

    panel_w = panel_right - panel_left
    panel_h = panel_bottom - panel_top
    gap = max(1, int(min(panel_w / cells_x, panel_h / cells_y) * 0.016))
    cell_w = (panel_w - gap * (cells_x - 1)) / float(cells_x)
    cell_h = (panel_h - gap * (cells_y - 1)) / float(cells_y)
    seam_width = max(1, texture_size // 900)
    finger_width = max(1, texture_size // 900)
    busbar_width = max(1, texture_size // 520)
    fingers_per_cell = 18
    horizontal_lines = 5

    for row in range(cells_y):
        for col in range(cells_x):
            left = round(panel_left + col * (cell_w + gap))
            top = round(panel_top + row * (cell_h + gap))
            right = round(left + cell_w)
            bottom = round(top + cell_h)
            fill = cell_a if (row + col) % 2 == 0 else cell_b
            draw.rectangle(
                ((left, top), (right, bottom)),
                fill=fill,
                outline=seam_color,
                width=seam_width,
            )

            inner_left = left + 2
            inner_right = right - 2
            inner_top = top + 2
            inner_bottom = bottom - 2
            if inner_right > inner_left and inner_bottom > inner_top:
                for k in range(1, fingers_per_cell + 1):
                    frac = k / float(fingers_per_cell + 1.0)
                    x = round(inner_left + frac * (inner_right - inner_left))
                    draw.line(
                        [(x, inner_top), (x, inner_bottom)],
                        fill=finger_color,
                        width=finger_width,
                    )
                for k in range(1, horizontal_lines + 1):
                    frac = k / float(horizontal_lines + 1.0)
                    y = round(inner_top + frac * (inner_bottom - inner_top))
                    draw.line(
                        [(inner_left, y), (inner_right, y)],
                        fill=(84, 112, 142),
                        width=1,
                    )
                y_top = round(inner_top + 0.18 * (inner_bottom - inner_top))
                y_bottom = round(inner_top + 0.82 * (inner_bottom - inner_top))
                draw.line([(inner_left, y_top), (inner_right, y_top)], fill=busbar_color, width=busbar_width)
                draw.line([(inner_left, y_bottom), (inner_right, y_bottom)], fill=busbar_color, width=busbar_width)

    texture = pv.Texture(np.asarray(image))
    texture.repeat = True
    return texture


# ---------------------------------------------------------------------------
# Overlay box constants (shared by both left and right boxes)
# ---------------------------------------------------------------------------
_CARD_BOX_COLOR: tuple[int, int, int] = (20, 24, 30)
_CARD_BOX_ALPHA: float = 0.72
_CARD_GRID_COL: tuple[int, int, int, int] = (130, 150, 195, 110)
_CARD_SECTION_OUTLINE: tuple[int, int, int, int] = (195, 53, 49, 185)


def _box_dimensions(scale: float, show_tacho: bool) -> tuple[int, int]:
    """Return (box_w, box_h) shared by both overlay boxes for the given frame scale.

    Args:
        scale: frame_height / 600.0.
        show_tacho: Whether the left box includes a tachometer section.

    Returns:
        (box_w, box_h) in pixels.
    """
    pad_x = max(8, int(12 * scale))
    pad_y = max(6, int(10 * scale))
    section_gap = max(6, int(8 * scale))
    section_h = max(34, int(46 * scale))
    tacho_radius = max(28, int(58 * scale))
    box_w = max(210, int(258 * scale))
    box_w = max(box_w, int((tacho_radius * 2) + 2 * pad_x + 8))
    # Left box content: drag + solar + (CSE row if no tacho, else tacho arc block)
    n_base = 2 if show_tacho else 3
    box_h = n_base * section_h + max(0, n_base - 1) * section_gap + 2 * pad_y
    if show_tacho:
        tacho_block_h = int((2 * tacho_radius) + max(40, int(52 * scale)))
        box_h += section_gap + tacho_block_h
    return box_w, box_h


def box_builder(
    box_w: int,
    box_h: int,
    scale: float,
    box_color: tuple[int, int, int] = _CARD_BOX_COLOR,
    box_alpha: float = _CARD_BOX_ALPHA,
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Create the shared semi-transparent card background used by both overlay boxes.

    Draws a rounded-rect fill, a subtle grid, and a red outer border.
    Callers receive (img, draw) and render content at local (0, 0) coordinates.

    Args:
        box_w: Card width in pixels.
        box_h: Card height in pixels.
        scale: frame_height / 600.0 for DPI-independent sizing.
        box_color: RGB background fill color.
        box_alpha: Opacity in [0.0, 1.0] — same key for both boxes.

    Returns:
        (img, draw) — RGBA PIL Image and its draw context.
    """
    img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    a = int(max(0.0, min(1.0, box_alpha)) * 255)
    radius = max(8, int(10 * scale))
    # Filled rounded rect
    draw.rounded_rectangle(
        ((0, 0), (box_w - 1, box_h - 1)),
        radius=radius,
        fill=(box_color[0], box_color[1], box_color[2], a),
        outline=None,
    )
    # Grid — clipped to rounded corners via a binary mask so corners stay round
    grid_step = max(28, int(40 * scale))
    for gx in range(1, box_w, grid_step):
        draw.line([(gx, 1), (gx, box_h - 2)], fill=_CARD_GRID_COL, width=1)
    for gy in range(1, box_h, grid_step):
        draw.line([(1, gy), (box_w - 2, gy)], fill=_CARD_GRID_COL, width=1)
    # Clip alpha channel to rounded rect so grid lines don't bleed into sharp corners
    clip_mask = Image.new("L", (box_w, box_h), 0)
    ImageDraw.Draw(clip_mask).rounded_rectangle(((0, 0), (box_w - 1, box_h - 1)), radius=radius, fill=255)
    cur_alpha = np.asarray(img.split()[3], dtype=np.uint16)
    mask_np = np.asarray(clip_mask, dtype=np.uint16)
    new_alpha = ((cur_alpha * mask_np) // 255).astype(np.uint8)
    r_ch, g_ch, b_ch, _ = img.split()
    img = Image.merge("RGBA", (r_ch, g_ch, b_ch, Image.fromarray(new_alpha)))
    # Red outer border drawn after clipping so it sits cleanly on top
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rounded_rectangle(
        ((0, 0), (box_w - 1, box_h - 1)),
        radius=radius,
        fill=None,
        outline=_CARD_SECTION_OUTLINE,
        width=max(1.2, int(1.2 * scale)),
    )
    return img, draw


def box_position(
    frame_w: int,
    frame_h: int,
    box_w: int,
    box_h: int,
    margin_x: int,
) -> tuple[int, int, int, int]:
    """Return pixel positions for both overlay boxes.

    Both boxes are vertically centered on the frame.
    Left box is margin_x from the left edge; right box is margin_x from the right edge.

    Args:
        frame_w: Full composite frame width in pixels.
        frame_h: Full composite frame height in pixels.
        box_w: Width of each box (both use the same width).
        box_h: Height of each box.
        margin_x: Horizontal gap from each respective frame edge.

    Returns:
        (left_x, box_y, right_x, box_y) — top-left corners for left and right boxes.
    """
    box_y = (frame_h - box_h) // 2
    left_x = margin_x
    right_x = frame_w - margin_x - box_w
    return left_x, box_y, right_x, box_y


def _composite_box_on_frame(
    frame_rgb: np.ndarray,
    box_img: Image.Image,
    x: int,
    y: int,
) -> np.ndarray:
    """Alpha-composite an RGBA card image onto an RGB frame at (x, y).

    Args:
        frame_rgb: (H, W, 3) uint8 numpy array.
        box_img: RGBA PIL Image produced by box_builder.
        x: Left edge destination x in the frame.
        y: Top edge destination y in the frame.

    Returns:
        (H, W, 3) uint8 numpy array with the card composited in.
    """
    frame = Image.fromarray(frame_rgb).convert("RGBA")
    dest_x = max(0, x)
    dest_y = max(0, y)
    crop_w = min(box_img.width, frame_rgb.shape[1] - dest_x)
    crop_h = min(box_img.height, frame_rgb.shape[0] - dest_y)
    if crop_w <= 0 or crop_h <= 0:
        return frame_rgb
    frame.alpha_composite(box_img.crop((0, 0, crop_w, crop_h)), dest=(dest_x, dest_y))
    return np.asarray(frame.convert("RGB"))


def draw_right_box(  # noqa: PLR0915
    draw: ImageDraw.ImageDraw,
    box_w: int,
    box_h: int,
    scale: float,
    z0: float,
    z1: float,
    z_radius: float,
    Z_pretrained: np.ndarray | None,
    frame_idx: int,
    n_frames: int,
    mode: str,
    trajectory_coords: list[tuple[float, float]] | None,
    panel_scale_radius: float,
    show_anchors: bool,
    colors: dict | None,
) -> None:
    """Draw the latent-space map at local box coordinates (0, 0).

    Args:
        draw: ImageDraw context from box_builder.
        box_w: Box width in pixels.
        box_h: Box height in pixels.
        scale: frame_height / 600.0.
        z0: Current latent z₀ coordinate.
        z1: Current latent z₁ coordinate.
        z_radius: Training manifold radius for the circle overlay.
        Z_pretrained: (N, 2) array of pretrained anchor coordinates, or None.
        frame_idx: Current frame index (for progress arc).
        n_frames: Total frame count.
        mode: Traversal mode ("circle", "spiral", "line", "cross", "epoch_gif").
        trajectory_coords: Full trajectory as [(z0, z1), ...], or None.
        panel_scale_radius: Effective radius used to scale the latent map view.
        show_anchors: Whether to draw pretrained anchor dots.
        colors: Optional colour overrides dict (same keys as panel_colors in main).
    """
    _C = colors or {}
    axis_col = _C.get("axis", (75, 85, 115))
    circle_col = _C.get("circle", (110, 125, 165))
    anchor_col = _C.get("anchor", (220, 90, 90))
    anchor_lbl_col = _C.get("anchor_label", (240, 160, 160))
    progress_col = _C.get("progress", (80, 200, 255))
    cursor_col = _C.get("cursor", (80, 200, 255))
    cursor_out_col = _C.get("cursor_outline", (255, 255, 255))
    axis_lbl_col = _C.get("axis_label", (180, 190, 220))

    lw_axis = max(1, int(1 * scale))
    lw_thick = max(1, int(2 * scale))
    pad_x = max(8, int(12 * scale))
    pad_y = max(6, int(10 * scale))

    inner_x0 = pad_x
    inner_y0 = pad_y
    inner_x1 = box_w - pad_x
    inner_y1 = box_h - pad_y
    inner_w = inner_x1 - inner_x0
    inner_h = inner_y1 - inner_y0
    icx = inner_x0 + inner_w // 2
    icy = inner_y0 + inner_h // 2

    map_scale = (min(inner_w, inner_h) * 0.40) / max(panel_scale_radius * 1.5, 1e-6)

    def z2px(z0v: float, z1v: float) -> tuple[int, int]:
        return icx + int(z0v * map_scale), icy - int(z1v * map_scale)

    # Axis lines.
    draw.line([(inner_x0 + 4, icy), (inner_x1 - 4, icy)], fill=axis_col, width=lw_axis)
    draw.line([(icx, inner_y0 + 4), (icx, inner_y1 - 4)], fill=axis_col, width=lw_axis)

    # Training manifold circle.
    r_px = int(z_radius * map_scale)
    if mode in ("circle", "spiral", "epoch_gif"):
        draw.ellipse(
            [(icx - r_px, icy - r_px), (icx + r_px, icy + r_px)],
            outline=circle_col,
            width=lw_axis,
        )

    # Pretrained anchor dots.
    if show_anchors and Z_pretrained is not None:
        for k, (z0a, z1a) in enumerate(Z_pretrained):
            apx, apy = z2px(float(z0a), float(z1a))
            r_a = max(2, int(4 * scale))
            off = max(3, int(7 * scale))
            draw.ellipse([(apx - r_a, apy - r_a), (apx + r_a, apy + r_a)], fill=anchor_col)
            draw.text((apx + off, apy - off), str(k), fill=anchor_lbl_col, font=_get_font(max(8, int(11 * scale))))

    # Trajectory / progress.
    if mode == "circle":
        if n_frames > 1 and r_px > 5:
            arc_end_deg = -90 + (360.0 * frame_idx / n_frames)
            with contextlib.suppress(Exception):
                draw.arc(
                    [(icx - r_px, icy - r_px), (icx + r_px, icy + r_px)],
                    start=-90,
                    end=arc_end_deg,
                    fill=progress_col,
                    width=lw_thick,
                )
    elif trajectory_coords is not None and len(trajectory_coords) > 1:
        full_pts = [z2px(zc0, zc1) for zc0, zc1 in trajectory_coords]
        draw.line(full_pts, fill=circle_col, width=lw_axis)
        traversed = full_pts[: frame_idx + 1]
        if len(traversed) >= 2:
            draw.line(traversed, fill=progress_col, width=lw_thick)
        if mode == "line":
            ep_r = max(2, int(5 * scale))
            for ep_z0, ep_z1 in (trajectory_coords[0], trajectory_coords[-1]):
                ex, ey = z2px(ep_z0, ep_z1)
                draw.ellipse([(ex - ep_r, ey - ep_r), (ex + ep_r, ey + ep_r)], outline=circle_col, width=lw_thick)
        if mode == "cross":
            ep_r = max(2, int(5 * scale))
            seen: set[tuple[int, int]] = set()
            for ep_z0, ep_z1 in trajectory_coords:
                ep = z2px(ep_z0, ep_z1)
                if ep not in seen and abs(ep_z0) > 1e-6 and abs(ep_z1) > 1e-6:
                    draw.ellipse([(ep[0] - ep_r, ep[1] - ep_r), (ep[0] + ep_r, ep[1] + ep_r)], outline=circle_col, width=lw_thick)
                    seen.add(ep)

    # Current cursor.
    px_c, py_c = z2px(z0, z1)
    r_cursor = max(3, int(7 * scale))
    draw.ellipse(
        [(px_c - r_cursor, py_c - r_cursor), (px_c + r_cursor, py_c + r_cursor)],
        fill=cursor_col,
        outline=cursor_out_col,
        width=lw_thick,
    )

    # Axis labels.
    font_label = _get_font(max(9, int(13 * scale)))
    draw.text((inner_x1 - int(26 * scale), icy - int(16 * scale)), "z\u2080", fill=axis_lbl_col, font=font_label)
    draw.text((icx + int(5 * scale), inner_y0 + int(8 * scale)), "z\u2081", fill=axis_lbl_col, font=font_label)

    # Coordinate readout at bottom of box.
    coord_text = f"z = [{z0:+.3f}, {z1:+.3f}]"
    coord_font = _get_font(max(8, int(11 * scale)))
    coord_bbox = draw.textbbox((0, 0), coord_text, font=coord_font)
    coord_w = coord_bbox[2] - coord_bbox[0]
    coord_h_px = coord_bbox[3] - coord_bbox[1]
    draw.text(
        ((box_w - coord_w) // 2, box_h - max(8, int(12 * scale)) - coord_h_px),
        coord_text,
        fill=_C.get("coord_text", (130, 195, 255)),
        font=coord_font,
    )


def draw_epoch_label_on_frame(
    frame_rgb: np.ndarray,
    epoch_num: int | None,
    color: tuple[int, int, int] = (200, 205, 230),
    x_offset_px: int = 0,
    box_color: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Overlay the epoch label at top-left directly onto an RGB frame.

    Args:
        frame_rgb: Full RGB frame.
        epoch_num: Epoch number to render; if None no text is drawn.
        color: RGB text color.
        x_offset_px: Extra x-offset in pixels for the label anchor.
        box_color: Optional RGB fill for a background box behind the label.
    """
    if epoch_num is None:
        return frame_rgb

    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    h, w = frame_rgb.shape[:2]
    scale_factor = h / 600.0
    font_epoch = _get_font(max(10, int(15 * scale_factor)))
    label = f"Epoch {epoch_num}"
    bbox = draw.textbbox((0, 0), label, font=font_epoch)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    margin_bottom = max(8, int(12 * scale_factor))
    anchor_x = (w - text_w) // 2 + x_offset_px
    anchor_y = h - text_h - margin_bottom
    if box_color is not None:
        pad = max(4, int(4 * scale_factor))
        draw.rectangle(
            ((anchor_x - pad, anchor_y - pad), (anchor_x + text_w + pad, anchor_y + text_h + pad)),
            fill=box_color,
        )
    draw.text(
        (anchor_x, anchor_y),
        label,
        fill=color,
        font=font_epoch,
    )
    return np.asarray(img)


@contextlib.contextmanager
def _suppress_wandb_logging() -> Any:
    """Temporarily silence wandb.log for postprocessing-only metric inference."""
    try:
        import wandb  # noqa: PLC0415
    except Exception:
        yield
        return

    original_log = getattr(wandb, "log", None)
    if original_log is None:
        yield
        return

    wandb.log = lambda *_args, **_kwargs: None  # type: ignore[method-assign, assignment]
    try:
        yield
    finally:
        wandb.log = original_log  # type: ignore[method-assign]


def _load_wp4_default_cfd_config() -> dict[str, Any]:
    """Load default CFD config from WP4 v2 losses config."""
    cfg_path = Path(__file__).parents[1] / "wp4_aeroshell_optimizer" / "v2" / "configs" / "losses_v0_config.yaml"
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] failed to read default CFD config: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return {}
    return cfg.get("cfd", {}) if isinstance(cfg, dict) else {}


def _mesh_has_valid_face_indices(mesh: trimesh.Trimesh) -> bool:
    """Return True when all triangle indices are within vertex bounds."""
    try:
        faces = np.asarray(mesh.faces)
        n_verts = len(mesh.vertices)
    except Exception:
        return False

    if faces.ndim != 2 or faces.shape[1] != 3 or n_verts <= 0:
        return False
    if faces.size == 0:
        return False
    return int(faces.min()) >= 0 and int(faces.max()) < n_verts


def _init_gaot_drag_estimator(
    gaot_version: int,
    cfd_config: dict[str, Any],
    device: torch.device,
) -> Any | None:
    """Instantiate GAOT loader by version for drag estimation on extracted meshes."""
    try:
        if gaot_version == 3:
            from subprojects.postprocessing_visuals.cp_field_viz import GAOTPlyInference  # noqa: PLC0415

            return GAOTPlyInference(cfd_config=cfd_config, device=str(device))

        _ensure_wp4_model_imports()
        _GaotLoader: type[Any]
        if gaot_version == 1:
            from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT_working.model_loader import ModelLoader as _GaotLoader  # noqa: PLC0415
        elif gaot_version == 2:
            from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.model_loader import ModelLoader as _GaotLoader  # noqa: PLC0415
        else:
            warnings.warn(
                f"[latent_rollout] unsupported GAOT version: {gaot_version}",
                UserWarning,
                stacklevel=2,
            )
            return None
        return _GaotLoader(cfd_config=cfd_config, device=str(device))
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] failed to initialize GAOT drag estimator: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def _estimate_drag_newtons(  # noqa: PLR0912
    gaot_loader: Any,
    gaot_version: int,
    sdf_model: Any,
    mesh: trimesh.Trimesh,
    frame_step: int,
    gaot_pressure_field: Tensor | None = None,
) -> float | None:
    """Estimate drag (N) from an extracted mesh via GAOT internals.

    For GAOT3 inference, mirror the training-time composition:
    pressure-drag integration + shear-force estimate.
    """
    try:
        if not _mesh_has_valid_face_indices(mesh):
            warnings.warn(
                "[latent_rollout] skipping GAOT drag estimate because mesh face indices are invalid.",
                UserWarning,
                stacklevel=2,
            )
            return None

        loader_device = torch.device(getattr(gaot_loader, "device", "cpu"))
        verts_t = torch.tensor(mesh.vertices, dtype=torch.float32, device=loader_device)
        faces_t = torch.tensor(mesh.faces, dtype=torch.long, device=loader_device)

        # GAOT3 uses the read-only postprocessing inference path to avoid SDF-model
        # coupling and prevent CSE overlay failures from poisoning the render CUDA context.
        if gaot_version == 3 and hasattr(gaot_loader, "_preprocess") and hasattr(gaot_loader, "_latent_tokens"):
            from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.shear_force import ShearForce  # noqa: PLC0415
            from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.pressure_integration import cp_field_to_drag  # noqa: PLC0415

            with torch.no_grad(), _suppress_wandb_logging():
                batch = gaot_loader._preprocess(verts_t, faces_t)
                if gaot_pressure_field is None:
                    output = gaot_loader.gaot.forward(batch=batch, tokens_pos=gaot_loader._latent_tokens)
                    pressure_field = output["field"]
                else:
                    pressure_field = gaot_pressure_field

                # Match GAOT3 ModelLoader: integrate with model-space geometry (x-flipped).
                integration_normals = batch.c if hasattr(batch, "c") and batch.c is not None else batch.c_raw
                integration_pos = batch.pos_raw if hasattr(batch, "pos_raw") and batch.pos_raw is not None else (verts_t * 1e-3)
                pressure_drag_tensor = cp_field_to_drag(
                    cp_field=pressure_field,
                    normals=integration_normals,
                    face=faces_t.T,
                    pos=integration_pos,
                    v_inf=float(gaot_loader.cfd_config["v_inf"]),
                    rho=float(gaot_loader.cfd_config["rho"]),
                )

                if not hasattr(gaot_loader, "_shear_force_calculator"):
                    gaot_loader._shear_force_calculator = ShearForce(unit="mm^2")
                shear_drag_tensor = gaot_loader._shear_force_calculator.calculate_shear_force(verts_t, faces_t)
                drag_tensor = pressure_drag_tensor + shear_drag_tensor

            drag_val = float(drag_tensor.detach().cpu().item())
            if not np.isfinite(drag_val):
                return None
            if drag_val < 0.0:
                warnings.warn(
                    f"[latent_rollout] GAOT drag came back negative ({drag_val:.3f} N); using abs() for CSE compatibility.",
                    UserWarning,
                    stacklevel=2,
                )
                drag_val = abs(drag_val)
            return drag_val

        gaot_loader.epoch_name = int(frame_step)
        if gaot_version == 3:
            gaot_loader.sdf_model = sdf_model

        with _suppress_wandb_logging():
            drag_tensor = gaot_loader.loss_fn(verts_t, faces_t)

        drag_val = float(drag_tensor.detach().cpu().item()) if torch.is_tensor(drag_tensor) else float(drag_tensor)

        diagnostics = getattr(gaot_loader, "_last_diagnostics", {})
        if isinstance(diagnostics, dict) and "drag_value" in diagnostics:
            drag_val = float(diagnostics["drag_value"])

        if not np.isfinite(drag_val):
            return None
        if drag_val < 0.0:
            warnings.warn(
                f"[latent_rollout] GAOT drag came back negative ({drag_val:.3f} N); using abs() for CSE compatibility.",
                UserWarning,
                stacklevel=2,
            )
            drag_val = abs(drag_val)
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] GAOT drag estimate failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return drag_val


def _estimate_pressure_field_cp(
    gaot_loader: Any,
    gaot_version: int,
    mesh: trimesh.Trimesh,
) -> Tensor | None:
    """Estimate GAOT pressure field (Cp) per vertex for visualization."""
    if gaot_version != 3:
        return None
    if not hasattr(gaot_loader, "_preprocess") or not hasattr(gaot_loader, "_latent_tokens"):
        return None
    if not _mesh_has_valid_face_indices(mesh):
        return None

    try:
        loader_device = torch.device(getattr(gaot_loader, "device", "cpu"))
        verts_t = torch.tensor(mesh.vertices, dtype=torch.float32, device=loader_device)
        faces_t = torch.tensor(mesh.faces, dtype=torch.long, device=loader_device)
        with torch.no_grad(), _suppress_wandb_logging():
            batch = gaot_loader._preprocess(verts_t, faces_t)
            output = gaot_loader.gaot.forward(batch=batch, tokens_pos=gaot_loader._latent_tokens)
            pressure_field = output["field"]
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] pressure-field estimate failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None
    else:
        return pressure_field


def _estimate_solar_power_watts(
    mesh: trimesh.Trimesh,
    device: torch.device,
    sun_dir_xyz: tuple[float, float, float],
    area_budget_mm2: float,
    effective_irradiance_w_m2: float,
    cone_face_mask: Tensor | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate solar power (W) with SR solar objective on the current mesh."""
    try:
        if not _mesh_has_valid_face_indices(mesh):
            warnings.warn(
                "[latent_rollout] skipping solar estimate because mesh face indices are invalid.",
                UserWarning,
                stacklevel=2,
            )
            return None, {}

        from subprojects.wp3_solar_estimator.rollout_losses.SR_SolarLoss import solar_loss_sr  # noqa: PLC0415

        verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
        faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        sun_dir = torch.tensor(sun_dir_xyz, dtype=torch.float32, device=device)
        cone_mask = None
        if cone_face_mask is not None and int(cone_face_mask.numel()) == int(faces.shape[0]):
            cone_mask = cone_face_mask.to(device=device, dtype=torch.bool)

        power_w, diagnostics = solar_loss_sr(
            vertices=verts,
            faces=faces,
            sun_dir=sun_dir,
            area_budget_mm2=float(area_budget_mm2),
            effective_irradiance_W_m2=float(effective_irradiance_w_m2),
            cone_face_mask=cone_mask,
            epoch_name=None,
        )
        power_val = float(power_w.detach().cpu().item())
        if not np.isfinite(power_val):
            return None, diagnostics
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] solar estimate failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None, {}
    else:
        return power_val, diagnostics


def _compute_cse_speed_ms(
    drag_force_newton: float,
    solar_power_watts: float,
    mass_kg: float,
    constant_losses_w: float,
    rolling_friction_coefficient: float,
) -> float | None:
    """Compute CSE cruising speed [m/s] from drag + solar estimates."""
    if drag_force_newton <= 0.0:
        return None
    try:
        from subprojects.wp4_aeroshell_optimizer.v1.src.losses.cruising_speed import cruising_speed  # noqa: PLC0415

        drag_t = torch.tensor(float(drag_force_newton), dtype=torch.float32)
        solar_t = torch.tensor(float(max(solar_power_watts, 0.0)), dtype=torch.float32)
        speed_ms, _diag = cruising_speed(
            solar_power=solar_t,
            drag_force_at_25ms=drag_t,
            mass=float(mass_kg),
            c_losses=float(constant_losses_w),
            rolling_friction_coefficient=float(rolling_friction_coefficient),
            max_iter=12,
            tol=1e-4,
        )
        speed_val = float(speed_ms.detach().cpu().item())
        return speed_val if np.isfinite(speed_val) else None
    except Exception as exc:
        warnings.warn(
            f"[latent_rollout] CSE computation failed: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return None


def draw_left_box(  # noqa: PLR0915
    draw: ImageDraw.ImageDraw,
    box_w: int,
    box_h: int,
    scale: float,
    box_color: tuple[int, int, int],
    drag_n: float | None,
    solar_w: float | None,
    cse_ms: float | None,
    show_tacho: bool,
    text_color: tuple[int, int, int],
    show_kmh: bool = False,
    tacho_max_kmh: float = 160.0,
) -> None:
    """Draw Drag / Solar / CSE (or tachometer) at local box coordinates (0, 0).

    Args:
        draw: ImageDraw context from box_builder.
        box_w: Box width in pixels.
        box_h: Box height in pixels.
        scale: frame_height / 600.0.
        box_color: RGB fill used for inner section backgrounds.
        drag_n: Drag force in Newtons, or None.
        solar_w: Solar power in Watts, or None.
        cse_ms: CSE speed in m/s, or None.
        show_tacho: Render a tachometer instead of a plain CSE row.
        text_color: RGB text color.
        show_kmh: Append km/h conversion to CSE text (ignored when show_tacho=True).
    """
    label_font = _get_font(max(10, int(11 * scale)))
    value_font = _get_font(max(10, int(13 * scale)))
    caption_font = _get_font(max(9, int(10 * scale)))
    pad_x = max(8, int(12 * scale))
    pad_y = max(6, int(10 * scale))
    section_gap = max(6, int(8 * scale))
    section_h = max(34, int(46 * scale))
    tacho_radius = max(28, int(58 * scale))
    tacho_kmh_max = tacho_max_kmh

    drag_str = f"{drag_n:.1f} N" if drag_n is not None else "n/a"
    solar_str = f"{solar_w:.1f} W" if solar_w is not None else "n/a"
    sections_to_draw: list[tuple[str, str]] = [("Drag", drag_str), ("Solar", solar_str)]
    if not show_tacho:
        if cse_ms is not None:
            cse_str = f"{cse_ms:.2f} m/s"
            if show_kmh:
                cse_str = f"{cse_str} ({cse_ms * 3.6:.1f} km/h)"
        else:
            cse_str = "n/a"
        sections_to_draw.append(("CSE", cse_str))

    def _section(y_top: int, title: str, value: str) -> int:
        y_bot = y_top + section_h
        draw.rounded_rectangle(
            ((pad_x, y_top), (box_w - pad_x, y_bot)),
            radius=max(6, int(8 * scale)),
            fill=None,
            outline=_CARD_SECTION_OUTLINE,
            width=max(1.2, int(1.2 * scale)),
        )
        txt_x = pad_x + max(8, int(10 * scale))
        draw.text((txt_x, y_top + max(4, int(6 * scale))), title, fill=text_color, font=label_font)
        draw.text((txt_x, y_top + max(18, int(24 * scale))), value, fill=text_color, font=value_font)
        return y_bot

    y = pad_y
    for si, (st, sv) in enumerate(sections_to_draw):
        y = _section(y, st, sv)
        if si < len(sections_to_draw) - 1:
            y += section_gap
    y += section_gap

    if show_tacho:
        tacho_top = y
        tacho_bottom = box_h - pad_y
        draw.rounded_rectangle(
            ((pad_x, tacho_top), (box_w - pad_x, tacho_bottom)),
            radius=max(6, int(8 * scale)),
            fill=None,
            outline=_CARD_SECTION_OUTLINE,
            width=max(1, int(1 * scale)),
        )
        caption = "Cruising Speed Estimate"
        cap_box = draw.textbbox((0, 0), caption, font=caption_font)
        cap_w = cap_box[2] - cap_box[0]
        draw.text(
            ((box_w - cap_w) // 2, tacho_top + max(4, int(6 * scale))),
            caption,
            fill=text_color,
            font=caption_font,
        )
        cx = box_w // 2
        cy = tacho_top + max(20, int(26 * scale)) + tacho_radius
        # 210 degrees is the bottom-left (0 km/h)
        # Sweeping -240 degrees moves clockwise over the top arch to the bottom-right
        arc_start_deg = 210.0
        arc_span_deg = -240.0
        arc_pts_outer: list[tuple[int, int]] = []
        arc_pts_inner: list[tuple[int, int]] = []
        for k in range(81):
            frac = k / 80.0
            angle = math.radians(arc_start_deg + (arc_span_deg * frac))
            ox = cx + int(tacho_radius * math.cos(angle))
            oy = cy - int(tacho_radius * math.sin(angle))
            ix = cx + int((tacho_radius - max(6, int(8 * scale))) * math.cos(angle))
            iy = cy - int((tacho_radius - max(6, int(8 * scale))) * math.sin(angle))
            arc_pts_outer.append((ox, oy))
            arc_pts_inner.append((ix, iy))
        if len(arc_pts_outer) >= 2 and len(arc_pts_inner) >= 2:
            glow_poly = arc_pts_outer + list(reversed(arc_pts_inner))
            draw.polygon(glow_poly, fill=(box_color[0], box_color[1], box_color[2], 155))
            draw.line(arc_pts_outer, fill=text_color, width=max(1.2, int(1.2 * scale)))
        tick_col: tuple[int, int, int] = (
            int(0.75 * text_color[0]),
            int(0.75 * text_color[1]),
            int(0.75 * text_color[2]),
        )
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            angle = math.radians(arc_start_deg + (arc_span_deg * frac))
            x_in = cx + int((tacho_radius - max(5, int(6 * scale))) * math.cos(angle))
            y_in = cy - int((tacho_radius - max(5, int(6 * scale))) * math.sin(angle))
            x_out = cx + int(tacho_radius * math.cos(angle))
            y_out = cy - int(tacho_radius * math.sin(angle))
            draw.line([(x_in, y_in), (x_out, y_out)], fill=tick_col, width=max(1.2, int(1.2 * scale)))
        if cse_ms is not None and np.isfinite(cse_ms):
            kmh = max(0.0, float(cse_ms) * 3.6)
            needle_frac = max(0.0, min(1.0, kmh / tacho_kmh_max))
            angle = math.radians(arc_start_deg + (arc_span_deg * needle_frac))
            nx = cx + int((tacho_radius - max(6, int(8 * scale))) * math.cos(angle))
            ny = cy - int((tacho_radius - max(6, int(8 * scale))) * math.sin(angle))
            needle_col = (235, 76, 66)
            draw.line([(cx, cy), (nx, ny)], fill=needle_col, width=max(1.2, int(1.2 * scale)))
            hub_r = max(2, int(3 * scale))
            draw.ellipse([(cx - hub_r, cy - hub_r), (cx + hub_r, cy + hub_r)], fill=needle_col)
            speed_txt = f"{kmh:.1f} km/h"
        else:
            speed_txt = "n/a"
        txt_box = draw.textbbox((0, 0), speed_txt, font=value_font)
        txt_w = txt_box[2] - txt_box[0]
        draw.text((cx - txt_w // 2, cy + max(14, int(18 * scale))), speed_txt, fill=text_color, font=value_font)


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------


def _resample_polyline_equal_arclength(
    points: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Resample a 2D polyline to approximately equal arc-length spacing."""
    if n_samples <= 1 or points.shape[0] <= 1:
        return points[:1].copy()

    diffs = points[1:] - points[:-1]
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(points[:1], repeats=n_samples, axis=0)

    targets = np.linspace(0.0, total, n_samples)
    out = np.empty((n_samples, 2), dtype=np.float64)

    # Monotonic piecewise-linear interpolation along cumulative arc-length.
    out[:, 0] = np.interp(targets, cum, points[:, 0])
    out[:, 1] = np.interp(targets, cum, points[:, 1])
    return out


def _spiral_point(
    t: float,
    r_start: float,
    r_end: float,
    turns: float,
) -> tuple[float, float]:
    """Evaluate Archimedean spiral point for normalized parameter t in [0, 1]."""
    tt = min(max(float(t), 0.0), 1.0)
    r = r_start + (r_end - r_start) * tt
    a = 2.0 * math.pi * turns * tt
    return (r * math.cos(a), r * math.sin(a))


def _build_spiral_equal_chord_coords(
    n_frames: int,
    r_start: float,
    r_end: float,
    turns: float,
) -> list[tuple[float, float]]:
    """Sample spiral with (approximately) equal Euclidean distance between neighbours."""
    if n_frames <= 1:
        return [_spiral_point(0.0, r_start, r_end, turns)]

    def _dist_sq(t0: float, t1: float) -> float:
        x0, y0 = _spiral_point(t0, r_start, r_end, turns)
        x1, y1 = _spiral_point(t1, r_start, r_end, turns)
        dx = x1 - x0
        dy = y1 - y0
        return dx * dx + dy * dy

    def _advance_by_chord(t_prev: float, chord: float) -> tuple[float, bool]:
        """Find t > t_prev such that ||p(t)-p(t_prev)|| = chord via bisection."""
        if chord <= 0.0:
            return t_prev, True

        target_sq = chord * chord
        max_sq = _dist_sq(t_prev, 1.0)
        if max_sq < target_sq:
            return 1.0, False

        lo = t_prev
        hi = 1.0
        for _ in range(46):
            mid = 0.5 * (lo + hi)
            if _dist_sq(t_prev, mid) >= target_sq:
                hi = mid
            else:
                lo = mid
        return hi, True

    # Search a chord length that lands the last sample near t=1 after n_frames-1 steps.
    lo_d = 0.0
    hi_d = math.sqrt(_dist_sq(0.0, 1.0))
    best_ts = [0.0] * n_frames

    for _ in range(36):
        d_mid = 0.5 * (lo_d + hi_d)
        ts = [0.0]
        t_cur = 0.0
        all_ok = True

        for _k in range(n_frames - 1):
            t_next, ok = _advance_by_chord(t_cur, d_mid)
            ts.append(t_next)
            t_cur = t_next
            if not ok:
                all_ok = False
                break

        # Pad to full length if we terminated early.
        while len(ts) < n_frames:
            ts.append(1.0)

        best_ts = ts
        if all_ok and ts[-1] < 1.0:
            # Chord too small: did not reach end by final frame.
            lo_d = d_mid
        else:
            # Chord too large or just enough: lower upper bound.
            hi_d = d_mid

    # Ensure exact final endpoint.
    best_ts[-1] = 1.0
    return [_spiral_point(t, r_start, r_end, turns) for t in best_ts]


def _build_trajectory(  # noqa: PLR0912  # supports 4 traversal modes; branches are mode dispatches, not complexity
    mode: str,
    n_frames: int,
    z_radius: float,
    spiral_turns: float = 3.0,
    spiral_radius_start: float = 0.0,
    spiral_radius_end: float | None = None,
    spiral_equal_distance: bool = False,
    line_start: tuple[float, float] = (0.0, -2.0),
    line_end: tuple[float, float] = (0.0, 2.0),
    line_bounces: int = 1,
    cross_radius: float | None = None,
) -> tuple[list[tuple[float, float]], float]:
    """Build the (z0, z1) trajectory and return the panel scale radius.

    Args:
        mode: Traversal mode — "circle", "spiral", "line", or "cross".
        n_frames: Total number of animation frames.
        z_radius: Reference radius inferred from the checkpoint.
        spiral_turns: Full rotations for spiral mode.
        spiral_radius_start: Inner radius for spiral (0.0 = start at origin).
        spiral_radius_end: Outer radius for spiral (None → use z_radius).
        spiral_equal_distance: If True, resample spiral to equal arc-length spacing.
        line_start: (z0, z1) start point for line mode.
        line_end: (z0, z1) end point for line mode.
        line_bounces: Complete back-and-forth oscillations for line mode.
        cross_radius: Half-width of the X pattern (None → use z_radius).

    Returns:
        z_coords: List of (z0, z1) tuples, one per frame.
        panel_scale_radius: Reference radius used to scale the latent panel.
    """
    if mode == "circle":
        angles = [2.0 * math.pi * i / max(n_frames, 1) for i in range(n_frames)]
        coords: list[tuple[float, float]] = [(z_radius * math.cos(a), z_radius * math.sin(a)) for a in angles]
        return coords, z_radius

    if mode == "spiral":
        r_end = spiral_radius_end if spiral_radius_end is not None else z_radius
        if n_frames <= 1:
            return [(spiral_radius_start, 0.0)], max(r_end, 1e-6)

        if spiral_equal_distance:
            coords = _build_spiral_equal_chord_coords(
                n_frames=n_frames,
                r_start=spiral_radius_start,
                r_end=r_end,
                turns=spiral_turns,
            )
        else:
            coords = []
            for i in range(n_frames):
                t = i / max(n_frames - 1, 1)
                r = spiral_radius_start + (r_end - spiral_radius_start) * t
                angle = 2.0 * math.pi * spiral_turns * t
                coords.append((r * math.cos(angle), r * math.sin(angle)))
        return coords, max(r_end, 1e-6)

    if mode == "line":
        x0, y0 = line_start
        x1, y1 = line_end
        total_segs = 2 * max(line_bounces, 1)
        coords = []
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1) if n_frames > 1 else 0.0
            scaled = t * total_segs
            seg = int(scaled)
            if seg >= total_segs:
                # t == 1.0 exactly: clamp to end of last segment
                seg = total_segs - 1
                frac = 1.0
            else:
                frac = scaled - seg  # in [0, 1)
            if seg % 2 == 0:
                coords.append((x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac))
            else:
                coords.append((x1 + (x0 - x1) * frac, y1 + (y0 - y1) * frac))
        scale_r = max(abs(x0), abs(y0), abs(x1), abs(y1), z_radius, 1e-6)
        return coords, scale_r

    if mode == "cross":
        r = cross_radius if cross_radius is not None else z_radius
        # Visit all 4 diagonal corners (±r, ±r) through the origin, forming an X.
        # All 8 segments have equal length r√2, so frames are distributed evenly.
        waypoints: list[tuple[float, float]] = [
            (r, r),
            (0.0, 0.0),
            (-r, -r),
            (0.0, 0.0),
            (r, -r),
            (0.0, 0.0),
            (-r, r),
            (0.0, 0.0),
            (r, r),
        ]
        n_segs = len(waypoints) - 1
        cross_coords: list[tuple[float, float]] = []
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1) if n_frames > 1 else 0.0
            scaled = t * n_segs
            seg = int(scaled)
            if seg >= n_segs:
                seg = n_segs - 1
                frac = 1.0
            else:
                frac = scaled - seg
            x0a, y0a = waypoints[seg]
            x1a, y1a = waypoints[seg + 1]
            cross_coords.append((x0a + (x1a - x0a) * frac, y0a + (y1a - y0a) * frac))
        return cross_coords, max(r, z_radius, 1e-6)

    raise ValueError(f"Unknown traversal mode '{mode}'. Expected: circle | spiral | line | cross")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0912, PLR0915  # CLI entrypoint; high statement count from argument parsing and frame loop
    # ------------------------------------------------------------------
    # Step 1: pre-parse --config so YAML can supply defaults for everything
    # ------------------------------------------------------------------
    pv.set_jupyter_backend("static")
    _default_config = Path(__file__).parent / "latent_rollout_config.yaml"

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=_default_config if _default_config.exists() else None)
    pre_args, _ = pre.parse_known_args()

    cfg: dict = {}
    if pre_args.config:
        with open(pre_args.config) as _f:
            cfg = yaml.safe_load(_f) or {}

    def _c(key: str, default: Any = None) -> Any:
        return cfg.get(key, default)

    # ------------------------------------------------------------------
    # Step 2: full argument parsing — CLI overrides YAML
    # ------------------------------------------------------------------
    _dev_default = _c("device", "auto")
    if _dev_default == "auto":
        _dev_default = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Latent-space rollout GIF for a pretrained ConcatWIREV5.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_default_config if _default_config.exists() else None,
        help="Path to YAML config file (default: latent_rollout_config.yaml next to script)",
    )
    parser.add_argument("--checkpoint", type=Path, default=_c("checkpoint"), help="Path to pretrained_latent_*.pth")
    parser.add_argument("--out", type=Path, default=_c("out"), help="Full output GIF path (overrides output_dir + auto-name)")
    parser.add_argument(
        "--cse-samples-out",
        type=Path,
        default=_c("cse_samples_out"),
        help="Optional CSV output path for per-frame CSE samples (default: next to --out)",
    )
    parser.add_argument(
        "--final-png-out",
        type=Path,
        default=_c("final_png_out"),
        help="Optional PNG output path for final aeroshell still (default: next to --out)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_c("output_dir", "/cluster/work/fuge/AISCO/wp9/pyvista_gif"),
        help="Output directory",
    )
    parser.add_argument("--n_frames", type=int, default=_c("n_frames", 240), help="Frames per full circle")
    parser.add_argument("--resolution", type=int, default=_c("resolution", 40), help="Marching-cubes grid resolution per axis")
    parser.add_argument("--radius", type=float, default=_c("radius"), help="Traversal radius (default: infer from checkpoint)")
    parser.add_argument("--fps", type=int, default=_c("fps", 24), help="GIF playback speed")
    parser.add_argument(
        "--final_hold_seconds",
        type=float,
        default=float(_c("final_hold_seconds", 5.0)),
        help="Extra hold duration at the end of the animation [s] by repeating the final frame",
    )
    parser.add_argument("--device", default=_dev_default, help="Compute device: cuda | cpu | auto")
    parser.add_argument("--width", type=int, default=_c("width", 4096), help="3D render width in px")
    parser.add_argument("--height", type=int, default=_c("height", 2160), help="3D render height in px")
    parser.add_argument("--panel_width", type=int, default=_c("panel_width", 320), help="Latent-map panel width in px")
    parser.add_argument("--azim", type=float, default=_c("azim", -60.0), help="Camera azimuth in degrees")
    parser.add_argument("--elev", type=float, default=_c("elev", 25.0), help="Camera elevation in degrees")
    parser.add_argument(
        "--mesh_zoom",
        type=float,
        default=_c("mesh_zoom", 1.0),
        help="Mesh zoom factor: >1 brings camera closer (larger mesh), <1 zooms out. Default 1.0.",
    )
    parser.add_argument(
        "--mesh_x_shift",
        type=float,
        default=_c("mesh_x_shift", 0.0),
        help="Shift the mesh left/right in world X [mm]. Positive = right, negative = left. Default 0.0.",
    )
    parser.add_argument("--mesh_color", default=_c("mesh_color", "#000000"), help="Aeroshell color (hex)")
    parser.add_argument("--render_bg", default=_c("render_bg", "#23283E"), help="3D render background color (hex)")
    parser.add_argument(
        "--transparent-background",
        dest="transparent_background",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("transparent_background", False)),
        help="Export final PNG with transparent background when using aeroshell-only output",
    )
    parser.add_argument(
        "--mesh_alpha", type=float, default=_c("mesh_alpha", 0.75), help="Aeroshell opacity 0.0-1.0 (1.0 = fully opaque)"
    )
    parser.add_argument(
        "--bbox_padding",
        type=float,
        default=_c("bbox_padding", 1.25),
        help="Expand each face of the extraction prism by this many mm",
    )
    parser.add_argument(
        "--outer_prism_scale",
        type=float,
        default=float(_c("outer_prism_scale", _DEFAULT_OUTER_PRISM_SCALE)),
        help="Scale factor for WP2 outer prism sampling bounds (default: 1.3)",
    )
    parser.add_argument(
        "--show_assembly",
        action="store_true",
        default=bool(_c("show_assembly", False)),
        help="Render inner assembly inside the aeroshell",
    )
    parser.add_argument("--assembly_color", default=_c("assembly_color", "#F32804"), help="Inner assembly color (hex)")
    parser.add_argument(
        "--show-solar-selection",
        dest="show_solar_selection",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_solar_selection", False)),
        help="Show/hide solar selection overlay on the aeroshell",
    )
    parser.add_argument(
        "--show-pressure-field",
        dest="show_pressure_field",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_pressure_field", False)),
        help="Show/hide GAOT pressure field on aeroshell (replaces aeroshell, windshield, and solar overlays)",
    )
    parser.add_argument(
        "--show-windshield",
        dest="show_windshield",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_windshield", False)),
        help="Show/hide windshield (vision cone) overlay on the aeroshell",
    )
    parser.add_argument(
        "--full-wheels",
        dest="full_wheels",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("full_wheels", True)),
        help="Replace cut-off wheel geometry with full wheels in inner assembly",
    )
    parser.add_argument(
        "--cutoff-wheel",
        dest="cutoff_wheel",
        action="store_true",
        default=bool(_c("cutoff_wheel", False)),
        help="Use cut-off wheel geometry (overrides --full-wheels)",
    )
    parser.add_argument(
        "--windshield-assembly-id",
        type=int,
        default=int(_c("windshield_assembly_id", 1)),
        help="WP2 assembly id used for windshield face-mask computation",
    )
    parser.add_argument(
        "--solar_selection_color", default=_c("solar_selection_color", "#F0D33E"), help="Solar selection color (hex)"
    )
    parser.add_argument("--windshield_color", default=_c("windshield_color", "#BAE9F1"), help="Windshield overlay color (hex)")
    parser.add_argument(
        "--windshield-alpha",
        type=float,
        default=float(_c("windshield_alpha", 0.65)),
        help="Windshield opacity in [0,1] (0 = cut out windshield area from aeroshell)",
    )
    parser.add_argument(
        "--solar_sun_dir",
        type=float,
        nargs=3,
        default=_c("solar_sun_dir", [-1.0, 0.0, 1.0]),
        metavar=("X", "Y", "Z"),
        help="Sun direction for solar selection overlay",
    )
    parser.add_argument(
        "--solar_area_budget_mm2",
        type=float,
        default=float(_c("solar_area_budget_mm2", 4_200_000.0)),
        help="Solar area budget in mm^2",
    )
    parser.add_argument(
        "--solar_binary_postprocess_iters",
        type=int,
        default=int(_c("solar_binary_postprocess_iters", 2)),
        help="Binary postprocess iterations for solar footprint smoothing (no gradients)",
    )
    parser.add_argument(
        "--solar_contour_postprocess_iters",
        type=int,
        default=int(_c("solar_contour_postprocess_iters", 4)),
        help="Boundary-only postprocess iterations to smooth solar contour while preserving interior",
    )
    parser.add_argument(
        "--solar_contour_hole_fill_iters",
        type=int,
        default=int(_c("solar_contour_hole_fill_iters", 2)),
        help="Postprocess iterations to fill small holes inside the solar selection contour",
    )
    parser.add_argument(
        "--solar_overlay_smooth_iters",
        type=int,
        default=int(_c("solar_overlay_smooth_iters", 10)),
        help="Post-op Taubin smoothing iterations for rendered solar overlay edges",
    )
    parser.add_argument(
        "--show-cse-overlay",
        dest="show_cse_overlay",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_cse_overlay", False)),
        help="Show GAOT drag, solar estimate, and CSE speed in a top-left overlay",
    )
    parser.add_argument(
        "--save-cse-samples",
        dest="save_cse_samples",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("save_cse_samples", True)),
        help="Save drag/solar/CSE values for each explored latent point to CSV",
    )
    parser.add_argument(
        "--show-cse-kmh",
        dest="show_cse_kmh",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_cse_kmh", False)),
        help="Also show CSE converted to km/h in the overlay",
    )
    parser.add_argument(
        "--show-cse-tacho",
        dest="show_cse_tacho",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_cse_tacho", False)),
        help="Show an optional CSE speed tachometer in the overlay",
    )
    parser.add_argument(
        "--cse-tacho-max-kmh",
        dest="cse_tacho_max_kmh",
        type=float,
        default=float(_c("cse_tacho_max_kmh", 160.0)),
        help="Fixed maximum speed (km/h) on the tachometer scale",
    )
    parser.add_argument(
        "--show-latent-panel",
        dest="show_latent_panel",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_latent_panel", True)),
        help="Show the latent spiral card in the right panel (including epoch_gif mode)",
    )
    parser.add_argument(
        "--export-final-png",
        dest="export_final_png",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("export_final_png", True)),
        help="Export the final rendered aeroshell frame as a standalone PNG",
    )
    parser.add_argument(
        "--final-png-include-composite",
        dest="final_png_include_composite",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("final_png_include_composite", False)),
        help="When exporting final PNG, include full composite (panel + overlays) instead of aeroshell-only render",
    )
    parser.add_argument(
        "--final-frame-use-best-cse",
        dest="final_frame_use_best_cse",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("final_frame_use_best_cse", True)),
        help="Use the highest-CSE frame (instead of the last frame) for final hold and final PNG export",
    )
    parser.add_argument(
        "--gaot_version",
        type=int,
        choices=[1, 2, 3],
        default=int(_c("gaot_version", 3)),
        help="GAOT version used for drag estimation",
    )
    parser.add_argument(
        "--gaot_model_id",
        default=_c("gaot_model_id"),
        help="GAOT model ID (or legacy key for v1) used by the selected GAOT version",
    )
    parser.add_argument(
        "--cse_metrics_device",
        default=str(_c("cse_metrics_device", "cpu")),
        help="Device for CSE helper metrics (recommended: cpu, to avoid poisoning render CUDA context)",
    )
    parser.add_argument(
        "--gaot_v_inf",
        type=float,
        default=float(_c("gaot_v_inf", 25.0)),
        help="Freestream velocity for drag integration [m/s]",
    )
    parser.add_argument(
        "--gaot_rho",
        type=float,
        default=float(_c("gaot_rho", 1.225)),
        help="Air density for drag integration [kg/m^3]",
    )
    parser.add_argument(
        "--gaot_mesh_resolution",
        type=int,
        default=int(_c("gaot_mesh_resolution", 50000)),
        help="GAOT internal mesh target resolution (loader compatibility parameter)",
    )
    parser.add_argument(
        "--gaot_gradient_max",
        type=float,
        default=float(_c("gaot_gradient_max", 0.001)),
        help="GAOT gradient clipping parameter (loader compatibility parameter)",
    )
    parser.add_argument(
        "--solar_effective_irradiance_w_m2",
        type=float,
        default=float(_c("solar_effective_irradiance_w_m2", 250.0)),
        help="Effective irradiance used for SR solar power estimate [W/m^2]",
    )
    parser.add_argument(
        "--cse_mass_kg",
        type=float,
        default=float(_c("cse_mass_kg", 210.0)),
        help="Vehicle mass for CSE cruising speed estimate [kg]",
    )
    parser.add_argument(
        "--cse_constant_losses_w",
        type=float,
        default=float(_c("cse_constant_losses_w", 20.0)),
        help="Constant electrical losses for CSE [W]",
    )
    parser.add_argument(
        "--cse_rolling_friction_coefficient",
        type=float,
        default=float(_c("cse_rolling_friction_coefficient", 0.006)),
        help="Rolling friction coefficient for CSE",
    )
    parser.add_argument(
        "--cse_text_color",
        default=_c("cse_text_color", "#EAF2F8"),
        help="CSE overlay text color (hex)",
    )
    parser.add_argument(
        "--cse_box_color",
        default=_c("cse_box_color", "#17202A"),
        help="CSE overlay background box color (hex)",
    )
    parser.add_argument(
        "--cse_box_alpha",
        type=float,
        default=float(_c("cse_box_alpha", 0.72)),
        help="CSE overlay background opacity in [0,1]",
    )
    parser.add_argument(
        "--latent_box_alpha",
        type=float,
        default=float(_c("latent_box_alpha", 0.72)),
        help="Latent overlay background opacity in [0,1]",
    )
    parser.add_argument(
        "--assembly_scale",
        type=float,
        default=float(_c("assembly_scale", 1.0)),
        help="Fine-tune scale of the inner assembly (1.0 = default)",
    )
    parser.add_argument(
        "--mask_refine_subdivisions",
        type=int,
        default=int(_c("mask_refine_subdivisions", 2)),
        help="Subdivision steps for windshield/solar mask geometry (higher = cleaner edges, no post-smoothing)",
    )
    parser.add_argument(
        "--solar_mask_refine_subdivisions",
        type=int,
        default=int(_c("solar_mask_refine_subdivisions", 6)),
        help="Subdivision steps specifically for solar-selection mask geometry (higher = smoother solar corners)",
    )
    parser.add_argument(
        "--show-anchors",
        dest="show_anchors",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("show_anchors", True)),
        help="Show/hide pretrained anchor dots in the latent-space panel",
    )
    parser.add_argument(
        "--epoch_gif",
        "--epoch-gif",
        dest="epoch_gif",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("epoch_gif", False)),
        help="Render one frame per checkpoint epoch (chronological) at fixed latent coordinates",
    )
    parser.add_argument(
        "--epoch_coords",
        type=float,
        nargs=2,
        default=_c("epoch_coords", [0.0, 0.0]),
        metavar=("Z0", "Z1"),
        help="Fixed latent coordinates [z0 z1] for epoch_gif mode",
    )
    # Traversal mode
    parser.add_argument(
        "--mode",
        default=_c("mode", "circle"),
        choices=["circle", "spiral", "line", "cross", "epoch_gif"],
        help="Latent-space traversal mode: circle | spiral | line | cross | epoch_gif",
    )
    # Spiral-mode params
    parser.add_argument("--spiral_turns", type=float, default=_c("spiral_turns", 3.0), help="Full rotations in spiral mode")
    parser.add_argument(
        "--spiral-equal-distance",
        dest="spiral_equal_distance",
        action=argparse.BooleanOptionalAction,
        default=bool(_c("spiral_equal_distance", False)),
        help="Use equal arc-length spacing for spiral samples (prevents visual speed-up)",
    )
    parser.add_argument(
        "--spiral_radius_start",
        type=float,
        default=_c("spiral_radius_start", 0.0),
        help="Inner radius for spiral (0 = start at origin)",
    )
    parser.add_argument(
        "--spiral_radius_end",
        type=float,
        default=_c("spiral_radius_end"),
        help="Outer radius for spiral (default: use z_radius from checkpoint)",
    )
    # Line-mode params
    parser.add_argument(
        "--line_start",
        type=float,
        nargs=2,
        default=_c("line_start", [0.0, -2.0]),
        metavar=("Z0", "Z1"),
        help="Start latent coordinates [z0 z1] for line mode",
    )
    parser.add_argument(
        "--line_end",
        type=float,
        nargs=2,
        default=_c("line_end", [0.0, 2.0]),
        metavar=("Z0", "Z1"),
        help="End latent coordinates [z0 z1] for line mode",
    )
    parser.add_argument(
        "--line_bounces",
        type=int,
        default=_c("line_bounces", 1),
        help="Complete back-and-forth oscillations for line mode (1 = A→B→A)",
    )
    # Cross-mode params
    parser.add_argument(
        "--cross_radius",
        type=float,
        default=_c("cross_radius"),
        help="Half-width of the X pattern in cross mode (default: use z_radius from checkpoint)",
    )
    args = parser.parse_args()
    if args.cutoff_wheel:
        args.full_wheels = False

    if args.checkpoint is None:
        sys.exit("ERROR: --checkpoint is required (or set 'checkpoint' in the YAML config)")
    args.checkpoint = Path(args.checkpoint)
    if not args.checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")

    # Resolve output path
    if args.out is None:
        stem = args.checkpoint.stem  # e.g. "39_26423_feasible-monkey_epoch1400"
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = Path(args.output_dir) / f"latent_rollout_{stem}_{ts}.gif"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.cse_samples_out is None:
        args.cse_samples_out = args.out.with_name(f"{args.out.stem}_cse_samples.csv")
    if args.final_png_out is None:
        args.final_png_out = args.out.with_name(f"{args.out.stem}_final_aeroshell.png")
    Path(args.cse_samples_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.final_png_out).parent.mkdir(parents=True, exist_ok=True)

    # Parse render colors
    mesh_color_rgba = _parse_color_rgba_float(args.mesh_color)
    render_bg_rgba = _parse_color_rgba_float(args.render_bg)
    assembly_color_rgba = _parse_color_rgba_float(args.assembly_color, 1.0)
    solar_selection_color_rgba = _parse_color_rgba_float(args.solar_selection_color, 1.0)
    windshield_color_rgba = _parse_color_rgba_float(args.windshield_color, float(args.windshield_alpha))

    # Build panel colors dict (sourced from YAML cfg, with hardcoded fallbacks)
    panel_colors = {
        "bg": _parse_color_rgb(_c("panel_bg", "#14181E")),
        "bg_epoch": _parse_color_rgb(_c("panel_bg", "#14181E")),
        "grid": _parse_color_rgb(_c("panel_grid", "#36435F")),
        "axis": _parse_color_rgb(_c("panel_axis", "#4B5573")),
        "circle": _parse_color_rgb(_c("panel_circle", "#6E7DA5")),
        "anchor": _parse_color_rgb(_c("panel_anchor", "#DC5A5A")),
        "anchor_label": _parse_color_rgb(_c("panel_anchor_label", "#F0A0A0")),
        "progress": _parse_color_rgb(_c("panel_progress", "#50C8FF")),
        "cursor": _parse_color_rgb(_c("panel_cursor", "#50C8FF")),
        "cursor_outline": _parse_color_rgb(_c("panel_cursor_outline", "#FFFFFF")),
        "axis_label": _parse_color_rgb(_c("panel_axis_label", "#B4BEDC")),
        "coord_text": _parse_color_rgb(_c("panel_coord_text", "#82C3FF")),
        "frame_text": _parse_color_rgb(_c("panel_frame_text", "#8C91AA")),
        "title": _parse_color_rgb(_c("panel_title", "#C8CDE6")),
    }
    cse_text_rgb = _parse_color_rgb(args.cse_text_color)
    cse_box_rgb = _parse_color_rgb(args.cse_box_color)

    device = torch.device(args.device)
    print(f"Device: {device}")

    metrics_device_str = str(args.cse_metrics_device).strip().lower()
    if metrics_device_str == "render":
        metrics_device = device
    elif metrics_device_str == "auto":
        metrics_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        metrics_device = torch.device(metrics_device_str)

    gaot_loader: Any | None = None
    if args.show_cse_overlay or args.show_pressure_field:
        default_cfd_cfg = _load_wp4_default_cfd_config()
        gaot_model_id = args.gaot_model_id if args.gaot_model_id is not None else default_cfd_cfg.get("model_id")
        if gaot_model_id is None:
            warnings.warn(
                "[latent_rollout] GAOT model_id missing; GAOT-based overlays are unavailable.",
                UserWarning,
                stacklevel=2,
            )
        else:
            gaot_cfd_cfg = {
                "version": int(args.gaot_version),
                "model_id": gaot_model_id,
                "v_inf": float(args.gaot_v_inf),
                "rho": float(args.gaot_rho),
                "mesh_resolution": int(args.gaot_mesh_resolution),
                "gradient_max": float(args.gaot_gradient_max),
                "mesh_resolution_tolerance": float(default_cfd_cfg.get("mesh_resolution_tolerance", 0.1)),
                "visualisations": False,
                "pressure_backprop_mode": "full",
                "pressure_integration_mode": "face",
                "cp_smoothing": {"enabled": False},
            }
            gaot_loader = _init_gaot_drag_estimator(
                gaot_version=int(args.gaot_version),
                cfd_config=gaot_cfd_cfg,
                device=metrics_device,
            )

    if args.show_pressure_field and int(args.gaot_version) != 3:
        warnings.warn(
            "[latent_rollout] pressure-field view currently requires gaot_version=3. Falling back to standard mesh rendering.",
            UserWarning,
            stacklevel=2,
        )

    active_mode = "epoch_gif" if args.epoch_gif else args.mode
    base_epoch_num = _parse_epoch_from_stem(args.checkpoint.stem)

    # --- Load model ---
    model, z_dim, z_radius_ckpt, Z_pretrained = load_checkpoint(args.checkpoint, device)
    z_radius = args.radius if args.radius is not None else z_radius_ckpt

    epoch_checkpoints: list[tuple[int, Path]] = []
    if active_mode == "epoch_gif":
        epoch_checkpoints = _find_epoch_checkpoints(args.checkpoint)
        if args.n_frames > 0:
            epoch_checkpoints = epoch_checkpoints[: min(int(args.n_frames), len(epoch_checkpoints))]
        print(
            f"Traversal radius: {z_radius:.3f}  |  mode: {active_mode}  |  "
            f"frames: {len(epoch_checkpoints)}  |  resolution: {args.resolution}"
        )
    else:
        print(
            f"Traversal radius: {z_radius:.3f}  |  mode: {active_mode}  |  "
            f"frames: {args.n_frames}  |  resolution: {args.resolution}"
        )

    Z_np = Z_pretrained.cpu().numpy() if Z_pretrained is not None else None

    # --- Load inner assembly (if requested) ---
    inner_assembly: trimesh.Trimesh | None = None
    if args.show_assembly:
        epoch_num = _parse_epoch_from_stem(args.checkpoint.stem)
        if epoch_num is not None:
            inner_assembly = load_inner_assembly(args.checkpoint, epoch_num)
            if inner_assembly is not None and args.full_wheels:
                inner_assembly = _apply_full_wheels(inner_assembly)
            # Apply assembly scale fine-tune
            if inner_assembly is not None and float(args.assembly_scale) != 1.0:
                inner_assembly.vertices *= float(args.assembly_scale)
        else:
            print("[assembly] WARN: could not parse epoch from checkpoint stem, skipping")

    # --- Build extractor once ---
    extractor = build_extractor_with_wp2_prism(
        device,
        args.resolution,
        bbox_padding=args.bbox_padding,
        prism_scale=float(args.outer_prism_scale),
    )

    # --- Build z-vector sequence for selected traversal mode ---
    line_start_pt = (float(args.line_start[0]), float(args.line_start[1]))
    line_end_pt = (float(args.line_end[0]), float(args.line_end[1]))
    frame_checkpoints: list[Path | None]
    if active_mode == "epoch_gif":
        fixed_z0 = float(args.epoch_coords[0])
        fixed_z1 = float(args.epoch_coords[1])
        z_coords = [(fixed_z0, fixed_z1) for _ in epoch_checkpoints]
        panel_scale_radius = max(z_radius, abs(fixed_z0), abs(fixed_z1), 1e-6)
        frame_checkpoints = [ckpt for _, ckpt in epoch_checkpoints]
    else:
        z_coords, panel_scale_radius = _build_trajectory(
            mode=active_mode,
            n_frames=args.n_frames,
            z_radius=z_radius,
            spiral_turns=args.spiral_turns,
            spiral_equal_distance=bool(args.spiral_equal_distance),
            spiral_radius_start=args.spiral_radius_start,
            spiral_radius_end=args.spiral_radius_end,
            line_start=line_start_pt,
            line_end=line_end_pt,
            line_bounces=args.line_bounces,
            cross_radius=args.cross_radius,
        )
        frame_checkpoints = [None] * len(z_coords)

    total_frames = len(z_coords)
    panel_trajectory = None if active_mode == "epoch_gif" else z_coords

    global_bounds = None
    if inner_assembly is not None and len(inner_assembly.vertices) > 0:
        b = inner_assembly.bounds
        global_bounds = [b[0][0], b[1][0], b[0][1], b[1][1], b[0][2], b[1][2]]

        # INCREASE MARGIN PAD TO 25% TO COMPENSATE FOR CORNER CLIPPING AT ULTRA-HD WIDESCREEN RENDER DISTANCES
        padding = max(global_bounds[1] - global_bounds[0], global_bounds[3] - global_bounds[2]) * 0.25
        global_bounds[0] -= padding
        global_bounds[1] += padding
        global_bounds[2] -= padding
        global_bounds[3] += padding
        global_bounds[4] -= padding
        global_bounds[5] += padding

    frames: list[Any] = []
    cse_samples: list[dict[str, Any]] = []
    last_valid_rgb: np.ndarray | None = None
    last_aeroshell_rgb: np.ndarray | None = None
    last_composite_frame: np.ndarray | None = None
    best_cse_ms: float | None = None
    best_cse_frame_idx: int | None = None
    best_cse_aeroshell_rgb: np.ndarray | None = None
    best_cse_composite_frame: np.ndarray | None = None
    render_width = args.width
    assembly_smooth_shading = not args.full_wheels
    solar_selection_texture = _build_solar_cell_texture() if args.show_solar_selection else None

    # Lock camera once from WP2 prism bounds — identical scale across all modes.
    _prism_center, _prism_diag = _get_prism_center_diag(float(args.outer_prism_scale), args.bbox_padding)
    locked_camera_position, locked_camera_focal_point, locked_camera_view_up = _compute_camera_from_center(
        _prism_center,
        _prism_diag,
        args.azim,
        args.elev,
        distance_mult=1.6 / max(0.1, float(args.mesh_zoom)),
    )
    if global_bounds is not None:
        bounds_center = np.array(
            [
                0.5 * (global_bounds[0] + global_bounds[1]),
                0.5 * (global_bounds[2] + global_bounds[3]),
                0.5 * (global_bounds[4] + global_bounds[5]),
            ],
            dtype=np.float64,
        )
        cam_offset = locked_camera_position - locked_camera_focal_point
        locked_camera_focal_point = bounds_center
        locked_camera_position = bounds_center + cam_offset

    # Apply mesh_x_shift: pan camera focal point + position along world X so the
    # mesh appears shifted left (negative) or right (positive) in the frame.
    if float(args.mesh_x_shift) != 0.0:
        x_shift = float(args.mesh_x_shift)
        locked_camera_focal_point = locked_camera_focal_point + np.array([x_shift, 0.0, 0.0], dtype=np.float64)
        locked_camera_position = locked_camera_position + np.array([x_shift, 0.0, 0.0], dtype=np.float64)
        print(f"[camera] mesh_x_shift={x_shift:+.1f} mm applied to camera focal point")

    # Pre-compute the vision cone mesh once — loading it per-frame dominates runtime.
    _cached_cone_mesh: tuple[Tensor, Tensor] | None = None
    if args.show_windshield or args.show_solar_selection:
        _cached_cone_mesh = _precompute_vision_cone_mesh(int(args.windshield_assembly_id), device)

    print("\nGenerating frames...")
    outer_margin_x = max(4, int(0.02 * args.width))
    frame_scale = args.height / 600.0
    box_w, box_h = _box_dimensions(frame_scale, bool(args.show_cse_tacho))
    for i, (z0, z1) in enumerate(z_coords):
        frame_ckpt = frame_checkpoints[i]
        frame_epoch_num: int | None = base_epoch_num
        if frame_ckpt is not None:
            _load_model_state_from_checkpoint(model, frame_ckpt, device)
            frame_epoch_num = _parse_epoch_from_stem(frame_ckpt.stem)
            ep_str = f" epoch={frame_epoch_num}" if frame_epoch_num is not None else ""
        else:
            ep_str = ""

        print(f"  [{i + 1:3d}/{total_frames}]" + ep_str + f" z=[{z0:+.3f}, {z1:+.3f}]", end="", flush=True)

        mesh = extract_mesh(model, extractor, z0, z1, z_dim, device)
        if mesh is None or len(mesh.vertices) == 0:
            print("  → SKIP (no mesh)")
            if bool(args.save_cse_samples):
                cse_samples.append(
                    {
                        "frame_index": i,
                        "frame_number": i + 1,
                        "epoch": frame_epoch_num,
                        "checkpoint": frame_ckpt.name if frame_ckpt is not None else "",
                        "z0": float(z0),
                        "z1": float(z1),
                        "mesh_valid": 0,
                        "drag_n": np.nan,
                        "solar_w": np.nan,
                        "cse_ms": np.nan,
                        "cse_kmh": np.nan,
                    }
                )
            if last_valid_rgb is not None:
                composite_skip = np.array(last_valid_rgb, copy=True)
                if bool(args.show_latent_panel):
                    right_img, right_draw = box_builder(
                        box_w,
                        box_h,
                        frame_scale,
                        box_color=cse_box_rgb,
                        box_alpha=float(args.latent_box_alpha),
                    )
                    draw_right_box(
                        right_draw,
                        box_w,
                        box_h,
                        frame_scale,
                        z0,
                        z1,
                        z_radius,
                        Z_np,
                        i,
                        total_frames,
                        active_mode,
                        panel_trajectory,
                        panel_scale_radius,
                        bool(args.show_anchors),
                        panel_colors,
                    )
                    _, skip_box_y, right_x, _ = box_position(
                        composite_skip.shape[1],
                        composite_skip.shape[0],
                        box_w,
                        box_h,
                        outer_margin_x,
                    )
                    composite_skip = _composite_box_on_frame(composite_skip, right_img, right_x, skip_box_y)
                composite_skip = draw_epoch_label_on_frame(composite_skip, frame_epoch_num, box_color=(20, 24, 30))
                frames.append(composite_skip)
            continue

        cone_face_mask: Tensor | None = None
        solar_selection_mesh: trimesh.Trimesh | None = None
        windshield_mesh: trimesh.Trimesh | None = None
        drag_n: float | None = None
        solar_w: float | None = None
        cse_ms: float | None = None
        mesh_for_render = mesh
        mask_mesh = mesh
        pressure_field_cp_t: Tensor | None = None
        pressure_field_cp_np: np.ndarray | None = None
        show_pressure_field_now = bool(args.show_pressure_field) and gaot_loader is not None and int(args.gaot_version) == 3
        if show_pressure_field_now:
            pressure_field_cp_t = _estimate_pressure_field_cp(
                gaot_loader=gaot_loader,
                gaot_version=int(args.gaot_version),
                mesh=mesh,
            )
            if pressure_field_cp_t is not None:
                pressure_field_cp_np = pressure_field_cp_t.detach().cpu().numpy().squeeze(-1)

        if (not show_pressure_field_now) and (args.show_windshield or args.show_solar_selection):
            refine_subdiv = args.mask_refine_subdivisions
            if args.show_solar_selection:
                refine_subdiv = max(refine_subdiv, args.solar_mask_refine_subdivisions)
            mask_mesh = _subdivide_mesh_for_masks(mesh, refine_subdiv)
            cone_face_mask = _compute_windshield_face_mask(
                mask_mesh, args.windshield_assembly_id, device, precomputed_cone=_cached_cone_mesh
            )
        if (not show_pressure_field_now) and args.show_windshield and cone_face_mask is not None:
            cone_np = cone_face_mask.detach().cpu().numpy()
            cutout_mesh = _cutout_faces_from_mesh(mask_mesh, cone_np)
            if cutout_mesh is not None:
                mesh_for_render = cutout_mesh
            else:
                warnings.warn(
                    "[latent_rollout] windshield cutout produced empty mesh; using original aeroshell.",
                    UserWarning,
                    stacklevel=2,
                )
            if float(args.windshield_alpha) > 0.0:
                windshield_mesh = _build_face_overlay_mesh(
                    mask_mesh,
                    cone_np,
                    normal_offset_mm=1.5,
                )
        if (not show_pressure_field_now) and args.show_solar_selection:
            solar_mask = _compute_solar_selection_mask(
                mask_mesh,
                device=device,
                sun_dir_xyz=(float(args.solar_sun_dir[0]), float(args.solar_sun_dir[1]), float(args.solar_sun_dir[2])),
                area_budget_mm2=float(args.solar_area_budget_mm2),
                cone_face_mask=cone_face_mask,
                binary_postprocess_iters=int(args.solar_binary_postprocess_iters),
                contour_postprocess_iters=int(args.solar_contour_postprocess_iters),
                contour_hole_fill_iters=int(args.solar_contour_hole_fill_iters),
            )
            if solar_mask is not None:
                solar_selection_mesh_raw = _build_face_overlay_mesh(
                    mask_mesh,
                    solar_mask.detach().cpu().numpy(),
                    normal_offset_mm=1.0,
                )
                solar_selection_mesh = _post_smooth_overlay_mesh(
                    solar_selection_mesh_raw,
                    int(args.solar_overlay_smooth_iters),
                )

        need_best_cse_frame = bool(args.final_frame_use_best_cse) and (
            float(args.final_hold_seconds) > 0.0 or bool(args.export_final_png)
        )
        need_cse_metrics = bool(args.show_cse_overlay) or bool(args.save_cse_samples) or need_best_cse_frame
        if need_cse_metrics:
            if gaot_loader is not None:
                drag_n = _estimate_drag_newtons(
                    gaot_loader=gaot_loader,
                    gaot_version=int(args.gaot_version),
                    sdf_model=model,
                    mesh=mesh,
                    frame_step=i,
                    gaot_pressure_field=pressure_field_cp_t,
                )

            solar_w, _solar_diag = _estimate_solar_power_watts(
                mesh=mask_mesh,
                device=metrics_device,
                sun_dir_xyz=(float(args.solar_sun_dir[0]), float(args.solar_sun_dir[1]), float(args.solar_sun_dir[2])),
                area_budget_mm2=float(args.solar_area_budget_mm2),
                effective_irradiance_w_m2=float(args.solar_effective_irradiance_w_m2),
                cone_face_mask=cone_face_mask,
            )

            if drag_n is not None and solar_w is not None:
                cse_ms = _compute_cse_speed_ms(
                    drag_force_newton=drag_n,
                    solar_power_watts=solar_w,
                    mass_kg=float(args.cse_mass_kg),
                    constant_losses_w=float(args.cse_constant_losses_w),
                    rolling_friction_coefficient=float(args.cse_rolling_friction_coefficient),
                )

        if bool(args.save_cse_samples):
            cse_samples.append(
                {
                    "frame_index": i,
                    "frame_number": i + 1,
                    "epoch": frame_epoch_num,
                    "checkpoint": frame_ckpt.name if frame_ckpt is not None else "",
                    "z0": float(z0),
                    "z1": float(z1),
                    "mesh_valid": 1,
                    "drag_n": float(drag_n) if drag_n is not None else np.nan,
                    "solar_w": float(solar_w) if solar_w is not None else np.nan,
                    "cse_ms": float(cse_ms) if cse_ms is not None else np.nan,
                    "cse_kmh": (float(cse_ms) * 3.6) if cse_ms is not None else np.nan,
                }
            )

        rgb = render_mesh(
            mesh=mesh_for_render,
            bg_color=render_bg_rgba,
            mesh_color=mesh_color_rgba,
            mesh_alpha=args.mesh_alpha,
            assembly_mesh=inner_assembly,
            assembly_color=assembly_color_rgba,
            width=render_width,
            height=args.height,
            azim_deg=args.azim,
            elev_deg=args.elev,
            look_at_bounds=global_bounds,
            camera_position=locked_camera_position,
            camera_focal_point=locked_camera_focal_point,
            camera_view_up=locked_camera_view_up,
            solar_selection_mesh=solar_selection_mesh,
            solar_selection_color=solar_selection_color_rgba,
            solar_selection_texture=solar_selection_texture,
            windshield_mesh=windshield_mesh,
            windshield_color=windshield_color_rgba,
            assembly_smooth_shading=assembly_smooth_shading,
            pressure_field=pressure_field_cp_np,
            pressure_field_enabled=show_pressure_field_now,
        )
        composite = np.array(rgb, copy=True)
        lbox_x, box_y, rbox_x, _ = box_position(
            composite.shape[1],
            composite.shape[0],
            box_w,
            box_h,
            outer_margin_x,
        )
        if bool(args.show_latent_panel):
            right_img, right_draw = box_builder(
                box_w,
                box_h,
                frame_scale,
                box_color=cse_box_rgb,
                box_alpha=float(args.latent_box_alpha),
            )
            draw_right_box(
                right_draw,
                box_w,
                box_h,
                frame_scale,
                z0,
                z1,
                z_radius,
                Z_np,
                i,
                total_frames,
                active_mode,
                panel_trajectory,
                panel_scale_radius,
                bool(args.show_anchors),
                panel_colors,
            )
            composite = _composite_box_on_frame(composite, right_img, rbox_x, box_y)
        if args.show_cse_overlay:
            left_img, left_draw = box_builder(
                box_w,
                box_h,
                frame_scale,
                box_color=cse_box_rgb,
                box_alpha=float(args.latent_box_alpha),
            )
            draw_left_box(
                left_draw,
                box_w,
                box_h,
                frame_scale,
                cse_box_rgb,
                drag_n,
                solar_w,
                cse_ms,
                bool(args.show_cse_tacho),
                cse_text_rgb,
                show_kmh=bool(args.show_cse_kmh),
                tacho_max_kmh=float(args.cse_tacho_max_kmh),
            )
            composite = _composite_box_on_frame(composite, left_img, lbox_x, box_y)

        composite = draw_epoch_label_on_frame(composite, frame_epoch_num, box_color=(20, 24, 30))
        frames.append(composite)
        last_valid_rgb = rgb
        last_aeroshell_rgb = rgb
        last_composite_frame = composite
        if cse_ms is not None and np.isfinite(cse_ms):
            cse_val = float(cse_ms)
            if best_cse_ms is None or cse_val > best_cse_ms:
                best_cse_ms = cse_val
                best_cse_frame_idx = i
                best_cse_aeroshell_rgb = rgb
                best_cse_composite_frame = composite
        print(f"  → {len(mesh.vertices):,} verts")

    if not frames:
        sys.exit("ERROR: no valid frames were rendered — check model and checkpoint")

    if bool(args.save_cse_samples):
        cse_out = Path(args.cse_samples_out)
        cse_fieldnames = [
            "frame_index",
            "frame_number",
            "epoch",
            "checkpoint",
            "z0",
            "z1",
            "mesh_valid",
            "drag_n",
            "solar_w",
            "cse_ms",
            "cse_kmh",
        ]
        with open(cse_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=cse_fieldnames)
            writer.writeheader()
            writer.writerows(cse_samples)
        print(f"Saved CSE samples: {cse_out}")

    hold_source_label = "last"
    hold_source_frame = frames[-1]
    png_source_aeroshell = last_aeroshell_rgb
    png_source_composite = last_composite_frame
    if (
        bool(args.final_frame_use_best_cse)
        and best_cse_aeroshell_rgb is not None
        and best_cse_composite_frame is not None
        and best_cse_frame_idx is not None
        and best_cse_ms is not None
    ):
        hold_source_label = f"best CSE (frame {int(best_cse_frame_idx) + 1}, {float(best_cse_ms):.2f} m/s)"
        hold_source_frame = best_cse_composite_frame
        png_source_aeroshell = best_cse_aeroshell_rgb
        png_source_composite = best_cse_composite_frame
    elif bool(args.final_frame_use_best_cse):
        warnings.warn(
            "[latent_rollout] highest-CSE frame requested but unavailable; falling back to last frame.",
            UserWarning,
            stacklevel=2,
        )

    hold_frames = max(0, round(float(args.final_hold_seconds) * max(int(args.fps), 1)))
    if hold_frames > 0:
        frames.extend([hold_source_frame] * hold_frames)
        print(f"Appended final hold: {hold_frames} frame(s) ({float(args.final_hold_seconds):.2f}s) from {hold_source_label}")

    if bool(args.export_final_png):
        png_out = Path(args.final_png_out)
        if bool(args.final_png_include_composite):
            if png_source_composite is not None:
                if bool(args.transparent_background):
                    bg_rgb = np.array(_parse_color_rgb(args.render_bg), dtype=np.uint8)
                    alpha_mask = np.where(np.all(png_source_composite == bg_rgb[None, None, :], axis=-1), 0, 255).astype(np.uint8)
                    png_rgba = np.dstack([png_source_composite, alpha_mask])
                    imageio.imwrite(png_out, png_rgba)
                    print(f"Saved transparent composite PNG ({hold_source_label}): {png_out}")
                else:
                    imageio.imwrite(png_out, png_source_composite)
                    print(f"Saved final composite PNG ({hold_source_label}): {png_out}")
            else:
                warnings.warn(
                    "[latent_rollout] final composite frame missing; skipping PNG export.",
                    UserWarning,
                    stacklevel=2,
                )
        elif png_source_aeroshell is not None:
            if bool(args.transparent_background):
                bg_rgb = np.array(_parse_color_rgb(args.render_bg), dtype=np.uint8)
                alpha = np.where(np.all(png_source_aeroshell == bg_rgb[None, None, :], axis=-1), 0, 255).astype(np.uint8)
                png_rgba = np.dstack([png_source_aeroshell, alpha])
                imageio.imwrite(png_out, png_rgba)
                print(f"Saved transparent final aeroshell PNG ({hold_source_label}): {png_out}")
            else:
                imageio.imwrite(png_out, png_source_aeroshell)
                print(f"Saved final aeroshell PNG ({hold_source_label}): {png_out}")
        else:
            warnings.warn(
                "[latent_rollout] final aeroshell frame missing; skipping PNG export.",
                UserWarning,
                stacklevel=2,
            )

    out_path = Path(args.out)
    print(f"\nWriting {len(frames)} frames \u2192 {out_path}  (fps={args.fps})")
    if out_path.suffix.lower() == ".mp4":
        # Uses standard system video codecs for pixel-perfect compression
        imageio.mimsave(out_path, frames, fps=args.fps, quality=9, macro_block_size=16)
        print(f"Done.  MP4 saved to: {out_path}")
    else:
        # Forces a dedicated local palette optimization per frame instead of a global blur
        out_path = Path(args.out)
        print(f"\nWriting {len(frames)} frames → {out_path}  (fps={args.fps})")

        # Force imageio to calculate an optimized palette per frame rather than a single blurry global palette
        imageio.mimsave(
            out_path,
            frames,
            fps=args.fps,
            loop=0,
            quantizer="nq",  # NeuQuant high-quality neural network quantizer
            palettesize=256,
        )
        print(f"Done.  GIF saved to: {out_path}")


if __name__ == "__main__":
    main()
