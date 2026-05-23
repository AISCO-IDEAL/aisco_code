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

import imageio
import numpy as np
import pyrender
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
        sys.exit(
            f"ERROR: no epoch checkpoints found next to {ckpt_path}. "
            f"Expected files matching pattern '{pattern}'."
        )

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

_ADD_WHEELS_PATH = (
    Path(__file__).parents[1] / "wp2_assembler" / "version4" / "3D_printing_helpers" / "add_wheels.py"
)


def _apply_full_wheels(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Replace cut-off wheel geometry with full wheel cylinders.

    Returns the original mesh unchanged on any failure.
    """
    try:
        spec = importlib.util.spec_from_file_location("_aw", _ADD_WHEELS_PATH)
        if spec is None or spec.loader is None:
            warnings.warn("[latent_rollout] add_wheels.py not found — skipping full-wheel replacement.", UserWarning, stacklevel=2)
            return mesh
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        centers = mod.infer_wheel_centers(mesh)
        if not centers:
            warnings.warn("[latent_rollout] No wheel-like components found — skipping full-wheel replacement.", UserWarning, stacklevel=2)
            return mesh
        base = mod.remove_cut_wheel_components(mesh, centers)
        wheels = mod.make_full_wheels(centers)
        if wheels is None:
            return base
        result = trimesh.util.concatenate([base, wheels])
        print(f"[latent_rollout] full_wheels: {len(centers)} wheel(s) replaced")
    except Exception as exc:
        warnings.warn(f"[latent_rollout] full_wheels failed: {exc}. Using original mesh.", UserWarning, stacklevel=2)
        return mesh
    else:
        return result


def _compute_windshield_face_mask(
    aeroshell_mesh: trimesh.Trimesh,
    assembly_id: int,
    device: torch.device,
) -> Tensor | None:
    """Return a bool tensor (F,) marking aeroshell faces inside the vision cone."""
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


def _load_windshield_from_wp2(assembly_id: int, tol: float = 2.0, atol: float = 0.4) -> trimesh.Trimesh | None:
    """Tessellate the WP2 vision NBZ mesh for direct windshield rendering."""
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
            f"[latent_rollout] WP2 windshield load failed: {exc}. Skipping.",
            UserWarning,
            stacklevel=2,
        )
        return None


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
            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            smoothed_scores = _smooth_binary_face_scores(
                mesh=aeroshell_mesh,
                base_scores=binary_np,
                valid_mask=valid_np,
                iterations=binary_postprocess_iters,
            )
            binary_mask = torch.tensor(smoothed_scores > 0.5, dtype=torch.bool, device=device)
        binary_mask = binary_mask & valid_mask

        if contour_postprocess_iters > 0:
            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            base_np = binary_mask.detach().cpu().numpy().astype(np.float32)
            contour_scores = _smooth_binary_contour_scores(
                mesh=aeroshell_mesh,
                base_scores=base_np,
                valid_mask=valid_np,
                iterations=contour_postprocess_iters,
            )

            valid_idx = np.flatnonzero(valid_np)
            if valid_idx.size > 0:
                illum_np = illumination.detach().cpu().numpy()
                area_np = face_area_m2.detach().cpu().numpy()
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
            valid_np = valid_mask.detach().cpu().numpy().astype(bool)
            filled_np = _fill_binary_contour_holes(
                mesh=aeroshell_mesh,
                binary_mask=binary_mask.detach().cpu().numpy().astype(bool),
                valid_mask=valid_np,
                iterations=contour_hole_fill_iters,
            )
            binary_mask = torch.tensor(filled_np, dtype=torch.bool, device=device)

        # Post-op reallocation: if windshield exclusion removed selected area,
        # refill from best remaining valid faces to reach the max allowed area.
        target_area_m2 = torch.minimum(
            torch.tensor(area_budget_mm2 * 1e-6, device=device),
            face_area_m2[valid_mask].sum(),
        )
        selected_area_m2 = face_area_m2[binary_mask].sum()
        if float(selected_area_m2.item()) > float(target_area_m2.item()):
            selected_idx = torch.where(binary_mask & valid_mask)[0]
            if selected_idx.numel() > 0:
                order = selected_idx[torch.argsort(illumination[selected_idx], descending=False)]
                running = selected_area_m2
                for idx in order:
                    if float(running.item()) <= float(target_area_m2.item()):
                        break
                    binary_mask[idx] = False
                    running = running - face_area_m2[idx]
                selected_area_m2 = running
        if float(selected_area_m2.item()) < float(target_area_m2.item()):
            candidates = torch.where(valid_mask & (~binary_mask))[0]
            if candidates.numel() > 0:
                order = candidates[torch.argsort(illumination[candidates], descending=True)]
                running = selected_area_m2
                for idx in order:
                    if float(running.item()) >= float(target_area_m2.item()):
                        break
                    binary_mask[idx] = True
                    running = running + face_area_m2[idx]
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
        if sub is None or len(sub.vertices) == 0:
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
        if sub is None or len(sub.vertices) == 0:
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
) -> np.ndarray:
    """Diffuse binary face scores across face adjacency while keeping a hard final mask possible."""
    if iterations <= 0:
        return base_scores

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
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
        scores = accum / np.clip(counts, 1.0, None)
        scores[~valid_mask] = -1.0

    return scores


def _smooth_binary_contour_scores(
    mesh: trimesh.Trimesh,
    base_scores: np.ndarray,
    valid_mask: np.ndarray,
    iterations: int,
) -> np.ndarray:
    """Smooth only boundary faces so contour gets cleaner while interior stays stable."""
    if iterations <= 0:
        return base_scores

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
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
) -> np.ndarray:
    """Fill small holes inside selected regions without blurring the outer contour."""
    if iterations <= 0:
        return binary_mask

    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
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
            (~filled)
            & valid_mask
            & (nbr_counts >= float(min_neighbor_count))
            & (pos_ratio >= float(min_positive_ratio))
        )
        if not np.any(fill_candidates):
            break
        filled[fill_candidates] = True

    return filled


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
        print(
            f"[extractor] WP2 outer prism scale={prism_scale:.3f} "
            f"x={prism_x} y={prism_y} z={prism_z}"
        )
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
) -> np.ndarray:
    """Render mesh and inner assembly using PyVista with Order-Independent Transparency."""
    # 1. Initialize offscreen plotter (removing the multi_samples keyword argument)
    plotter = pv.Plotter(off_screen=True, window_size=[width, height])
    plotter.background_color = bg_color[:3]

    # Configure Anti-Aliasing explicitly via its method instead of the constructor
    plotter.enable_anti_aliasing(aa_type='fxaa')

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

    # 4. Add Aeroshell (Semi-Transparent)
    pv_mesh = pv.wrap(mesh)
    plotter.add_mesh(
        pv_mesh,
        color=mesh_color[:3],
        opacity=mesh_alpha,
        ambient=0.3,
        diffuse=0.7,
        specular=0.1,
        smooth_shading=True
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
        plotter.camera_position = 'iso'  # Initial isometric view orientation
        plotter.camera.azimuth = azim_deg
        plotter.camera.elevation = elev_deg

        if look_at_bounds is not None:
            plotter.reset_camera(bounds=look_at_bounds)
        else:
            plotter.reset_camera()

    img = plotter.screenshot(return_img=True)
    plotter.close()

    return img[:, :, :3]  # (H, W, 3) uint8 RGB


def compute_camera_pose(
    mesh: trimesh.Trimesh,
    azim_deg: float = -60.0,
    elev_deg: float = 25.0,
    distance_mult: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute camera position/focus/up using the same logic as latent_rollout_gif.py."""
    vertices = mesh.vertices
    centroid = vertices.mean(axis=0)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    dist = diag * distance_mult

    azim = np.radians(azim_deg)
    elev = np.radians(elev_deg)
    cam_pos = centroid + dist * np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])

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

    return cam_pos, centroid, up

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
            left = int(round(panel_left + col * (cell_w + gap)))
            top = int(round(panel_top + row * (cell_h + gap)))
            right = int(round(left + cell_w))
            bottom = int(round(top + cell_h))
            fill = cell_a if (row + col) % 2 == 0 else cell_b
            draw.rectangle(
                [(left, top), (right, bottom)],
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
                    x = int(round(inner_left + frac * (inner_right - inner_left)))
                    draw.line(
                        [(x, inner_top), (x, inner_bottom)],
                        fill=finger_color,
                        width=finger_width,
                    )
                for k in range(1, horizontal_lines + 1):
                    frac = k / float(horizontal_lines + 1.0)
                    y = int(round(inner_top + frac * (inner_bottom - inner_top)))
                    draw.line(
                        [(inner_left, y), (inner_right, y)],
                        fill=(84, 112, 142),
                        width=1,
                    )
                y_top = int(round(inner_top + 0.18 * (inner_bottom - inner_top)))
                y_bottom = int(round(inner_top + 0.82 * (inner_bottom - inner_top)))
                draw.line([(inner_left, y_top), (inner_right, y_top)], fill=busbar_color, width=busbar_width)
                draw.line([(inner_left, y_bottom), (inner_right, y_bottom)], fill=busbar_color, width=busbar_width)

    texture = pv.Texture(np.asarray(image))
    texture.repeat = True
    return texture


def draw_latent_panel(  # noqa: PLR0915  # multi-mode rendering function; statement count reflects supported traversal modes
    w: int,
    h: int,
    z_radius: float,
    Z_pretrained: np.ndarray | None,
    z0: float,
    z1: float,
    frame_idx: int,
    n_frames: int,
    colors: dict | None = None,
    trajectory_coords: list[tuple[float, float]] | None = None,
    mode: str = "circle",
    panel_scale_radius: float | None = None,
    show_anchors: bool = True,
    epoch_num: int | None = None,
) -> np.ndarray:
    """Return (H, W, 3) uint8 side panel showing the 2D latent map."""
    _C = colors or {}
    bg = _C.get("bg", (35, 40, 62))
    bg_epoch = _C.get("bg_epoch", bg)
    axis_col = _C.get("axis", (75, 85, 115))
    circle_col = _C.get("circle", (110, 125, 165))
    anchor_col = _C.get("anchor", (220, 90, 90))
    anchor_lbl_col = _C.get("anchor_label", (240, 160, 160))
    progress_col = _C.get("progress", (80, 200, 255))
    cursor_col = _C.get("cursor", (80, 200, 255))
    cursor_out_col = _C.get("cursor_outline", (255, 255, 255))
    axis_lbl_col = _C.get("axis_label", (180, 190, 220))
    coord_text_col = _C.get("coord_text", (130, 195, 255))
    title_col = _C.get("title", (200, 205, 230))

    img = Image.new("RGB", (w, h), color=bg_epoch if mode == "epoch_gif" else bg)
    draw = ImageDraw.Draw(img)

    scale_factor = h / 600.0
    lw_grid = max(1, int(1 * scale_factor))
    lw_axis = max(1, int(1 * scale_factor))
    lw_thick = max(1, int(2 * scale_factor))
    font_info = _get_font(max(8, int(11 * scale_factor)))
    font_title = _get_font(max(9, int(12 * scale_factor)))

    # Epoch counter — top-left in all modes
    if epoch_num is not None:
        font_epoch = _get_font(max(10, int(15 * scale_factor)))
        draw.text(
            (int(8 * scale_factor), int(8 * scale_factor)),
            f"Epoch {epoch_num}",
            fill=title_col,
            font=font_epoch,
        )

    if mode == "epoch_gif":
        # epoch_gif: panel is plain background — no latent map content
        return np.asarray(img)

    cx, cy = w // 2, h // 2

    _scale_ref = panel_scale_radius if panel_scale_radius is not None else z_radius
    scale = (min(w, h) * 0.40) / max(_scale_ref * 1.5, 1e-6)

    def z2px(z0v: float, z1v: float) -> tuple[int, int]:
        return cx + int(z0v * scale), cy - int(z1v * scale)

    # Axis lines
    draw.line([(12, cy), (w - 12, cy)], fill=axis_col, width=lw_axis)
    draw.line([(cx, 12), (cx, h - 12)], fill=axis_col, width=lw_axis)

    # Training manifold circle (shown as reference boundary in circle/spiral modes)
    r_px = int(z_radius * scale)
    if mode in ("circle", "spiral"):
        draw.ellipse(
            [(cx - r_px, cy - r_px), (cx + r_px, cy + r_px)],
            outline=circle_col,
            width=lw_axis,
        )

    # Pretrained anchor dots (Proportionally Scaled)
    if show_anchors and Z_pretrained is not None:
        font_sm = _get_font(max(8, int(11 * scale_factor)))
        for i, (z0a, z1a) in enumerate(Z_pretrained):
            px, py = z2px(float(z0a), float(z1a))
            r = max(2, int(4 * scale_factor))
            offset = max(3, int(7 * scale_factor))
            draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=anchor_col)
            draw.text((px + offset, py - offset), str(i), fill=anchor_lbl_col, font=font_sm)

    # Trajectory / progress visualization
    if mode == "circle":
        if n_frames > 1 and r_px > 5:
            arc_end_deg = -90 + (360.0 * frame_idx / n_frames)
            with contextlib.suppress(Exception):
                draw.arc(
                    [(cx - r_px, cy - r_px), (cx + r_px, cy + r_px)],
                    start=-90,
                    end=arc_end_deg,
                    fill=progress_col,
                    width=lw_thick,
                )
    elif trajectory_coords is not None and len(trajectory_coords) > 1:
        full_pts = [z2px(zc0, zc1) for zc0, zc1 in trajectory_coords]
        draw.line(full_pts, fill=circle_col, width=lw_grid)
        traversed = full_pts[: frame_idx + 1]
        if len(traversed) >= 2:
            draw.line(traversed, fill=progress_col, width=lw_thick)
        if mode == "line":
            ep_r = max(2, int(5 * scale_factor))
            for ep_z0, ep_z1 in (trajectory_coords[0], trajectory_coords[-1]):
                ex, ey = z2px(ep_z0, ep_z1)
                draw.ellipse([(ex - ep_r, ey - ep_r), (ex + ep_r, ey + ep_r)], outline=circle_col, width=lw_thick)
        if mode == "cross":
            ep_r = max(2, int(5 * scale_factor))
            seen: set[tuple[int, int]] = set()
            for ep_z0, ep_z1 in trajectory_coords:
                ep = z2px(ep_z0, ep_z1)
                if ep not in seen and abs(ep_z0) > 1e-6 and abs(ep_z1) > 1e-6:
                    draw.ellipse([(ep[0] - ep_r, ep[1] - ep_r), (ep[0] + ep_r, ep[1] + ep_r)], outline=circle_col, width=lw_thick)
                    seen.add(ep)

    # Current cursor (bright dot - Proportionally Scaled)
    px, py = z2px(z0, z1)
    r = max(3, int(7 * scale_factor))
    draw.ellipse([(px - r, py - r), (px + r, py + r)], fill=cursor_col, outline=cursor_out_col, width=lw_thick)

    # Axis labels & Info Text
    font_label = ImageFont.truetype("DejaVuSans-Bold.ttf", max(9, int(13 * scale_factor)), layout_engine=ImageFont.Layout.BASIC)

    draw.text((w - int(26 * scale_factor), cy - int(16 * scale_factor)), "z₀", fill=axis_lbl_col, font=font_label)
    draw.text((cx + int(5 * scale_factor), int(8 * scale_factor)), "z₁", fill=axis_lbl_col, font=font_label)

    draw.text((int(8 * scale_factor), h - int(36 * scale_factor)), f"z = [{z0:+.3f}, {z1:+.3f}]", fill=coord_text_col, font=font_info)

    _mode_label = {
        "circle": "Circle",
        "spiral": "Spiral",
        "line": "Line",
        "cross": "Cross",
        "epoch_gif": "Epoch GIF",
    }.get(mode, mode.capitalize())
    # Title only shown in latent-space modes (epoch_gif returned early above).
    # Skip title when epoch label is shown to keep top-left uncluttered.
    if epoch_num is None:
        draw.text((int(8 * scale_factor), int(8 * scale_factor)), f"Latent Space · {_mode_label}", fill=title_col, font=font_title)

    return np.asarray(img)


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
    h = frame_rgb.shape[0]
    scale_factor = h / 600.0
    font_epoch = _get_font(max(10, int(15 * scale_factor)))
    anchor_x = x_offset_px + int(8 * scale_factor)
    anchor_y = int(8 * scale_factor)
    label = f"Epoch {epoch_num}"
    if box_color is not None:
        left, top, right, bottom = draw.textbbox((anchor_x, anchor_y), label, font=font_epoch)
        pad_x = int(8 * scale_factor)
        pad_y = int(4 * scale_factor)
        draw.rectangle(
            [(left - pad_x, top - pad_y), (right + pad_x, bottom + pad_y)],
            fill=box_color,
        )
    draw.text(
        (anchor_x, anchor_y),
        label,
        fill=color,
        font=font_epoch,
    )
    return np.asarray(img)


def align_epoch_frame_to_standard_view(
    frame_rgb: np.ndarray,
    standard_width: int,
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    """Shift wide epoch_gif render so framing matches non-epoch modes.

    Non-epoch renders use ``standard_width`` for the 3D viewport and center the
    camera there. Epoch mode renders wider (``standard_width + panel_width``) so
    geometry can spill into the right area. This helper shifts that wide image
    left by half of the extra width, preserving scale while restoring the same
    object placement as non-epoch modes.
    """
    h, w, _ = frame_rgb.shape
    if w <= standard_width:
        return frame_rgb

    extra = w - standard_width
    shift_px = extra // 2
    if shift_px <= 0:
        return frame_rgb

    out = np.empty((h, w, 3), dtype=np.uint8)
    out[:, :, 0] = bg_color[0]
    out[:, :, 1] = bg_color[1]
    out[:, :, 2] = bg_color[2]

    # Move content left; keep right-side overflow visible in the blank area.
    out[:, : (w - shift_px), :] = frame_rgb[:, shift_px:, :]
    return out


def compose_epoch_frame(
    standard_rgb: np.ndarray,
    wide_aligned_rgb: np.ndarray,
    standard_width: int,
    bg_color: tuple[int, int, int],
) -> np.ndarray:
    """Compose epoch frame with exact standard-mode left view plus right overflow.

    Left region [0:standard_width] is taken from the standard-width render to keep
    size/location identical to non-epoch modes. Right region is taken from the
    aligned wide render to preserve spillover beyond the former seam.
    """
    h, w_std, _ = standard_rgb.shape
    h_wide, w_wide, _ = wide_aligned_rgb.shape
    if h != h_wide:
        raise ValueError("compose_epoch_frame requires matching frame heights")
    if w_std != standard_width:
        raise ValueError("compose_epoch_frame standard width mismatch")
    if w_wide <= standard_width:
        return wide_aligned_rgb

    out = np.empty((h, w_wide, 3), dtype=np.uint8)
    out[:, :, 0] = bg_color[0]
    out[:, :, 1] = bg_color[1]
    out[:, :, 2] = bg_color[2]
    out[:, :standard_width, :] = standard_rgb
    out[:, standard_width:, :] = wide_aligned_rgb[:, standard_width:, :]
    return out


# ---------------------------------------------------------------------------
# Trajectory builders
# ---------------------------------------------------------------------------


def _build_trajectory(  # noqa: PLR0912  # supports 4 traversal modes; branches are mode dispatches, not complexity
    mode: str,
    n_frames: int,
    z_radius: float,
    spiral_turns: float = 3.0,
    spiral_radius_start: float = 0.0,
    spiral_radius_end: float | None = None,
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
    pv.set_jupyter_backend('static')
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
        "--output_dir",
        type=Path,
        default=_c("output_dir", "/cluster/work/fuge/AISCO/wp9/pyvista_gif"),
        help="Output directory",
    )
    parser.add_argument("--n_frames", type=int, default=_c("n_frames", 240), help="Frames per full circle")
    parser.add_argument("--resolution", type=int, default=_c("resolution", 40), help="Marching-cubes grid resolution per axis")
    parser.add_argument("--radius", type=float, default=_c("radius"), help="Traversal radius (default: infer from checkpoint)")
    parser.add_argument("--fps", type=int, default=_c("fps", 24), help="GIF playback speed")
    parser.add_argument("--device", default=_dev_default, help="Compute device: cuda | cpu | auto")
    parser.add_argument("--width", type=int, default=_c("width", 4096), help="3D render width in px")
    parser.add_argument("--height", type=int, default=_c("height", 2160), help="3D render height in px")
    parser.add_argument("--panel_width", type=int, default=_c("panel_width", 320), help="Latent-map panel width in px")
    parser.add_argument("--azim", type=float, default=_c("azim", -60.0), help="Camera azimuth in degrees")
    parser.add_argument("--elev", type=float, default=_c("elev", 25.0), help="Camera elevation in degrees")
    parser.add_argument("--mesh_color", default=_c("mesh_color", "#000000"), help="Aeroshell color (hex)")
    parser.add_argument("--render_bg", default=_c("render_bg", "#ffffff"), help="3D render background color (hex)")
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
    parser.add_argument("--solar_selection_color", default=_c("solar_selection_color", "#F0D33E"), help="Solar selection color (hex)")
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

    # Parse render colors
    mesh_color_rgba = _parse_color_rgba_float(args.mesh_color)
    render_bg_rgba = _parse_color_rgba_float(args.render_bg)
    assembly_color_rgba = _parse_color_rgba_float(args.assembly_color, 1.0)
    solar_selection_color_rgba = _parse_color_rgba_float(args.solar_selection_color, 1.0)
    windshield_color_rgba = _parse_color_rgba_float(args.windshield_color, float(args.windshield_alpha))

    # Build panel colors dict (sourced from YAML cfg, with hardcoded fallbacks)
    panel_colors = {
        "bg": _parse_color_rgb(_c("panel_bg", "#23283E")),
        "bg_epoch": _parse_color_rgb(args.render_bg),
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

    device = torch.device(args.device)
    print(f"Device: {device}")

    active_mode = "epoch_gif" if args.epoch_gif else args.mode
    base_epoch_num = _parse_epoch_from_stem(args.checkpoint.stem)

    # --- Load model ---
    model, z_dim, z_radius_ckpt, Z_pretrained = load_checkpoint(args.checkpoint, device)
    z_radius = args.radius if args.radius is not None else z_radius_ckpt

    epoch_checkpoints: list[tuple[int, Path]] = []
    if active_mode == "epoch_gif":
        epoch_checkpoints = _find_epoch_checkpoints(args.checkpoint)
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
        padding = (
            max(global_bounds[1] - global_bounds[0], global_bounds[3] - global_bounds[2]) * 0.25
        )
        global_bounds[0] -= padding; global_bounds[1] += padding
        global_bounds[2] -= padding; global_bounds[3] += padding
        global_bounds[4] -= padding; global_bounds[5] += padding

    frames: list[Any] = []
    last_valid_rgb: np.ndarray | None = None
    last_valid_epoch_base: np.ndarray | None = None
    locked_camera_position: np.ndarray | None = None
    locked_camera_focal_point: np.ndarray | None = None
    locked_camera_view_up: np.ndarray | None = None
    render_width = args.width + args.panel_width if active_mode == "epoch_gif" else args.width
    assembly_smooth_shading = not args.full_wheels
    solar_selection_texture = _build_solar_cell_texture() if args.show_solar_selection else None

    if active_mode == "epoch_gif" and frame_checkpoints:
        # Lock camera from the user-selected checkpoint (not earliest epoch), so
        # epoch_gif framing matches non-epoch runs that use this checkpoint.
        ref_idx = 0
        with contextlib.suppress(ValueError):
            ref_idx = frame_checkpoints.index(args.checkpoint)
        ref_ckpt = frame_checkpoints[ref_idx]
        if ref_ckpt is not None:
            _load_model_state_from_checkpoint(model, ref_ckpt, device)
            ref_z0, ref_z1 = z_coords[ref_idx]
            ref_mesh = extract_mesh(model, extractor, ref_z0, ref_z1, z_dim, device)
            if ref_mesh is not None and len(ref_mesh.vertices) > 0:
                (
                    locked_camera_position,
                    locked_camera_focal_point,
                    locked_camera_view_up,
                ) = compute_camera_pose(
                    ref_mesh,
                    azim_deg=args.azim,
                    elev_deg=args.elev,
                )
                print(f"[camera lock] using reference checkpoint: {ref_ckpt.name}")

    print("\nGenerating frames...")
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
            if last_valid_rgb is not None:
                if active_mode == "epoch_gif":
                    if last_valid_epoch_base is not None:
                        frames.append(
                            draw_epoch_label_on_frame(
                                last_valid_epoch_base,
                                frame_epoch_num,
                                color=panel_colors["title"],
                                x_offset_px=args.width,
                                box_color=panel_colors["bg"],
                            )
                        )
                else:
                    panel = draw_latent_panel(
                        args.panel_width,
                        args.height,
                        z_radius,
                        Z_np,
                        z0,
                        z1,
                        i,
                        total_frames,
                        colors=panel_colors,
                        trajectory_coords=panel_trajectory,
                        mode=active_mode,
                        panel_scale_radius=panel_scale_radius,
                        show_anchors=args.show_anchors,
                        epoch_num=frame_epoch_num,
                    )
                    frames.append(np.concatenate([last_valid_rgb, panel], axis=1))
            continue

        cone_face_mask: Tensor | None = None
        solar_selection_mesh: trimesh.Trimesh | None = None
        windshield_mesh: trimesh.Trimesh | None = None
        mesh_for_render = mesh
        mask_mesh = mesh
        if args.show_windshield or args.show_solar_selection:
            refine_subdiv = args.mask_refine_subdivisions
            if args.show_solar_selection:
                refine_subdiv = max(refine_subdiv, args.solar_mask_refine_subdivisions)
            mask_mesh = _subdivide_mesh_for_masks(mesh, refine_subdiv)
            cone_face_mask = _compute_windshield_face_mask(mask_mesh, args.windshield_assembly_id, device)
        if args.show_windshield and cone_face_mask is not None:
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
        if args.show_solar_selection:
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

        if locked_camera_position is None:
            locked_camera_position, locked_camera_focal_point, locked_camera_view_up = compute_camera_pose(
                mesh,
                azim_deg=args.azim,
                elev_deg=args.elev,
            )
            print("  [camera locked]", end="")

        rgb = render_mesh(
            mesh=mesh_for_render,
            bg_color=render_bg_rgba,
            mesh_color=mesh_color_rgba,
            mesh_alpha=args.mesh_alpha,
            assembly_mesh=inner_assembly,
            assembly_color=assembly_color_rgba,
            width=args.width if active_mode == "epoch_gif" else render_width,
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
        )
        if active_mode == "epoch_gif":
            rgb_wide = render_mesh(
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
            )
            aligned = align_epoch_frame_to_standard_view(
                rgb_wide,
                standard_width=args.width,
                bg_color=panel_colors["bg_epoch"],
            )
            epoch_base = compose_epoch_frame(
                standard_rgb=rgb,
                wide_aligned_rgb=aligned,
                standard_width=args.width,
                bg_color=panel_colors["bg_epoch"],
            )
            composite = draw_epoch_label_on_frame(
                epoch_base,
                frame_epoch_num,
                color=panel_colors["title"],
                x_offset_px=args.width,
                box_color=panel_colors["bg"],
            )
            last_valid_epoch_base = epoch_base
        else:
            panel = draw_latent_panel(
                args.panel_width,
                args.height,
                z_radius,
                Z_np,
                z0,
                z1,
                i,
                total_frames,
                colors=panel_colors,
                trajectory_coords=panel_trajectory,
                mode=active_mode,
                panel_scale_radius=panel_scale_radius,
                show_anchors=args.show_anchors,
                epoch_num=frame_epoch_num,
            )
            composite = np.concatenate([rgb, panel], axis=1) 

        frames.append(composite)
        last_valid_rgb = rgb
        print(f"  → {len(mesh.vertices):,} verts")

    if not frames:
        sys.exit("ERROR: no valid frames were rendered — check model and checkpoint")

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
            quantizer='nq',  # NeuQuant high-quality neural network quantizer
            palettesize=256
        )
        print(f"Done.  GIF saved to: {out_path}")

if __name__ == "__main__":
    main()
