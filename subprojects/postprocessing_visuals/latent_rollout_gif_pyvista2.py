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
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

# MUST be set before any OpenGL / pyrender import.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse

import imageio
import numpy as np
import pyrender
import torch
import trimesh
import yaml
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
import pyvista as pv


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

_PRISM_X = (-2900.0, 2900.0)
_PRISM_Y = (-1150.0, 1150.0)
_PRISM_Z = (0.0, 1650.0)

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
    # --- 1. Parent package stubs ---
    for pkg in [
        "subprojects.wp4_aeroshell_optimizer",
        "subprojects.wp4_aeroshell_optimizer.v2",
        "subprojects.wp4_aeroshell_optimizer.v2.src",
        "subprojects.wp4_aeroshell_optimizer.v2.src.models",
    ]:
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = []  # type: ignore[attr-defined]  # marks it as a package
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
) -> np.ndarray:
    """Render mesh and inner assembly using PyVista with Order-Independent Transparency."""
    print("!!!!width", width, "height", height, "!!!!")
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
            smooth_shading=True, 
            specular=0.2,          # Slightly raise specular reflection highlight
            specular_power=50,     # Tighten shine focus to accentuate smooth curvature
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
            print(f"!!!Trying font: {fp}")
            return ImageFont.truetype(fp, size)
        except Exception:
            print(f"!!!Trying font failed: {fp}")
            pass
    return ImageFont.load_default()


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
    axis_col = _C.get("axis", (75, 85, 115))
    circle_col = _C.get("circle", (110, 125, 165))
    anchor_col = _C.get("anchor", (220, 90, 90))
    anchor_lbl_col = _C.get("anchor_label", (240, 160, 160))
    progress_col = _C.get("progress", (80, 200, 255))
    cursor_col = _C.get("cursor", (80, 200, 255))
    cursor_out_col = _C.get("cursor_outline", (255, 255, 255))
    axis_lbl_col = _C.get("axis_label", (180, 190, 220))
    coord_text_col = _C.get("coord_text", (130, 195, 255))
    frame_text_col = _C.get("frame_text", (140, 145, 170))
    title_col = _C.get("title", (200, 205, 230))

    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)

    scale_factor = h / 600.0
    lw_grid = max(1, int(1 * scale_factor))
    lw_axis = max(1, int(1 * scale_factor))
    lw_thick = max(1, int(2 * scale_factor))
    font_info = _get_font(max(8, int(11 * scale_factor)))
    font_title = _get_font(max(9, int(12 * scale_factor)))

    # Epoch counter — top-left, always shown when epoch_num is provided
    if epoch_num is not None:
        font_epoch = _get_font(max(10, int(15 * scale_factor)))
        draw.text(
            (int(8 * scale_factor), int(8 * scale_factor)),
            f"Epoch {epoch_num}",
            fill=title_col,
            font=font_epoch,
        )
        draw.text(
            (int(8 * scale_factor), h - int(20 * scale_factor)),
            f"frame {frame_idx + 1} / {n_frames}",
            fill=frame_text_col,
            font=font_info,
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
    draw.text((int(8 * scale_factor), h - int(20 * scale_factor)), f"frame {frame_idx + 1} / {n_frames}", fill=frame_text_col, font=font_info)

    _mode_label = {
        "circle": "Circle",
        "spiral": "Spiral",
        "line": "Line",
        "cross": "Cross",
        "epoch_gif": "Epoch GIF",
    }.get(mode, mode.capitalize())
    # Title only shown in latent-space modes (epoch_gif returned early above)
    if epoch_num is None:
        draw.text((int(8 * scale_factor), int(8 * scale_factor)), f"Latent Space · {_mode_label}", fill=title_col, font=font_title)

    return np.asarray(img)


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
        print(f"!!!Building spiral trajectory with {n_frames} frames, radius {z_radius}, turns {spiral_turns}")
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
    parser.add_argument("--mesh_color", default=_c("mesh_color", "#33ffff"), help="Aeroshell color (hex)")
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
        "--show_assembly",
        action="store_true",
        default=bool(_c("show_assembly", False)),
        help="Render inner assembly inside the aeroshell",
    )
    parser.add_argument("--assembly_color", default=_c("assembly_color", "#F39C12"), help="Inner assembly color (hex)")
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

    # Build panel colors dict (sourced from YAML cfg, with hardcoded fallbacks)
    panel_colors = {
        "bg": _parse_color_rgb(_c("panel_bg", "#23283E")),
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
        else:
            print("[assembly] WARN: could not parse epoch from checkpoint stem, skipping")

    # --- Build extractor once ---
    extractor = build_extractor(device, args.resolution, bbox_padding=args.bbox_padding)

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
    locked_camera_position: np.ndarray | None = None
    locked_camera_focal_point: np.ndarray | None = None
    locked_camera_view_up: np.ndarray | None = None

    print("\nGenerating frames...")
    for i, (z0, z1) in enumerate(z_coords):
        frame_ckpt = frame_checkpoints[i]
        frame_epoch_num: int | None = None
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

        if locked_camera_position is None:
            locked_camera_position, locked_camera_focal_point, locked_camera_view_up = compute_camera_pose(
                mesh,
                azim_deg=args.azim,
                elev_deg=args.elev,
            )
            print("  [camera locked]", end="")

        rgb = render_mesh(
            mesh=mesh,
            bg_color=render_bg_rgba,
            mesh_color=mesh_color_rgba,
            mesh_alpha=args.mesh_alpha,
            assembly_mesh=inner_assembly,
            assembly_color=assembly_color_rgba,
            width=args.width,
            height=args.height,
            azim_deg=args.azim,
            elev_deg=args.elev,
            look_at_bounds=global_bounds,
            camera_position=locked_camera_position,
            camera_focal_point=locked_camera_focal_point,
            camera_view_up=locked_camera_view_up,
        )
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
        composite = np.concatenate([rgb, panel], axis=1)  # (H, W+panel, 3)
        if i == 0:
            Image.fromarray(composite).save("/cluster/work/fuge/AISCO/wp9/ginn_gif/DEBUG_FRAME_3.png")
            print(f"  → DEBUG_FRAME_2.png saved")
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
