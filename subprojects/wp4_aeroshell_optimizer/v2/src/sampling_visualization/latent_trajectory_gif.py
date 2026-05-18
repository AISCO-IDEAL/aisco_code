"""Latent-trajectory GIF tool for pretrained ConcatWIREV5 checkpoints.

Renders a GIF where each frame shows the shape produced by the model at a
specific 2D latent coordinate (z0, z1), with an inset 2D plot of the
trajectory through latent space and a marker at the current point. Makes
the latent-to-shape mapping intuitive by sweeping a coordinate path while
the corresponding shape morphs.

Trajectory types
----------------
1. path     — piecewise-linear interpolation through >=2 (z0, z1) waypoints.
2. circle   — sweep theta in [0, 2pi) at fixed radius r.
3. diameter — sweep r in [-R, R] along a fixed theta (cross the disk).

Usage
-----
    python3 subprojects/wp4_aeroshell_optimizer/v2/src/sampling_visualization/latent_trajectory_gif.py \\
        --checkpoint /path/to/pretrained_latent_<hash>.pth \\
        --trajectory circle [--radius 2.0] [--n_frames 60]

    python3 ... --trajectory diameter --theta 45 [--radius 2.0] [--n_frames 60]

    python3 ... --trajectory path --waypoints "0,2;2,0;0,-2;-2,0" [--closed] [--n_per_segment 20]
"""

from __future__ import annotations

import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import argparse  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from dataclasses import replace as _dc_replace  # noqa: E402
from pathlib import Path  # noqa: E402

import imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyrender  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402
from torch import Tensor  # noqa: E402

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.explore_latent_space import (  # noqa: E402
    _extract_and_save,
    _load_checkpoint,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (  # noqa: E402
    BACKGROUND_COLOR,
    CAMERA_DIRECTION,
    CAMERA_DISTANCE_MULTIPLIER,
    CAMERA_ELEVATION_ANGLE,
    FPS,
    POINT_SIZE,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    _compute_camera_pose,
    _get_vertices,
    _render_frame,
)
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config  # noqa: E402

# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------


def _build_trajectory(
    args: argparse.Namespace,
    z_radius: float,
    z_dim: int,
    device: torch.device,
) -> tuple[Tensor, np.ndarray, str]:
    """Build a (N, z_dim) tensor of latent vectors plus a (N, 2) numpy array of (z0, z1) for plotting.

    Higher-than-2 latent dimensions are zero-filled to match the convention used by
    `explore_latent_space.py`. The trajectory name is suitable for use as a directory name.
    """
    if args.trajectory == "circle":
        R = args.radius if args.radius is not None else z_radius
        n = args.n_frames
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        z2d = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)
        name = f"circle_r={R:.2f}_n={n}"

    elif args.trajectory == "diameter":
        R = args.radius if args.radius is not None else z_radius
        n = args.n_frames
        rs = np.linspace(-R, R, n)
        theta_rad = math.radians(args.theta)
        z2d = np.stack([rs * math.cos(theta_rad), rs * math.sin(theta_rad)], axis=1)
        name = f"diameter_theta={args.theta:.0f}_r={R:.2f}_n={n}"

    elif args.trajectory == "path":
        waypoints: list[tuple[float, float]] = []
        for tup in args.waypoints.split(";"):
            parts = tup.strip().split(",")
            if len(parts) != 2:
                raise ValueError(f"Bad waypoint '{tup}' — expected 'z0,z1'.")
            waypoints.append((float(parts[0]), float(parts[1])))
        if len(waypoints) < 2:
            raise ValueError("--trajectory path requires at least 2 waypoints.")
        if args.closed:
            waypoints.append(waypoints[0])
        segments: list[np.ndarray] = []
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            ts = np.linspace(0.0, 1.0, args.n_per_segment, endpoint=False)
            seg = np.stack([a[0] * (1 - ts) + b[0] * ts, a[1] * (1 - ts) + b[1] * ts], axis=1)
            segments.append(seg)
        segments.append(np.array([waypoints[-1]]))  # include final waypoint exactly
        z2d = np.concatenate(segments, axis=0)
        name = f"path_{len(waypoints) - 1}seg_n={args.n_per_segment}"

    else:
        raise ValueError(f"Unknown trajectory '{args.trajectory}'.")

    n_frames = z2d.shape[0]
    z_full = np.zeros((n_frames, z_dim), dtype=np.float32)
    z_full[:, 0] = z2d[:, 0]
    if z_dim > 1:
        z_full[:, 1] = z2d[:, 1]
    return torch.from_numpy(z_full).to(device), z2d, name


# ---------------------------------------------------------------------------
# Latent overlay (matplotlib inset)
# ---------------------------------------------------------------------------


def _render_latent_overlay(
    trajectory_2d: np.ndarray,
    current_idx: int,
    z_anchors_2d: np.ndarray,
    z_radius: float,
    size_px: int,
) -> np.ndarray:
    """Render the (z0, z1) plot with trajectory + current marker to an (H, W, 4) RGBA array.

    The plot shows the pretrained ring (faint), pretrained anchors (grey dots),
    the full trajectory (blue line), and the current point as a bold red dot.
    """
    dpi = 100
    side_in = size_px / dpi
    fig, ax = plt.subplots(figsize=(side_in, side_in), dpi=dpi)

    traj_max = float(np.max(np.linalg.norm(trajectory_2d, axis=1)))
    R_max = max(traj_max, 1.5 * z_radius, 1.0)
    margin = 0.15 * R_max
    L = R_max + margin
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_aspect("equal")

    theta = np.linspace(0.0, 2.0 * np.pi, 200)
    ax.plot(z_radius * np.cos(theta), z_radius * np.sin(theta), color="gray", lw=0.8, alpha=0.5)

    if z_anchors_2d.size > 0:
        ax.scatter(z_anchors_2d[:, 0], z_anchors_2d[:, 1], c="gray", s=20, alpha=0.7, zorder=2)

    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], color="tab:blue", lw=1.5, alpha=0.7, zorder=3)

    cur = trajectory_2d[current_idx]
    ax.scatter([cur[0]], [cur[1]], c="tab:red", s=70, zorder=4, edgecolors="black", linewidths=0.8)

    r = float(np.linalg.norm(cur))
    theta_deg = float(np.degrees(np.arctan2(cur[1], cur[0])))
    ax.set_title(f"z=({cur[0]:+.2f}, {cur[1]:+.2f})  r={r:.2f}  θ={theta_deg:+.1f}°", fontsize=9)
    ax.set_xlabel("z0", fontsize=8)
    ax.set_ylabel("z1", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def _composite_frame(shape_rgb: np.ndarray, overlay_rgba: np.ndarray, margin: int = 20) -> np.ndarray:
    """Paste the latent overlay (with a semi-opaque white panel behind it) onto the shape frame, top-right."""
    frame = Image.fromarray(shape_rgb, "RGB").convert("RGBA")
    overlay = Image.fromarray(overlay_rgba, "RGBA")
    panel = Image.new("RGBA", overlay.size, (255, 255, 255, 210))
    overlay_on_panel = Image.alpha_composite(panel, overlay)
    pos = (frame.width - overlay_on_panel.width - margin, margin)
    frame.alpha_composite(overlay_on_panel, dest=pos)
    return np.asarray(frame.convert("RGB"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: PLR0915
    _viz_cfg_path = Path(__file__).parents[2] / "configs" / "vis_config.yaml"
    _lse_cfg: dict = {}
    if _viz_cfg_path.exists():
        _lse_cfg = yaml.safe_load(_viz_cfg_path.read_text()).get("latent_space_exploration", {})

    parser = argparse.ArgumentParser(description="Latent-trajectory GIF for a pretrained latent checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained_latent_*.pth")
    parser.add_argument("--trajectory", required=True, choices=["path", "circle", "diameter"])
    parser.add_argument("--waypoints", default=None, help="path mode: 'z0,z1;z0,z1;...' (>=2 tuples)")
    parser.add_argument("--n_per_segment", type=int, default=20, help="path mode: interp points per segment")
    parser.add_argument("--closed", action="store_true", help="path mode: close the loop back to the first waypoint")
    parser.add_argument("--radius", type=float, default=None, help="circle/diameter radius (default: inferred z_radius)")
    parser.add_argument("--theta", type=float, default=0.0, help="diameter sweep direction in degrees")
    parser.add_argument("--n_frames", type=int, default=60, help="circle/diameter number of frames")
    parser.add_argument(
        "--resolution",
        type=int,
        default=int(_lse_cfg.get("resolution", 64)),
        help="Marching cubes grid resolution (lower = faster)",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--fps", type=int, default=FPS, help="GIF playback frames per second")
    parser.add_argument("--out", default=None, help="Output dir (default auto-derived next to checkpoint)")
    parser.add_argument("--keep_plys", action="store_true", help="Keep intermediate frame PLYs after writing the GIF")
    args = parser.parse_args()

    if args.trajectory == "path" and not args.waypoints:
        parser.error("--trajectory path requires --waypoints (e.g. '0,2;1,1;-2,0').")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    model, Z_pretrained, z_dim, z_radius, _shape_names, prism = _load_checkpoint(ckpt_path, device)

    z_trajectory, z2d, traj_name = _build_trajectory(args, z_radius, z_dim, device)

    if args.out is not None:
        out_dir = Path(args.out)
    else:
        stem = ckpt_path.stem
        ckpt_hash = stem.removeprefix("pretrained_latent_")
        out_dir = ckpt_path.parent / f"latent_trajectory_{ckpt_hash}" / traj_name
        print(f"[traj_gif] Auto-derived output dir: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = out_dir / "frames_ply"
    ply_dir.mkdir(exist_ok=True)

    _mc_cfg = load_mesh_extraction_config()
    _mc_cfg = _dc_replace(_mc_cfg, device=str(device))
    extractor = MeshSDFExtractor(config=_mc_cfg)
    extractor.set_grid(
        bbox=[
            [prism["x"][0], prism["x"][1]],
            [prism["y"][0], prism["y"][1]],
            [prism["z"][0], prism["z"][1]],
        ],
        resolution=args.resolution,
    )

    print(f"\nGenerating {len(z_trajectory)} frames along trajectory '{traj_name}'")
    valid_flags: list[bool] = []
    for i in range(len(z_trajectory)):
        z_vec = z_trajectory[i]
        ply_path = ply_dir / f"frame_{i:04d}.ply"
        valid = _extract_and_save(model, z_vec, ply_path, extractor)
        valid_flags.append(valid)
        status = "OK" if valid else "EMPTY"
        print(f"  [{i + 1:3d}/{len(z_trajectory)}] z=({z2d[i, 0]:+.3f},{z2d[i, 1]:+.3f})  {status}")

    valid_plys = [ply_dir / f"frame_{i:04d}.ply" for i, ok in enumerate(valid_flags) if ok]
    if not valid_plys:
        raise RuntimeError("No valid meshes were extracted along the trajectory; cannot render GIF.")

    all_verts = [v for v in (_get_vertices(fp) for fp in valid_plys) if v is not None]
    cam_pose = _compute_camera_pose(
        np.vstack(all_verts),
        direction=CAMERA_DIRECTION,
        distance_mult=CAMERA_DISTANCE_MULTIPLIER,
        elevation_angle=CAMERA_ELEVATION_ANGLE,
    )

    z_anchors_2d = Z_pretrained[:, :2].cpu().numpy() if Z_pretrained.shape[1] >= 2 else np.zeros((0, 2))
    overlay_size_px = max(WINDOW_WIDTH // 4, 240)
    bg_rgb = (np.array(BACKGROUND_COLOR[:3]) * 255).astype(np.uint8)
    blank_frame = np.tile(bg_rgb[None, None, :], (WINDOW_HEIGHT, WINDOW_WIDTH, 1))

    renderer = pyrender.OffscreenRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, point_size=POINT_SIZE)
    frames: list[np.ndarray] = []
    try:
        for i in range(len(z_trajectory)):
            if valid_flags[i]:
                shape_rgb = _render_frame(
                    renderer,
                    ply_dir / f"frame_{i:04d}.ply",
                    cam_pose,
                    camera_direction=CAMERA_DIRECTION,
                    add_overlay=False,
                )
            else:
                shape_rgb = blank_frame
            overlay = _render_latent_overlay(z2d, i, z_anchors_2d, z_radius, overlay_size_px)
            frames.append(_composite_frame(shape_rgb, overlay))
    finally:
        renderer.delete()

    gif_path = out_dir / "latent_trajectory.gif"
    imageio.mimsave(str(gif_path), frames, fps=args.fps)  # type: ignore[arg-type]
    print(f"\nGIF written: {gif_path}  ({len(frames)} frames @ {args.fps} fps)")

    if not args.keep_plys:
        for fp in ply_dir.glob("*.ply"):
            fp.unlink()
        ply_dir.rmdir()

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
