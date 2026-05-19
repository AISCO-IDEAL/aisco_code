"""Render a directory of PLY snapshots to a single composite PNG.

Intended for the latent circle survey: all N angle snapshots are visible
at a glance in one strip or grid image.  Uses the same pyrender / EGL
infrastructure as ply_to_gif.py but produces a static PNG rather than an
animation.
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pyrender
import yaml
from PIL import Image

_CFG = yaml.safe_load((Path(__file__).parents[2] / "configs" / "vis_config.yaml").read_text())
_WINDOW_WIDTH: int = _CFG["window_width"]
_WINDOW_HEIGHT: int = _CFG["window_height"]
_CAMERA_DIRECTION: str = _CFG["camera_direction"]
_CAMERA_DISTANCE_MULTIPLIER: float = _CFG["camera_distance_multiplier"]
_CAMERA_ELEVATION_ANGLE: float = _CFG["camera_elevation_angle"]
_POINT_SIZE: int = _CFG["point_size"]
_BACKGROUND_COLOR: list[float] = _CFG["background_color"]

_FLAT_PATTERN = re.compile(r"^epoch_?(\d{5})\.ply$")
_SURVEY_PATTERN = re.compile(r"^(angle_\d{3}|x_\d{2}_y_\d{2}|r\d{2}_a\d{3})\.ply$")


def generate_survey_png(  # noqa: PLR0915, PLR0912
    input_dir: Path,
    output_path: Path | None = None,
    layout: str = "strip",
    thumb_w: int | None = None,
    thumb_h: int | None = None,
    cam_json: Path | None = None,
) -> Path:
    """Render all ``angle_XXX.ply`` files in *input_dir* to a single composite PNG.

    Each PLY is rendered to a thumbnail; thumbnails are assembled side-by-side
    (``layout='strip'``) or in a roughly-square grid (``layout='grid'``).

    Args:
        input_dir: Directory containing ``angle_XXX.ply`` files.
        output_path: Destination PNG path.  Defaults to
            ``input_dir / "survey.png"``.
        layout: ``"strip"`` (single row) or ``"grid"`` (NxN grid).
        thumb_w: Thumbnail width in pixels.  Defaults to
            ``window_width // 4`` from vis_config.yaml.
        thumb_h: Thumbnail height in pixels.  Defaults to
            ``window_height // 4`` from vis_config.yaml.
        cam_json: Optional path to a ``camera_config.json`` from a prior
            interactive setup.  Overrides auto-computed camera when valid.

    Returns:
        Path to the written PNG.

    Raises:
        ValueError: If no ``angle_XXX.ply`` files are found in *input_dir*.
    """
    from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (  # noqa: PLC0415
        _compute_camera_pose,
        _get_vertices,
        _load_camera_pose,
        _render_frame,
    )

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    input_dir = Path(input_dir)
    if output_path is None:
        output_path = input_dir / "survey.png"

    tw = thumb_w or max(_WINDOW_WIDTH // 4, 64)
    th = thumb_h or max(_WINDOW_HEIGHT // 4, 64)

    matches = [fpath for fpath in input_dir.iterdir() if _SURVEY_PATTERN.match(fpath.name)]
    if not matches:
        matches = [fpath for fpath in input_dir.iterdir() if fpath.suffix == ".ply"]
    if not matches:
        raise ValueError(f"No PLY files found in {input_dir}.")

    # Sort: if all match angle pattern, sort numerically by angle.
    # If all match grid pattern, sort by y then x index.
    # Mixed patterns shouldn't happen in practice.
    def _sort_key(fpath: Path) -> tuple:
        name = fpath.stem
        if name.startswith("angle_"):
            # angle_XXX -> (0, XXX)
            return (0, int(name.split("_")[1]))
        elif name.startswith("x_"):
            # x_YY_y_ZZ -> (1, YY, ZZ)
            parts = name.split("_")
            x_idx = int(parts[1])
            y_idx = int(parts[3])
            return (1, y_idx, x_idx)  # sort by y, then x (row-major order)
        elif name.startswith("r") and "_a" in name:
            # r{r_idx:02d}_a{angle_idx:03d} -> (2, r_idx, angle_idx)
            # Row = radius ring, col = angle (row-major: r outer loop)
            r_part, a_part = name.split("_a")
            r_idx = int(r_part[1:])  # strip leading 'r'
            a_idx = int(a_part)
            return (2, r_idx, a_idx)
        return (3, 0)  # fallback

    matches.sort(key=_sort_key)
    file_paths = matches
    n = len(file_paths)

    # Camera: shared pose computed from the union of all vertex positions.
    cam_pose_override: np.ndarray | None = None
    if cam_json is not None and Path(cam_json).exists():
        cam_pose_override = _load_camera_pose(Path(cam_json))

    if cam_pose_override is not None:
        cam_pose = cam_pose_override
    else:
        all_verts = [v for v in (_get_vertices(fp) for fp in file_paths) if v is not None]
        if not all_verts:
            raise ValueError(f"No vertex data in PLY files in {input_dir}.")
        cam_pose = _compute_camera_pose(
            np.vstack(all_verts),
            direction=_CAMERA_DIRECTION,
            distance_mult=_CAMERA_DISTANCE_MULTIPLIER,
            elevation_angle=_CAMERA_ELEVATION_ANGLE,
        )

    # Render each frame at full resolution, then resize to thumbnail.
    renderer = pyrender.OffscreenRenderer(_WINDOW_WIDTH, _WINDOW_HEIGHT, point_size=_POINT_SIZE)
    thumbs: list[Image.Image] = []
    try:
        for fp in file_paths:
            frame = _render_frame(renderer, fp, cam_pose, camera_direction=_CAMERA_DIRECTION, add_overlay=False)
            img = Image.fromarray(frame).resize((tw, th), Image.LANCZOS)
            thumbs.append(img)
    finally:
        renderer.delete()

    # Assemble composite canvas.
    if layout == "pizza":
        # rows = distinct radii, cols = angles per ring
        r_indices = sorted({_sort_key(fp)[1] for fp in file_paths if _sort_key(fp)[0] == 2})
        a_indices = sorted({_sort_key(fp)[2] for fp in file_paths if _sort_key(fp)[0] == 2})
        rows = len(r_indices) if r_indices else 1
        cols = len(a_indices) if a_indices else n
    elif layout == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:  # strip
        cols = n
        rows = 1

    canvas = Image.new("RGB", (cols * tw, rows * th), color=(255, 255, 255))
    for i, thumb in enumerate(thumbs):
        col = i % cols
        row = i // cols
        canvas.paste(thumb, (col * tw, row * th))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path))
    print(
        json.dumps(
            {
                "event": "survey_png_completed",
                "output_path": str(output_path),
                "n_thumbs": n,
                "layout": layout,
                "grid": f"{cols}x{rows}",
            }
        )
    )
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render PLY files in a directory to a survey PNG.")
    parser.add_argument("input_dir", type=Path, help="Directory containing PLY files.")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path. Defaults to <input_dir>/survey.png.")
    parser.add_argument("--layout", default="grid", choices=["strip", "grid", "pizza"], help="Thumbnail layout.")
    parser.add_argument("--thumb_w", type=int, default=None, help="Thumbnail width in pixels.")
    parser.add_argument("--thumb_h", type=int, default=None, help="Thumbnail height in pixels.")
    parser.add_argument("--cam_json", type=Path, default=None, help="Optional camera_config.json override.")
    args = parser.parse_args()

    generate_survey_png(
        input_dir=args.input_dir,
        output_path=args.out,
        layout=args.layout,
        thumb_w=args.thumb_w,
        thumb_h=args.thumb_h,
        cam_json=args.cam_json,
    )
