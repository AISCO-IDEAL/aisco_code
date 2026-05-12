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


def generate_survey_png( # noqa: PLR0912
    input_dir: Path,
    output_path: Path | None = None,
    layout: str = "strip",
    thumb_w: int | None = None,
    thumb_h: int | None = None,
    cam_json: Path | None = None,
) -> Path:
    """Render all ``epoch_XXXXX.ply`` files in *input_dir* to a single composite PNG.

    Each PLY is rendered to a thumbnail; thumbnails are assembled side-by-side
    (``layout='strip'``) or in a roughly-square grid (``layout='grid'``).

    Args:
        input_dir: Directory containing ``epoch_XXXXX.ply`` files.
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
        ValueError: If no ``epoch_XXXXX.ply`` files are found in *input_dir*.
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

    matches: list[tuple[int, Path]] = []
    for fpath in input_dir.iterdir():
        m = _FLAT_PATTERN.match(fpath.name)
        if m:
            matches.append((int(m.group(1)), fpath))
    if not matches:
        raise ValueError(f"No epoch_XXXXX.ply files found in {input_dir}.")
    matches.sort(key=lambda x: x[0])
    file_paths = [fp for _, fp in matches]
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
    if layout == "grid":
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
