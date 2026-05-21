#!/usr/bin/env python3
"""Visualise a PLY mesh with configurable color, lighting, and camera.

All visual settings live in a YAML config (default: configs/mesh_viz.yaml).

Usage
-----
    python scripts/mesh_viz.py                          # default config
    python scripts/mesh_viz.py --mesh /path/to.ply
    python scripts/mesh_viz.py --interactive            # write HTML instead of PNG
    python scripts/mesh_viz.py --config /path/to/cfg.yaml
    python scripts/mesh_viz.py --out /path/to/out.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import trimesh
import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "mesh_viz.yaml"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)  # type: ignore[no-any-return]


def _get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    """Load and validate a triangle mesh.

    Args:
        mesh_path: Path to a PLY / OBJ / STL file.

    Returns:
        A ``trimesh.Trimesh`` instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file does not contain a triangle mesh.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}\nSet 'mesh_path' in the YAML config or pass --mesh.")
    geom = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(geom, trimesh.Trimesh):
        raise TypeError(
            f"Expected a triangle mesh, got {type(geom).__name__}.\n"
            "Convert the file to a PLY / OBJ / STL with a single mesh body."
        )
    return geom


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------


def _rgb_str(rgba: list[float]) -> str:
    r, g, b = [round(c * 255) for c in rgba[:3]]
    return f"rgb({r},{g},{b})"


def _rgba_str(rgba: list[float]) -> str:
    r, g, b = [round(c * 255) for c in rgba[:3]]
    a = float(rgba[3]) if len(rgba) > 3 else 1.0
    return f"rgba({r},{g},{b},{a:.3f})"


# ---------------------------------------------------------------------------
# Figure builder
# ---------------------------------------------------------------------------


def build_figure(mesh: trimesh.Trimesh, cfg: dict[str, Any]) -> go.Figure:
    """Build a Plotly figure: one solid mesh trace with optional wireframe.

    Args:
        mesh: Loaded triangle mesh.
        cfg:  Full config dict loaded from YAML.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    # ---- Color & opacity -------------------------------------------------
    face_color = list(_get(cfg, "color") or [0.30, 0.55, 0.80, 1.0])
    opacity = float(_get(cfg, "opacity", default=1.0))

    # ---- Lighting --------------------------------------------------------
    # ambient  : base brightness of surfaces facing away from the light
    # diffuse  : shading strength based on angle to light (main shaping cue)
    # specular : shiny highlight intensity
    # roughness: 0 = mirror-like, 1 = fully matte
    # fresnel  : edge-glow effect (0-5); 0 = off
    ambient = float(_get(cfg, "lighting", "ambient", default=0.4))
    diffuse = float(_get(cfg, "lighting", "diffuse", default=0.8))
    specular = float(_get(cfg, "lighting", "specular", default=0.3))
    roughness = float(_get(cfg, "lighting", "roughness", default=0.5))
    fresnel = float(_get(cfg, "lighting", "fresnel", default=0.2))
    light_x = float(_get(cfg, "lighting", "position_x", default=1.0))
    light_y = float(_get(cfg, "lighting", "position_y", default=1.0))
    light_z = float(_get(cfg, "lighting", "position_z", default=2.0))

    # ---- Camera ----------------------------------------------------------
    # Plotly's camera eye is in *normalised scene space*, not data units, so
    # `zoom` works the same regardless of mesh scale.
    # zoom ~2.0 → mesh fills the frame.  Smaller = closer. Larger = farther.
    azim = float(_get(cfg, "camera", "azimuth", default=-30))
    elev = float(_get(cfg, "camera", "elevation", default=20))
    zoom = float(_get(cfg, "camera", "zoom", default=2.0))
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)
    eye_x = zoom * math.cos(elev_rad) * math.cos(azim_rad)
    eye_y = zoom * math.cos(elev_rad) * math.sin(azim_rad)
    eye_z = zoom * math.sin(elev_rad)

    # ---- Canvas ----------------------------------------------------------
    w_px = int(_get(cfg, "image_width", default=1600))
    h_px = int(_get(cfg, "image_height", default=800))
    bg = _rgba_str(list(_get(cfg, "background_color") or [1.0, 1.0, 1.0, 1.0]))

    # ---- Mesh trace ------------------------------------------------------
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 0],
        y=mesh.vertices[:, 1],
        z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color=_rgb_str(face_color),
        opacity=opacity,
        flatshading=False,
        lighting={
            "ambient": ambient,
            "diffuse": diffuse,
            "specular": specular,
            "roughness": roughness,
            "fresnel": fresnel,
        },
        lightposition={"x": light_x, "y": light_y, "z": light_z},
        hoverinfo="skip",
    )

    traces: list[go.BaseTraceType] = [mesh_trace]

    # ---- Optional wireframe ----------------------------------------------
    edge_color = list(_get(cfg, "edges", "color") or [0.15, 0.15, 0.15, 0.0])
    if not (len(edge_color) > 3 and edge_color[3] < 0.01):
        faces = mesh.faces
        v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
        all_edges = np.sort(
            np.concatenate(
                [
                    np.stack([v0, v1], axis=1),
                    np.stack([v1, v2], axis=1),
                    np.stack([v2, v0], axis=1),
                ]
            ),
            axis=1,
        )
        edges = np.unique(all_edges, axis=0)
        n = len(edges)
        sx, sy, sz = np.empty(n * 3), np.empty(n * 3), np.empty(n * 3)
        sx[0::3] = mesh.vertices[edges[:, 0], 0]
        sx[1::3] = mesh.vertices[edges[:, 1], 0]
        sx[2::3] = np.nan
        sy[0::3] = mesh.vertices[edges[:, 0], 1]
        sy[1::3] = mesh.vertices[edges[:, 1], 1]
        sy[2::3] = np.nan
        sz[0::3] = mesh.vertices[edges[:, 0], 2]
        sz[1::3] = mesh.vertices[edges[:, 1], 2]
        sz[2::3] = np.nan
        traces.append(
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="lines",
                line={"color": _rgba_str(edge_color), "width": 1},
                hoverinfo="skip",
            )
        )

    # ---- Layout ----------------------------------------------------------
    fig = go.Figure(data=traces)
    fig.update_layout(
        width=w_px,
        height=h_px,
        paper_bgcolor=bg,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "camera": {
                "eye": {"x": eye_x, "y": eye_y, "z": eye_z},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "bgcolor": bg,
            "aspectmode": "data",
        },
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PLY mesh visualisation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--mesh", type=Path, default=None, help="Override mesh_path from config")
    parser.add_argument("--out", type=Path, default=None, help="Override output_path from config")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Write an interactive HTML file instead of a static PNG",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config not found: {args.config}\nPass --config to specify a YAML config file.")

    cfg = _load_config(args.config)
    if args.mesh is not None:
        cfg["mesh_path"] = str(args.mesh)

    mesh_path = _get(cfg, "mesh_path")
    if mesh_path is None:
        raise ValueError("No mesh_path set.\nAdd 'mesh_path: /path/to/mesh.ply' to your config or pass --mesh.")

    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    print(f"  Vertices: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}")

    print("Building figure …")
    fig = build_figure(mesh, cfg)

    output_path: Path
    if args.out is not None:
        output_path = args.out
    else:
        mesh_stem = Path(mesh_path).stem
        output_dir = Path(str(_get(cfg, "output_dir", default=".")))
        output_path = output_dir / f"{mesh_stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.interactive or output_path.suffix.lower() in (".html", ".htm"):
        html_path = output_path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"Saved: {html_path}")
    else:
        scale = int(_get(cfg, "render_scale", default=2))
        fig.write_image(str(output_path), scale=scale)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
