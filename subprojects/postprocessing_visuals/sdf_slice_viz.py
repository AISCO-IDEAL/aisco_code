#!/usr/bin/env python3
"""Visualize a mesh alongside an analytical SDF slice in a configurable plane.

Uses Plotly + kaleido for GPU-accelerated rendering with correct depth
buffering (no painter's algorithm artefacts).

Usage
-----
    python scripts/sdf_slice_viz.py                          # default config
    python scripts/sdf_slice_viz.py --config path/to/cfg.yaml
    python scripts/sdf_slice_viz.py --interactive            # save HTML instead
    python scripts/sdf_slice_viz.py --mesh /path/to/mesh.ply
    python scripts/sdf_slice_viz.py --out /path/to/out.png

All visual settings live in the YAML config (configs/sdf_slice_viz.yaml).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import trimesh
import yaml

_DEFAULT_CONFIG = Path(__file__).parent / "sdf_slice_viz.yaml"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_config(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _get(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is None:
            return default
    return node


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------


def load_mesh(mesh_path: str | Path) -> trimesh.Trimesh:
    """Load a mesh from *mesh_path* (PLY / OBJ / STL).

    Args:
        mesh_path: Path to the mesh file.

    Returns:
        A trimesh.Trimesh instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the geometry is not a triangle mesh.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}\nUpdate 'mesh_path' in your sdf_slice_viz.yaml config.")
    geom = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(geom, trimesh.Trimesh):
        raise TypeError(f"Loaded geometry is not a triangle mesh (got {type(geom).__name__}).")
    return geom


# ---------------------------------------------------------------------------
# SDF computation
# ---------------------------------------------------------------------------


def _sphere_sdf(points: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    """Analytical SDF of a sphere.

    Args:
        points: (N, 3) query points.
        centre: (3,) sphere centre.
        radius: Sphere radius.

    Returns:
        (N,) signed distances.
    """
    return np.linalg.norm(points - centre, axis=-1) - radius


def compute_sdf_on_grid(
    mesh: trimesh.Trimesh,
    axis: str,
    position_fraction: float,
    resolution: int,
    source: str,
    sphere_radius_factor: float,
    plane_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    """Sample the SDF on a 2-D axis-aligned grid.

    Args:
        mesh: Source mesh.
        axis: Perpendicular axis ('x', 'y', or 'z').
        position_fraction: Fractional position along axis in [0, 1].
        resolution: Grid points per axis.
        source: 'mesh' for KDTree proximity, 'sphere' for analytic test.
        sphere_radius_factor: Sphere radius as fraction of bounding-sphere radius.
        plane_scale: Factor to expand slice plane beyond mesh bounds.

    Returns:
        (u_coords, v_coords, sdf_grid, extent)

    Raises:
        ValueError: If axis is not 'x', 'y', or 'z'.
    """
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Invalid slice axis '{axis}'. Must be x, y, or z.")

    bounds_min: np.ndarray = mesh.bounds[0]
    bounds_max: np.ndarray = mesh.bounds[1]
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    plane_axes = [i for i in range(3) if i != axis_idx]

    plane_coord = bounds_min[axis_idx] + position_fraction * (bounds_max[axis_idx] - bounds_min[axis_idx])

    u_c = (bounds_min[plane_axes[0]] + bounds_max[plane_axes[0]]) / 2.0
    v_c = (bounds_min[plane_axes[1]] + bounds_max[plane_axes[1]]) / 2.0
    u_half = (bounds_max[plane_axes[0]] - bounds_min[plane_axes[0]]) / 2.0 * plane_scale
    v_half = (bounds_max[plane_axes[1]] - bounds_min[plane_axes[1]]) / 2.0 * plane_scale
    u_min, u_max = u_c - u_half, u_c + u_half
    v_min, v_max = v_c - v_half, v_c + v_half
    u_coords = np.linspace(u_min, u_max, resolution)
    v_coords = np.linspace(v_min, v_max, resolution)
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")

    pts = np.zeros((resolution * resolution, 3))
    pts[:, axis_idx] = plane_coord
    pts[:, plane_axes[0]] = uu.ravel()
    pts[:, plane_axes[1]] = vv.ravel()

    if source == "sphere":
        centre = mesh.centroid
        radius = float(np.linalg.norm(bounds_max - bounds_min) / 2.0 * sphere_radius_factor)
        sdf_values = _sphere_sdf(pts, centre, radius)
    else:
        # Dense surface sampling eliminates Voronoi banding from vertex-only KDTree.
        # trimesh.sample does NOT require rtree.
        from scipy.spatial import cKDTree  # noqa: PLC0415

        surface_pts, face_idx = trimesh.sample.sample_surface(mesh, 200_000)
        face_normals = mesh.face_normals[face_idx]
        tree = cKDTree(surface_pts)
        raw_dist, nearest_idx = tree.query(pts, k=1, workers=-1)
        to_query = pts - surface_pts[nearest_idx]
        sign = np.where(np.sum(to_query * face_normals[nearest_idx], axis=-1) >= 0.0, 1.0, -1.0)
        sdf_values = sign * raw_dist

    sdf_grid = sdf_values.reshape(resolution, resolution)
    extent = (float(u_min), float(u_max), float(v_min), float(v_max))
    return u_coords, v_coords, sdf_grid, extent


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------

_CMAP_MAP: dict[str, str] = {
    "jet": "Jet",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "magma": "Magma",
    "inferno": "Inferno",
    "hot": "Hot",
    "rainbow": "Rainbow",
    "turbo": "Turbo",
    "rdbu": "RdBu",
    "blues": "Blues",
    "reds": "Reds",
}


def _plotly_colorscale(name: str) -> str:
    return _CMAP_MAP.get(name.lower(), name.capitalize())


def _rgb_str(rgba: list[float]) -> str:
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    return f"rgb({r},{g},{b})"


def _rgba_str(rgba: list[float]) -> str:
    r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    a = float(rgba[3]) if len(rgba) > 3 else 1.0
    return f"rgba({r},{g},{b},{a:.3f})"


# ---------------------------------------------------------------------------
# Mesh plane clipping (no shapely dependency)
# ---------------------------------------------------------------------------


def _clip_mesh_at_plane(
    mesh: trimesh.Trimesh,
    axis_idx: int,
    plane_coord: float,
    keep_positive: bool,
) -> trimesh.Trimesh:
    """Clip *mesh* to the half-space on one side of an axis-aligned plane.

    Triangles that straddle the plane are subdivided at the intersection so
    the output has a geometrically clean, straight cut edge.  No external
    dependencies beyond trimesh + numpy.

    Args:
        mesh: Input triangle mesh.
        axis_idx: Index of the axis perpendicular to the cut plane (0/1/2).
        plane_coord: Coordinate value of the plane on that axis.
        keep_positive: If True keep the half where vertex[axis_idx] >= plane_coord.

    Returns:
        A new :class:`trimesh.Trimesh` containing only the kept geometry.
    """
    verts = mesh.vertices
    faces = mesh.faces

    # Signed distances: positive = keep side, negative = discard side
    d: np.ndarray = verts[:, axis_idx] - plane_coord
    if not keep_positive:
        d = -d

    new_verts: list[np.ndarray] = list(verts)
    new_faces: list[list[int]] = []

    for tri in faces:
        ds = d[tri]
        front = ds >= 0.0
        n_front = int(front.sum())

        if n_front == 3:
            new_faces.append(list(tri))
        elif n_front == 0:
            continue
        else:
            # Reorder so front vertices come first
            order = np.argsort(~front)
            tri_r = tri[order]
            ds_r = ds[order]

            if n_front == 1:
                # 1 front vertex (tri_r[0]), 2 back — one output triangle
                p01 = _lerp(new_verts[tri_r[0]], new_verts[tri_r[1]], ds_r[0], ds_r[1])
                p02 = _lerp(new_verts[tri_r[0]], new_verts[tri_r[2]], ds_r[0], ds_r[2])
                i01 = len(new_verts)
                new_verts.append(p01)
                i02 = len(new_verts)
                new_verts.append(p02)
                new_faces.append([tri_r[0], i01, i02])
            else:
                # 2 front vertices (tri_r[0], tri_r[1]), 1 back — two output triangles
                p02 = _lerp(new_verts[tri_r[0]], new_verts[tri_r[2]], ds_r[0], ds_r[2])
                p12 = _lerp(new_verts[tri_r[1]], new_verts[tri_r[2]], ds_r[1], ds_r[2])
                i02 = len(new_verts)
                new_verts.append(p02)
                i12 = len(new_verts)
                new_verts.append(p12)
                new_faces.append([tri_r[0], tri_r[1], i02])
                new_faces.append([tri_r[1], i12, i02])

    if not new_faces:
        # Nothing left after clipping — return empty mesh as fallback
        return trimesh.Trimesh(vertices=verts[:1], faces=np.empty((0, 3), dtype=int), process=False)

    return trimesh.Trimesh(
        vertices=np.array(new_verts),
        faces=np.array(new_faces, dtype=int),
        process=False,
    )


def _lerp(p0: np.ndarray, p1: np.ndarray, d0: float, d1: float) -> np.ndarray:
    """Linear interpolation along an edge to the point where d = 0."""
    t = d0 / (d0 - d1)
    return p0 + t * (p1 - p0)


def _mesh_shadow_mask(
    vis_mesh: trimesh.Trimesh,
    axis_idx: int,
    u_coords: np.ndarray,
    v_coords: np.ndarray,
) -> np.ndarray:
    """Return a bool grid where True = this SDF cell is occluded by the mesh.

    Projects each mesh face onto the UV plane (dropping the slice axis) and
    rasterises with edge-function tests.  AABB pre-culling keeps this fast
    even for 10k+ faces.

    Args:
        vis_mesh: Clipped mesh (front half only).
        axis_idx: Slice-axis index (0/1/2).
        u_coords: 1-D u-axis sample positions (length = resolution).
        v_coords: 1-D v-axis sample positions (length = resolution).

    Returns:
        Bool array of shape ``(len(u_coords), len(v_coords))``.
    """
    plane_axes = [i for i in range(3) if i != axis_idx]
    u_ax, v_ax = plane_axes[0], plane_axes[1]

    proj = vis_mesh.vertices[:, [u_ax, v_ax]]  # (N, 2) projected verts
    faces = vis_mesh.faces

    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")
    pts_u = uu.ravel()
    pts_v = vv.ravel()
    covered = np.zeros(len(pts_u), dtype=bool)

    for f in faces:
        a, b, c = proj[f[0]], proj[f[1]], proj[f[2]]

        # Fast AABB rejection
        u_lo = min(a[0], b[0], c[0])
        u_hi = max(a[0], b[0], c[0])
        v_lo = min(a[1], b[1], c[1])
        v_hi = max(a[1], b[1], c[1])
        idx = np.where((pts_u >= u_lo) & (pts_u <= u_hi) & (pts_v >= v_lo) & (pts_v <= v_hi))[0]
        if len(idx) == 0:
            continue

        pu, pv = pts_u[idx], pts_v[idx]
        # Signed edge functions (correct for both winding orders)
        e0 = (b[0] - a[0]) * (pv - a[1]) - (b[1] - a[1]) * (pu - a[0])
        e1 = (c[0] - b[0]) * (pv - b[1]) - (c[1] - b[1]) * (pu - b[0])
        e2 = (a[0] - c[0]) * (pv - c[1]) - (a[1] - c[1]) * (pu - c[0])
        inside = ((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        covered[idx] |= inside

    return covered.reshape(uu.shape)


def _build_lighting(cfg: dict[str, Any]) -> dict[str, float]:
    """Return a Plotly lighting dict from the ``lighting:`` config section.

    When ``lighting.shading`` is false (default), returns flat lighting so
    the mesh color is unaffected by light direction.  When true, applies
    the same diffuse/specular model as mesh_viz.
    """
    if not _get(cfg, "lighting", "shading", default=False):
        return {"ambient": 1.0, "diffuse": 0.0, "specular": 0.0}
    return {
        "ambient": float(_get(cfg, "lighting", "ambient", default=0.4)),
        "diffuse": float(_get(cfg, "lighting", "diffuse", default=0.8)),
        "specular": float(_get(cfg, "lighting", "specular", default=0.3)),
        "roughness": float(_get(cfg, "lighting", "roughness", default=0.5)),
        "fresnel": float(_get(cfg, "lighting", "fresnel", default=0.2)),
    }


def _build_lightposition(cfg: dict[str, Any]) -> dict[str, float]:
    """Return a Plotly lightposition dict from the ``lighting:`` config section."""
    return {
        "x": float(_get(cfg, "lighting", "position_x", default=1.0)),
        "y": float(_get(cfg, "lighting", "position_y", default=1.0)),
        "z": float(_get(cfg, "lighting", "position_z", default=2.0)),
    }


def _build_wireframe_trace(
    mesh: trimesh.Trimesh,
    edge_color: list[float],
) -> go.Scatter3d | None:
    """Build a Scatter3d line trace for mesh wireframe edges.

    Args:
        mesh: Triangle mesh.
        edge_color: RGBA list in [0, 1].

    Returns:
        A go.Scatter3d trace, or None if alpha is zero.
    """
    if len(edge_color) > 3 and edge_color[3] < 0.01:
        return None

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

    verts = mesh.vertices
    n = len(edges)
    sx = np.empty(n * 3)
    sy = np.empty(n * 3)
    sz = np.empty(n * 3)
    sx[0::3] = verts[edges[:, 0], 0]
    sx[1::3] = verts[edges[:, 1], 0]
    sx[2::3] = np.nan
    sy[0::3] = verts[edges[:, 0], 1]
    sy[1::3] = verts[edges[:, 1], 1]
    sy[2::3] = np.nan
    sz[0::3] = verts[edges[:, 0], 2]
    sz[1::3] = verts[edges[:, 1], 2]
    sz[2::3] = np.nan

    return go.Scatter3d(
        x=sx,
        y=sy,
        z=sz,
        mode="lines",
        line={"color": _rgba_str(edge_color), "width": 1},
        hoverinfo="skip",
    )


# ---------------------------------------------------------------------------
# SDF isoline traces
# ---------------------------------------------------------------------------


def _build_isoline_traces(
    u_coords: np.ndarray,
    v_coords: np.ndarray,
    sdf_grid: np.ndarray,
    axis_idx: int,
    plane_coord: float,
    plane_axes: list[int],
    cfg: dict[str, Any],
) -> list[go.Scatter3d]:
    """Build Scatter3d line traces for SDF isolines on the slice plane.

    Extracts contour paths in UV space using matplotlib (Agg backend, no
    display required), then lifts them back to 3-D on the slice plane.

    Args:
        u_coords: 1-D u-axis sample positions (length = Nu).
        v_coords: 1-D v-axis sample positions (length = Nv).
        sdf_grid: (Nu, Nv) SDF values.
        axis_idx: Index of the axis perpendicular to the slice (0/1/2).
        plane_coord: Fixed coordinate on that axis.
        plane_axes: Indices of the two axes spanning the slice plane.
        cfg: Full config dict.

    Returns:
        List of go.Scatter3d traces — one segment per contour path.
    """
    if not _get(cfg, "isolines", "show", default=True):
        return []

    exterior = sdf_grid >= 0
    if not exterior.any():
        return []

    n_levels = int(_get(cfg, "isolines", "count", default=8))
    iso_color = list(_get(cfg, "isolines", "color") or [0.0, 0.0, 0.0, 0.6])
    iso_width = float(_get(cfg, "isolines", "width", default=1.5))
    levels_cfg = _get(cfg, "isolines", "levels")

    if levels_cfg is not None:
        levels: list[float] = [float(v) for v in levels_cfg]
    else:
        sdf_ext = sdf_grid[exterior]
        lo, hi = float(sdf_ext.min()), float(sdf_ext.max())
        if lo >= hi:
            return []
        levels = list(np.linspace(lo, hi, n_levels + 2)[1:-1])

    import matplotlib  # noqa: PLC0415  (lazy import — matplotlib not in core deps)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # contour expects Z of shape (Ny, Nx), i.e. (Nv, Nu) since grid is (Nu, Nv) with ij
    fig_mpl, ax = plt.subplots()
    cs = ax.contour(u_coords, v_coords, sdf_grid.T, levels=levels)
    plt.close(fig_mpl)

    iso_color_str = _rgba_str(iso_color)
    traces: list[go.Scatter3d] = []

    # allsegs: list[list[ndarray(N,2)]] — one outer list entry per level
    for level_segs in cs.allsegs:
        for seg in level_segs:
            if len(seg) < 2:
                continue
            xyz = np.zeros((len(seg), 3))
            xyz[:, axis_idx] = plane_coord
            xyz[:, plane_axes[0]] = seg[:, 0]
            xyz[:, plane_axes[1]] = seg[:, 1]
            traces.append(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="lines",
                    line={"color": iso_color_str, "width": iso_width},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    return traces


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------


def build_figure(mesh: trimesh.Trimesh, cfg: dict[str, Any]) -> go.Figure:  # noqa: PLR0915  # figure builder; statement count reflects multiple trace types and layout options
    """Build the Plotly figure: SDF surface + mesh solid + wireframe edges.

    Args:
        mesh: Loaded triangle mesh.
        cfg: Full config dict loaded from YAML.

    Returns:
        A plotly.graph_objects.Figure.
    """
    axis = str(_get(cfg, "slice", "axis", default="z")).lower()
    position = float(_get(cfg, "slice", "position", default=0.5))
    resolution = int(_get(cfg, "slice", "resolution", default=200))
    cmap_name = str(_get(cfg, "slice", "colormap", default="jet"))
    invert_cmap = bool(_get(cfg, "slice", "colormap_invert", default=False))
    alpha = float(_get(cfg, "slice", "alpha", default=0.85))
    vmin_cfg = _get(cfg, "slice", "vmin")
    vmax_cfg = _get(cfg, "slice", "vmax")
    source = str(_get(cfg, "sdf", "source", default="mesh"))
    sphere_factor = float(_get(cfg, "sdf", "sphere_radius_factor", default=0.35))
    plane_scale = float(_get(cfg, "slice", "plane_scale", default=1.0))

    face_color = list(_get(cfg, "mesh", "face_color") or [0.55, 0.75, 0.95, 1.0])
    edge_color = list(_get(cfg, "mesh", "edge_color") or [0.2, 0.2, 0.2, 0.25])
    bg_color = list(_get(cfg, "background_color") or [1.0, 1.0, 1.0, 0.0])

    w_px = int(_get(cfg, "image_width", default=1600))
    h_px = int(_get(cfg, "image_height", default=800))

    # Convert matplotlib azimuth/elevation to Plotly camera eye vector
    azim = float(_get(cfg, "camera", "azimuth", default=-15))
    elev = float(_get(cfg, "camera", "elevation", default=20))
    dist = float(_get(cfg, "camera", "distance_factor", default=2.5))
    azim_rad = math.radians(azim)
    elev_rad = math.radians(elev)
    eye_x = dist * math.cos(elev_rad) * math.cos(azim_rad)
    eye_y = dist * math.cos(elev_rad) * math.sin(azim_rad)
    eye_z = dist * math.sin(elev_rad)

    u_coords, v_coords, sdf_grid, _ = compute_sdf_on_grid(
        mesh=mesh,
        axis=axis,
        position_fraction=position,
        resolution=resolution,
        source=source,
        sphere_radius_factor=sphere_factor,
        plane_scale=plane_scale,
    )

    exterior = sdf_grid >= 0
    vmin = float(vmin_cfg) if vmin_cfg is not None else (float(sdf_grid[exterior].min()) if exterior.any() else 0.0)
    vmax = float(vmax_cfg) if vmax_cfg is not None else float(sdf_grid.max())

    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    plane_axes = [i for i in range(3) if i != axis_idx]
    b_min, b_max = mesh.bounds
    plane_coord = b_min[axis_idx] + position * (b_max[axis_idx] - b_min[axis_idx])

    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")
    coords = np.zeros((*uu.shape, 3))
    coords[:, :, axis_idx] = plane_coord
    coords[:, :, plane_axes[0]] = uu
    coords[:, :, plane_axes[1]] = vv
    X3, Y3, Z3 = coords[..., 0], coords[..., 1], coords[..., 2]

    # NaN-mask all three coordinates at interior vertices so Plotly skips
    # those cells entirely — preserves correct depth sorting.
    interior = sdf_grid < 0
    X3_plot = np.where(interior, np.nan, X3)
    Y3_plot = np.where(interior, np.nan, Y3)
    Z3_plot = np.where(interior, np.nan, Z3)
    sdf_masked = np.where(exterior, sdf_grid, np.nan)

    # Always render the full mesh so the back (tail) is visible beyond the plane.
    vis_mesh = mesh
    # The SDF surface is already masked at sdf_grid < 0 (interior of mesh).
    # Plotly's WebGL depth buffer occludes the SDF behind opaque mesh faces
    # automatically, so no pixel-raster shadow mask is needed.

    colorscale_name = _plotly_colorscale(cmap_name)
    if invert_cmap:
        colorscale_name = colorscale_name + "_r"

    sdf_trace = go.Surface(
        x=X3_plot,
        y=Y3_plot,
        z=Z3_plot,
        surfacecolor=sdf_masked,
        colorscale=colorscale_name,
        cmin=vmin,
        cmax=vmax,
        opacity=alpha,
        showscale=False,
        hoverinfo="skip",
        lighting={"ambient": 1.0, "diffuse": 0.0, "specular": 0.0},
        contours={
            "x": {"highlight": False},
            "y": {"highlight": False},
            "z": {"highlight": False},
        },
    )

    mesh_solid = go.Mesh3d(
        x=vis_mesh.vertices[:, 0],
        y=vis_mesh.vertices[:, 1],
        z=vis_mesh.vertices[:, 2],
        i=vis_mesh.faces[:, 0],
        j=vis_mesh.faces[:, 1],
        k=vis_mesh.faces[:, 2],
        color=_rgb_str(face_color),
        opacity=1.0,
        flatshading=False,
        lighting=_build_lighting(cfg),
        lightposition=_build_lightposition(cfg),
        hoverinfo="skip",
    )

    wire_trace = _build_wireframe_trace(vis_mesh, edge_color)

    isoline_traces = _build_isoline_traces(
        u_coords=u_coords,
        v_coords=v_coords,
        sdf_grid=sdf_grid,
        axis_idx=axis_idx,
        plane_coord=plane_coord,
        plane_axes=plane_axes,
        cfg=cfg,
    )

    traces: list[go.BaseTraceType] = [sdf_trace, mesh_solid]
    if wire_trace is not None:
        traces.append(wire_trace)
    traces.extend(isoline_traces)

    bg_str = _rgba_str(bg_color)

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=w_px,
        height=h_px,
        paper_bgcolor=bg_str,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "camera": {
                "eye": {"x": eye_x, "y": eye_y, "z": eye_z},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "xaxis": {"visible": False, "showgrid": False, "showline": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "showline": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "showline": False, "zeroline": False},
            "bgcolor": "rgba(0,0,0,0)",
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
        description="Mesh + analytical SDF slice visualisation (Plotly backend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Write an interactive HTML file instead of a PNG",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = _parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    cfg = _load_config(args.config)
    if args.mesh is not None:
        cfg["mesh_path"] = str(args.mesh)
    if args.out is not None:
        cfg["output_path"] = str(args.out)

    mesh_path = _get(cfg, "mesh_path")
    if mesh_path is None:
        print("No mesh_path specified.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading mesh: {mesh_path}")
    mesh = load_mesh(mesh_path)
    print(f"  Vertices: {len(mesh.vertices):,}   Faces: {len(mesh.faces):,}   Bounds: {mesh.bounds[0]} → {mesh.bounds[1]}")

    print("Building figure …")
    fig = build_figure(mesh, cfg)

    out_path = Path(str(_get(cfg, "output_path", default="sdf_slice_viz.png")))
    mesh_stem = Path(mesh_path).stem
    out_path = out_path.with_stem(f"{out_path.stem}_{mesh_stem}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        html_path = out_path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"Saved (HTML) → {html_path}")
    else:
        w_px = int(_get(cfg, "image_width", default=1600))
        h_px = int(_get(cfg, "image_height", default=800))
        render_scale = float(_get(cfg, "render_scale", default=1))
        fig.write_image(str(out_path), width=w_px, height=h_px, scale=render_scale)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
