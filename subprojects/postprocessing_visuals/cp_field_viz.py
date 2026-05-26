#!/usr/bin/env python3
"""Predict and visualize the Cp pressure field on a surface mesh using GAOT3.

Loads a PLY, runs the same GAOT3 model + preprocessing path used by WP4's
GaotLoss (same checkpoint lookup, same batch format, same rescaling), and
produces a Cp-coloured 3-D Plotly figure.

Preprocessing mirrors ModelLoader._reformat_data exactly, skipping only
the SDF-model diagnostic (normal-flip counter / wandb log) that is not
available outside of the optimizer loop.

Usage
-----
    python scripts/cp_field_viz.py                     # uses default config
    python scripts/cp_field_viz.py --mesh /path/to.ply
    python scripts/cp_field_viz.py --interactive       # write HTML instead
    python scripts/cp_field_viz.py --config /path/to/cfg.yaml
    python scripts/cp_field_viz.py --out /path/to/out.png
"""

from __future__ import annotations

import argparse
import ast
import copy
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn.functional as F  # noqa: N812  # F is the universal PyTorch convention for this module
import trimesh
import yaml
from torch_geometric.data import Batch, Data

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.metadata import DATASET_METADATA
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.dataloader.pyg_transforms import (
    RescalePosition,
    RescalePositionNew,
)

# GAOT3 internals — same imports as ModelLoader3, no wandb dependency
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.model.gaot_3d import GAOT3D
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.utils.scale import rescale, rescale_new

_DEFAULT_CONFIG = Path(__file__).parent / "cp_field_viz.yaml"


# ---------------------------------------------------------------------------
# Config helpers  (same pattern as sdf_slice_viz.py)
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
    """Load and validate a triangle mesh from *mesh_path* (PLY / OBJ / STL).

    Args:
        mesh_path: Path to the mesh file.

    Returns:
        A ``trimesh.Trimesh`` instance.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the geometry is not a triangle mesh.
    """
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}\nSet 'mesh_path' in your cp_field_viz.yaml config.")
    geom = trimesh.load(str(mesh_path), force="mesh")
    if not isinstance(geom, trimesh.Trimesh):
        raise TypeError(f"Loaded geometry is not a triangle mesh (got {type(geom).__name__}).")
    return geom


# ---------------------------------------------------------------------------
# GAOT3 inference  (mirrors ModelLoader3 without SDF/extractor dependency)
# ---------------------------------------------------------------------------

_METRICS_CSV_PATHS = [
    "/cluster/work/fuge/AISCO/wp1_cfd_ai/results/metrics.csv",
    "/home/aisco/results/metrics.csv",
]


class GAOTPlyInference:
    """GAOT3 inference runner for direct PLY meshes.

    Replicates the checkpoint-loading and batch-preprocessing logic from
    ``GAOT3/model_loader.ModelLoader``.  The SDF model and MeshSDFExtractor
    are not needed for inference — we skip only the diagnostic normal-flip
    counter that calls the SDF model inside ``_reformat_data``.
    """

    def __init__(self, cfd_config: dict[str, Any], device: str) -> None:
        self.cfd_config = cfd_config
        self.device = device
        self.model_path = self._get_checkpoint_path()
        self._load_model()

    # ------------------------------------------------------------------
    # Checkpoint / model loading  (mirrors ModelLoader3._get_checkpoint_path
    # and ModelLoader3._load_model verbatim)
    # ------------------------------------------------------------------

    def _get_checkpoint_path(self) -> str:
        df: pd.DataFrame | None = None
        for csv_path in _METRICS_CSV_PATHS:
            if Path(csv_path).exists():
                df = pd.read_csv(csv_path)
                break
        if df is None:
            raise FileNotFoundError("Could not find metrics.csv at any of:\n" + "\n".join(f"  {p}" for p in _METRICS_CSV_PATHS))
        for _, row in df.iterrows():
            if not isinstance(row["setup"], str):
                continue
            setup_dict = ast.literal_eval(row["setup"])
            if not isinstance(setup_dict, dict):
                continue
            if str(setup_dict.get("id")) == str(self.cfd_config["model_id"]):
                if not isinstance(row["path"], str):
                    continue
                path_dict = ast.literal_eval(row["path"])
                return path_dict.get("ckpt_path")
        raise ValueError(f"model_id={self.cfd_config['model_id']!r} not found in metrics.csv.")

    def _load_model(self) -> None:
        config_path = self.model_path.replace("model.pt", "config.yaml")
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Model config not found at {config_path}. Expected config.yaml next to model.pt.")

        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        self._og_config = copy.deepcopy(config)
        model_args = config["model"]["args"]
        model_args["magno"]["precompute_edges"] = False
        model_args["transformer"]["norm_eps"] = float(model_args["transformer"].get("norm_eps"))

        self.gaot = GAOT3D(config=model_args).to(self.device)
        self.gaot.eval()

        state = torch.load(self.model_path, map_location=self.device, weights_only=False)
        sd = state.get("model") or state.get("state_dict") or state
        sd = {k.removeprefix("module."): v for k, v in sd.items()}

        missing, unexpected = self.gaot.load_state_dict(sd, strict=False)
        if missing or unexpected:
            warnings.warn(
                f"Checkpoint mismatch. Missing: {missing}. Unexpected: {unexpected}.",
                stacklevel=2,
            )

        # Reconstruct latent tokens (same logic as ModelLoader3._load_model)
        dataset_cfg = self._og_config.get("dataset", {})
        self._use_rescale_new: bool = dataset_cfg.get("use_rescale_new", False)

        metaname = dataset_cfg.get("metaname")
        if metaname and metaname in DATASET_METADATA:
            self._phy_domain = DATASET_METADATA[metaname].domain_x
        else:
            norm_domain = model_args.get("norm_domain")
            if norm_domain is None:
                raise KeyError(
                    "Could not determine physical domain. Expected dataset.metaname "
                    "in DATASET_METADATA or model.args.norm_domain in saved config."
                )
            self._phy_domain = norm_domain

        latent_token_size = model_args.get("latent_tokens")
        if latent_token_size is None:
            raise KeyError("Expected model.args.latent_tokens in saved config.")

        x_min, y_min, z_min = self._phy_domain[0]
        x_max, y_max, z_max = self._phy_domain[1]
        meshgrid = torch.meshgrid(
            torch.linspace(x_min, x_max, latent_token_size[0]),
            torch.linspace(y_min, y_max, latent_token_size[1]),
            torch.linspace(z_min, z_max, latent_token_size[2]),
            indexing="ij",
        )
        latent_queries = torch.stack(meshgrid, dim=-1).reshape(-1, 3)

        if self._use_rescale_new:
            self._latent_tokens = rescale_new(latent_queries, (-1, 1), self._phy_domain).to(self.device)
        else:
            self._latent_tokens = rescale(latent_queries, (-1, 1)).to(self.device)

    # ------------------------------------------------------------------
    # Preprocessing  (mirrors ModelLoader3._reformat_data, SDF-free)
    # ------------------------------------------------------------------

    def _preprocess(self, verts_mm: torch.Tensor, faces: torch.Tensor) -> Batch:
        """Build a PyG Batch from raw PLY vertices (mm) and faces.

        Mirrors ``ModelLoader._reformat_data`` exactly, omitting only the
        SDF-model diagnostic pass (normal-flip counter / wandb.log).

        Args:
            verts_mm: (V, 3) float32 vertex positions in millimetres.
            faces: (F, 3) long face indices.

        Returns:
            A single-graph ``Batch`` ready for ``GAOT3D.forward``.
        """
        verts = verts_mm.to(self.device)
        faces = faces.to(self.device)

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]

        # Per-face unit normals (cross product of two edges)
        raw_face_n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_normals = F.normalize(raw_face_n, dim=-1)

        # Accumulate face normals to vertices (equal-weight, then normalize)
        num_verts = verts.shape[0]
        all_idx = faces.reshape(-1)  # (3·F,)
        all_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        vertex_normals = torch.zeros(num_verts, 3, device=self.device, dtype=verts.dtype)
        vertex_normals = vertex_normals.index_add(0, all_idx, all_normals)
        vertex_normals = F.normalize(vertex_normals, dim=-1)

        raw_vertex_normals = vertex_normals.clone()
        model_vertex_normals = vertex_normals.clone()
        model_vertex_normals[:, 0] *= -1.0  # x-flip (training convention)

        # mm → m  +  x-flip (training convention)
        model_verts_m = verts.clone() * 1e-3
        model_verts_m[:, 0] *= -1.0

        # Stash for post-inference smoothing (raw normals, original face indices)
        self._last_raw_normals = raw_vertex_normals.cpu().numpy().astype(np.float32)
        self._last_faces = faces.cpu().numpy().astype(np.int32)

        data = Data(
            pos=model_verts_m,
            face=faces.T,  # (3, F) — PyG convention
            c=model_vertex_normals,
            c_raw=raw_vertex_normals,
        )

        if self._use_rescale_new:
            data = RescalePositionNew(lims=(-1.0, 1.0), phy_domain=self._phy_domain)(data)
        else:
            data = RescalePosition(lims=(-1.0, 1.0))(data)

        return Batch.from_data_list([data]).to(self.device)

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def infer_from_ply(
        self,
        ply_path: str | Path,
        smoothing_cfg: dict[str, Any] | None = None,
        *,
        flip_about_x_axis: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run GAOT3 inference on a PLY file.

        Coordinate convention (PLY space):
          - X-axis: streamwise, **positive = forward / nose direction**
          - Y-axis: lateral (symmetric, +Y = port)
          - Z-axis: vertical, +Z = up
        The x-flip applied inside ``_preprocess`` is a model-internal transform;
        the returned *verts_mm* and *cp* are always in the (optionally flipped) PLY space.

        Args:
            ply_path:           Path to a PLY mesh file (coordinates in mm).
            smoothing_cfg:      Optional dict mirroring the ``cp_smoothing`` section
                                of the training config.
                flip_about_x_axis:  When True, negate Y and Z before inference
                                (180° rotation around the X axis).  Unused;
                                kept for API compatibility.

        Returns:
            ``(verts_mm, faces, cp)`` — numpy arrays for visualization.
            *verts_mm*: (V, 3) vertex positions in mm (post-flip if enabled).
            *faces*:    (F, 3) face indices.
            *cp*:       (V,) predicted (and optionally smoothed) Cp values.
        """
        mesh = load_mesh(ply_path)
        verts_np = mesh.vertices.astype(np.float64).copy()
        if flip_about_x_axis:
            # 180° rotation around Z: negate X and Y
            verts_np[:, 0] *= -1.0
            verts_np[:, 1] *= -1.0
        verts_t = torch.tensor(verts_np, dtype=torch.float32)
        faces_t = torch.tensor(mesh.faces, dtype=torch.long)

        batch = self._preprocess(verts_t, faces_t)

        with torch.no_grad():
            output = self.gaot.forward(batch=batch, tokens_pos=self._latent_tokens)

        cp = output["field"].squeeze(-1).cpu().numpy().astype(np.float32)

        if smoothing_cfg and bool(smoothing_cfg.get("enabled", False)):
            cp = smooth_cp_field(
                cp=cp,
                faces=self._last_faces,
                vertex_normals=self._last_raw_normals,
                iterations=int(smoothing_cfg.get("iterations", 50)),
                min_strength=float(smoothing_cfg.get("min_strength", 1.0)),
                max_strength=float(smoothing_cfg.get("max_strength", 1.0)),
                curvature_power=float(smoothing_cfg.get("curvature_power", 0.0)),
                curvature_quantile=float(smoothing_cfg.get("curvature_quantile", 0.95)),
            )

        return verts_np, mesh.faces.astype(np.int32), cp


# ---------------------------------------------------------------------------
# Cp field post-processing
# ---------------------------------------------------------------------------


def smooth_cp_field(
    cp: np.ndarray,
    faces: np.ndarray,
    vertex_normals: np.ndarray | None = None,
    *,
    iterations: int = 50,
    min_strength: float = 1.0,
    max_strength: float = 1.0,
    curvature_power: float = 0.0,
    curvature_quantile: float = 0.95,
) -> np.ndarray:
    """Curvature-aware Laplacian Cp smoothing.

    Mirrors ``ModelLoader._smooth_cp_field_if_enabled`` exactly, without the
    drag-preservation step (not needed for visualization).

    With the training defaults (min_strength=max_strength=1.0, curvature_power=0)
    every vertex is fully replaced by its neighbour mean on each iteration —
    equivalent to standard iterated Laplacian smoothing.

    Args:
        cp:               (V,) raw per-vertex Cp values.
        faces:            (F, 3) face indices.
        vertex_normals:   (V, 3) unit normals used for curvature weighting.
                          Only matters when ``curvature_power > 0``.  If None,
                          or when curvature_power==0, normals are not used.
        iterations:       Number of smoothing passes.
        min_strength:     Lower bound of per-vertex smoothing strength [0, 1].
        max_strength:     Upper bound of per-vertex smoothing strength [0, 1].
        curvature_power:  Exponent that maps normalised curvature → strength.
                          0 = uniform (ignore curvature), 2 = sharply peaked.
        curvature_quantile: Quantile used to normalise the curvature values.

    Returns:
        (V,) smoothed Cp array (same dtype as input).
    """
    if iterations <= 0 or len(faces) == 0:
        return cp

    device = torch.device("cpu")
    dtype = torch.float32
    p = torch.tensor(cp, dtype=dtype, device=device)
    f = torch.tensor(faces, dtype=torch.long, device=device)

    # Build bidirectional edge lists (src→dst and dst→src for all face edges)
    src = torch.cat([f[:, 0], f[:, 1], f[:, 1], f[:, 2], f[:, 2], f[:, 0]], dim=0)
    dst = torch.cat([f[:, 1], f[:, 0], f[:, 2], f[:, 1], f[:, 0], f[:, 2]], dim=0)
    num_verts = p.shape[0]
    degree = torch.zeros(num_verts, dtype=dtype, device=device)
    degree = degree.index_add(0, dst, torch.ones_like(src, dtype=dtype)).clamp_min(1.0)

    # Compute per-vertex smoothing strength from edge curvature
    if curvature_power > 0.0 and vertex_normals is not None:
        normals = F.normalize(torch.tensor(vertex_normals, dtype=dtype, device=device), dim=-1)
        edge_curv = (1.0 - (normals[src] * normals[dst]).sum(dim=-1)).clamp_min(0.0)
        curv = torch.zeros(num_verts, dtype=dtype, device=device)
        curv = curv.index_add(0, dst, edge_curv) / degree
        curv_scale = torch.quantile(curv, curvature_quantile).clamp_min(torch.finfo(dtype).eps)
        curv_norm = (curv / curv_scale).clamp(0.0, 1.0)
        strength = min_strength + (max_strength - min_strength) * (1.0 - curv_norm).pow(curvature_power)
    else:
        # Uniform strength (training default: power=0 → all weights = max_strength)
        strength = torch.full((num_verts,), fill_value=max_strength, dtype=dtype, device=device)

    p_smooth = p.clone()
    for _ in range(iterations):
        nbr_sum = torch.zeros(num_verts, dtype=dtype, device=device)
        nbr_sum = nbr_sum.index_add(0, dst, p_smooth[src])
        nbr_mean = nbr_sum / degree
        p_smooth = p_smooth + strength * (nbr_mean - p_smooth)

    return p_smooth.numpy().astype(cp.dtype)


# ---------------------------------------------------------------------------
# Plotly helpers  (same pattern as sdf_slice_viz.py)
# ---------------------------------------------------------------------------

# Plotly colorscale names.  matplotlib RdBu_r ≈ Plotly RdBu (blue=low, red=high).
_CMAP_MAP: dict[str, str] = {
    "rdbu": "RdBu",
    "rdbu_r": "RdBu",  # matplotlib RdBu_r == Plotly RdBu direction
    "bwr": "RdBu",
    "jet": "Jet",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "magma": "Magma",
    "inferno": "Inferno",
    "hot": "Hot",
    "turbo": "Turbo",
    "reds": "Reds",
    "blues": "Blues",
    "rainbow": "Rainbow",
    "coolwarm": "RdBu",
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


def _build_wireframe_trace(
    verts: np.ndarray,
    faces: np.ndarray,
    edge_color: list[float],
) -> go.Scatter3d | None:
    if len(edge_color) > 3 and edge_color[3] < 0.01:
        return None
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


# 6 axis-aligned views.  azimuth=0 → eye at +X ("pure front" per sdf_slice_viz convention).
_CARDINAL_VIEWS: dict[str, tuple[float, float]] = {
    "front": (0.0, 0.0),  # eye at +X, looking in -X
    "rear": (180.0, 0.0),  # eye at -X, looking in +X
    "left": (90.0, 0.0),  # eye at +Y, looking in -Y (port)
    "right": (-90.0, 0.0),  # eye at -Y, looking in +Y (starboard)
    "top": (0.0, 89.0),  # eye above, looking down  (89° avoids gimbal lock)
    "bottom": (0.0, -89.0),  # eye below, looking up
}


# ---------------------------------------------------------------------------
# Lighting helpers
# ---------------------------------------------------------------------------


def _build_lighting(cfg: dict[str, Any]) -> dict[str, float]:
    """Return a Plotly lighting dict based on config.

    When ``lighting.shading`` is false (or the section is absent), returns
    flat lighting (ambient=1, diffuse=0) so the Cp colormap is unaffected.
    When true, applies the same parameters as mesh_viz for surface shading.
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
    return {
        "x": float(_get(cfg, "lighting", "position_x", default=1.0)),
        "y": float(_get(cfg, "lighting", "position_y", default=1.0)),
        "z": float(_get(cfg, "lighting", "position_z", default=2.0)),
    }


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------


def build_figure(
    verts_mm: np.ndarray,
    faces: np.ndarray,
    cp: np.ndarray,
    cfg: dict[str, Any],
    *,
    azimuth_override: float | None = None,
    elevation_override: float | None = None,
) -> go.Figure:
    """Build a Plotly figure: mesh coloured by Cp.

    Args:
        verts_mm:           (V, 3) vertex positions in mm.
        faces:              (F, 3) face indices.
        cp:                 (V,) per-vertex Cp values.
        cfg:                Full config dict loaded from YAML.
        azimuth_override:   Override camera azimuth (degrees). If None, use config.
        elevation_override: Override camera elevation (degrees). If None, use config.

    Returns:
        A ``plotly.graph_objects.Figure``.
    """
    # Colormap / range
    cmap_name = str(_get(cfg, "cp_colormap", default="RdBu_r"))
    vmin_cfg = _get(cfg, "cp_vmin")
    vmax_cfg = _get(cfg, "cp_vmax")
    pct_clip = _get(cfg, "cp_percentile_clip")  # e.g. [0.5, 99.5]
    alpha = float(_get(cfg, "cp_alpha", default=1.0))

    finite = cp[np.isfinite(cp)]
    if vmin_cfg is not None:
        vmin = float(vmin_cfg)
    elif pct_clip is not None and len(finite):
        vmin = float(np.percentile(finite, float(pct_clip[0])))
    else:
        vmin = float(finite.min()) if len(finite) else -2.0

    if vmax_cfg is not None:
        vmax = float(vmax_cfg)
    elif pct_clip is not None and len(finite):
        vmax = float(np.percentile(finite, float(pct_clip[1])))
    else:
        vmax = float(finite.max()) if len(finite) else 1.0

    # Optional X-axis flip (negates X coordinates — reverses forward direction in view)
    if bool(_get(cfg, "flip_x_axis", default=False)):
        verts_mm = verts_mm.copy()
        verts_mm[:, 0] *= -1.0

    # Edge overlay
    edge_color = list(_get(cfg, "mesh", "edge_color") or [0.2, 0.2, 0.2, 0.0])

    # Camera — accept per-call overrides (used for cardinal views)
    azim = float(azimuth_override if azimuth_override is not None else _get(cfg, "camera", "azimuth", default=-15))
    elev = float(elevation_override if elevation_override is not None else _get(cfg, "camera", "elevation", default=20))
    dist = float(_get(cfg, "camera", "distance_factor", default=2.5))
    azim_rad, elev_rad = math.radians(azim), math.radians(elev)
    eye_x = dist * math.cos(elev_rad) * math.cos(azim_rad)
    eye_y = dist * math.cos(elev_rad) * math.sin(azim_rad)
    eye_z = dist * math.sin(elev_rad)

    w_px = int(_get(cfg, "image_width", default=1600))
    h_px = int(_get(cfg, "image_height", default=800))

    # Cp-coloured surface mesh
    cp_trace = go.Mesh3d(
        x=verts_mm[:, 0],
        y=verts_mm[:, 1],
        z=verts_mm[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        intensity=cp,
        intensitymode="vertex",
        colorscale=_plotly_colorscale(cmap_name),
        cmin=vmin,
        cmax=vmax,
        opacity=alpha,
        showscale=False,
        lighting=_build_lighting(cfg),
        lightposition=_build_lightposition(cfg),
        hoverinfo="skip",
    )

    traces: list[go.BaseTraceType] = [cp_trace]
    wire_trace = _build_wireframe_trace(verts_mm, faces, edge_color)
    if wire_trace is not None:
        traces.append(wire_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        width=w_px,
        height=h_px,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        scene={
            "camera": {
                "eye": {"x": eye_x, "y": eye_y, "z": eye_z},
                "up": {"x": 0, "y": 0, "z": 1},
            },
            "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
            "zaxis": {"visible": False, "showgrid": False, "zeroline": False},
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
        description="GAOT3 Cp pressure field visualisation",
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
        print("No mesh_path specified in config.", file=sys.stderr)
        sys.exit(1)

    cfd_config = _get(cfg, "cfd")
    if cfd_config is None:
        print("No 'cfd:' section found in config.", file=sys.stderr)
        sys.exit(1)

    device = str(_get(cfg, "device", default="cuda" if torch.cuda.is_available() else "cpu"))

    # Load model
    print(f"Loading GAOT3 model (id={cfd_config['model_id']}) …")
    runner = GAOTPlyInference(cfd_config=cfd_config, device=device)
    print(f"  Checkpoint: {runner.model_path}")

    # Run inference
    smoothing_cfg = _get(cfg, "cp_smoothing")
    flip_x = bool(_get(cfg, "mesh_rotate_z_180", default=False))
    print(f"Loading mesh: {mesh_path}" + ("  [rotate 180° around Z]" if flip_x else ""))
    verts_mm, faces, cp = runner.infer_from_ply(mesh_path, smoothing_cfg=smoothing_cfg, flip_about_x_axis=flip_x)
    print(
        f"  Vertices: {len(verts_mm):,}   Faces: {len(faces):,}\n"
        f"  Cp range: [{cp.min():.4f}, {cp.max():.4f}]  mean: {cp.mean():.4f}"
    )

    out_path = Path(str(_get(cfg, "output_path", default="cp_field.png")))
    # Name the output after the input mesh (keep parent dir from config)
    mesh_stem = Path(mesh_path).stem
    out_path = out_path.with_name(mesh_stem + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w_px = int(_get(cfg, "image_width", default=1600))
    h_px = int(_get(cfg, "image_height", default=800))
    scale = float(_get(cfg, "render_scale", default=2))

    if args.interactive:
        print("Building figure …")
        fig = build_figure(verts_mm, faces, cp, cfg)
        html_path = out_path.with_suffix(".html")
        fig.write_html(str(html_path))
        print(f"Saved (HTML) → {html_path}")
    elif _get(cfg, "cardinal_views"):
        print("Building 6 cardinal views …")
        for name, (azim, elev) in _CARDINAL_VIEWS.items():
            fig = build_figure(
                verts_mm,
                faces,
                cp,
                cfg,
                azimuth_override=azim,
                elevation_override=elev,
            )
            view_path = out_path.with_stem(out_path.stem + f"_{name}")
            fig.write_image(str(view_path), width=w_px, height=h_px, scale=scale)
            print(f"  [{name:7s}] → {view_path}")
    else:
        print("Building figure …")
        fig = build_figure(verts_mm, faces, cp, cfg)
        fig.write_image(str(out_path), width=w_px, height=h_px, scale=scale)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
