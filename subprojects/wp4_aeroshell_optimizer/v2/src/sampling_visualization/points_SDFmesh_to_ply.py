"""Core SamplingVisualizer — combines sampled points and an SDF mesh into one PLY.

Design decisions
----------------
- **open3d** is used exclusively: it can write a single PLY with vertex colours,
  which lets CloudCompare / MeshLab display mesh faces AND coloured point clouds
  in one file without a viewer plugin.
- Mesh vertices are typically coloured **dark blue** or **green**.
- Sampled points are layered with Red, Green, and White/Grey.
- All coordinates are assumed to be in **mm**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import yaml

_CFG = yaml.safe_load((Path(__file__).parents[2] / "configs" / "vis_config.yaml").read_text())
_MESH_COLOUR_DARK_BLUE: tuple[int, int, int] = tuple(_CFG["mesh_colour_dark_blue"])
_POINT_COLOUR_GREEN: tuple[int, int, int] = tuple(_CFG["point_colour_green"])
_POINT_COLOUR_WHITE: tuple[int, int, int] = tuple(_CFG["point_colour_grey"])
_POINT_COLOUR_RED: tuple[int, int, int] = tuple(_CFG["point_colour_red"])


class SamplingVisualizer:
    """Manages the export of training snapshots to PLY files."""

    def __init__(self, output_dir: Path, is_static: bool = False):
        self.output_dir = output_dir
        self.is_static = is_static
        self._already_written = False

    def save_snapshot(
        self,
        points_mm: torch.Tensor | np.ndarray,
        name: str,
        *,
        secondary_points_mm: torch.Tensor | np.ndarray | None = None,
        tertiary_points_mm: torch.Tensor | np.ndarray | None = None,
        mesh_verts: np.ndarray | None = None,
        mesh_faces: np.ndarray | None = None,
        points_colour: tuple[int, int, int] = _POINT_COLOUR_RED,
        secondary_colour: tuple[int, int, int] = _POINT_COLOUR_WHITE,
        tertiary_colour: tuple[int, int, int] = _POINT_COLOUR_WHITE,
        mesh_colour: tuple[int, int, int] = _MESH_COLOUR_DARK_BLUE,
    ) -> Path | None:
        """Write a combined PLY with a solid mesh and up to three point clouds."""
        if self.is_static and self._already_written:
            return None

        combined_verts = []
        combined_cols = []

        # 1. ADD MESH FIRST (Critical so face indices remain valid)
        has_solid_mesh = mesh_verts is not None and mesh_faces is not None
        if mesh_verts is not None:
            v_mesh = _to_numpy_f64(mesh_verts)
            combined_verts.append(v_mesh)
            # Create color array for mesh vertices
            m_col = np.array(mesh_colour) / 255.0
            combined_cols.append(np.tile(m_col, (v_mesh.shape[0], 1)))

        # 2. ADD PRIMARY POINTS
        v_primary = _to_numpy_f64(points_mm)
        combined_verts.append(v_primary)
        p_col = np.array(points_colour) / 255.0
        combined_cols.append(np.tile(p_col, (v_primary.shape[0], 1)))

        # 3. ADD SECONDARY POINTS
        if secondary_points_mm is not None:
            v_sec = _to_numpy_f64(secondary_points_mm)
            combined_verts.append(v_sec)
            s_col = np.array(secondary_colour) / 255.0
            combined_cols.append(np.tile(s_col, (v_sec.shape[0], 1)))

        # 4. ADD TERTIARY POINTS
        if tertiary_points_mm is not None:
            v_tert = _to_numpy_f64(tertiary_points_mm)
            combined_verts.append(v_tert)
            t_col = np.array(tertiary_colour) / 255.0
            combined_cols.append(np.tile(t_col, (v_tert.shape[0], 1)))

        # Concatenate all components
        all_v = np.vstack(combined_verts)
        all_c = np.vstack(combined_cols)
        out_path = self.output_dir / f"{name}.ply"

        if has_solid_mesh:
            # Create a TriangleMesh to preserve solid surface
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(all_v)
            mesh.vertex_colors = o3d.utility.Vector3dVector(all_c)
            # The face indices still point correctly to the first N vertices
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_faces, dtype=np.int32))
            o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
            print(f"[SamplingVisualizer] Saved combined mesh+points → {out_path}")
        else:
            # Create a PointCloud if no faces are provided
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_v)
            pcd.colors = o3d.utility.Vector3dVector(all_c)
            o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
            print(f"[SamplingVisualizer] Saved point cloud → {out_path}")

        self._already_written = True
        return out_path

    def write_legend(self, text: str) -> Path:
        """Write a ``legend.txt`` explaining the colour coding."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "legend.txt"
        path.write_text(text, encoding="utf-8")
        return path

    def reset(self) -> None:
        """Allow a new write even when ``is_static=True``."""
        self._already_written = False


###
# Private helpers
###


def _to_numpy_f64(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert input to (N, 3) float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    if x.ndim == 1:
        x = x.reshape(-1, 3)
    return np.asarray(x, dtype=np.float64)
