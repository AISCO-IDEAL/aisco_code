"""Core SamplingVisualizer — combines sampled points and an SDF mesh into one PLY.

Design decisions
----------------
- **open3d** is used exclusively: it can write a single PLY with vertex colours,
  which lets CloudCompare / MeshLab display mesh faces AND coloured point clouds
  in one file without a viewer plugin.
- Mesh vertices are coloured **dark blue** (0, 0, 139 / 255).
- Sampled points are coloured **red** (220, 30, 30 / 255).
- All coordinates are assumed to be in **mm**.  No unit conversion is performed.
- ``is_static=True``: the visualizer writes the PLY only once and skips subsequent
  calls.  Use this for eikonal / smoothness losses where ``self.points`` is fixed
  for the entire run.
- ``is_static=False``: every ``save_snapshot()`` call writes a new file.
  Use this for diversity / pretraining where points change each epoch.
- **open3d is imported lazily** inside ``save_snapshot()`` so the module can be
  imported in environments where open3d is not installed without raising at
  import time.  A clear ``ImportError`` is raised only when the function is
  actually called.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import open3d as o3d


_MESH_COLOUR_DARK_BLUE: tuple[int, int, int] = (0, 0, 139)
_POINT_COLOUR_GREEN: tuple[int, int, int] = (34, 139, 34)  # forest green — quality-of-fit / surface samples
_POINT_COLOUR_WHITE: tuple[int, int, int] = (220, 220, 220)  # light grey-white — domain / eikonal samples
_POINT_COLOUR_RED: tuple[int, int, int] = (220, 30, 30)  # kept for backwards compatibility


class SamplingVisualizer:
    """Write a combined mesh + sampled-points PLY for visual inspection.

    Args:
        output_dir: Directory where PLY files will be saved.  Created if absent.
        is_static: When ``True``, ``save_snapshot`` writes exactly one file and
            silently skips all subsequent calls.  Set this for losses whose
            sample points are fixed for the whole training run (eikonal,
            smoothness).  Set ``False`` for per-epoch sampling (diversity,
            pretraining).
    """

    def __init__(self, output_dir: Path | str, *, is_static: bool = False) -> None:
        self.output_dir = Path(output_dir)
        self.is_static = is_static
        self._already_written = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        points_mm: torch.Tensor | np.ndarray,
        mesh_verts: np.ndarray | None,
        mesh_faces: np.ndarray | None,
        name: str,
        *,
        secondary_points_mm: torch.Tensor | np.ndarray | None = None,
        points_colour: tuple[int, int, int] = _POINT_COLOUR_GREEN,
        secondary_colour: tuple[int, int, int] = _POINT_COLOUR_WHITE,
        max_points: int = 8192,
    ) -> Path | None:
        """Write a combined PLY file with coloured mesh and sampled points.

        Args:
            points_mm: (N, 3) primary sample points in mm (surface-concentrated).
            mesh_verts: (V, 3) float32 array of mesh vertices in mm.
                ``None`` to skip mesh (points-only file).
            mesh_faces: (F, 3) int64 array of triangle face indices.
                ``None`` to skip mesh.
            name: Filename stem (no extension).
            secondary_points_mm: (M, 3) optional secondary sample points in mm
                (domain / eikonal). Coloured ``secondary_colour`` (default light
                grey).  ``None`` to omit.
            points_colour: RGB tuple for primary points (default forest green).
            secondary_colour: RGB tuple for secondary points (default light grey).
            max_points: Cap on number of points written per group.  Random
                subsampling is applied when the input exceeds this limit.

        Returns:
            Path to the written PLY, or ``None`` when skipped (``is_static`` and
            already written).
        """
        if self.is_static and self._already_written:
            return None

        try:
            import open3d as o3d  # noqa: PLC0415 — lazy: open3d is optional
        except ImportError as exc:
            raise ImportError("open3d is required for SamplingVisualizer. Install it with: pip install open3d") from exc

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Normalise points to numpy (N, 3) float64 ---
        pts = _to_numpy_f64(points_mm)
        if pts.shape[0] > max_points:
            rng = np.random.default_rng()
            idx = rng.choice(pts.shape[0], max_points, replace=False)
            pts = pts[idx]

        combined = o3d.geometry.PointCloud()
        all_verts: list[np.ndarray] = []
        all_colours: list[np.ndarray] = []
        tri_mesh: o3d.geometry.TriangleMesh | None = None

        # --- Mesh component (dark blue) ---
        has_mesh = mesh_verts is not None and mesh_faces is not None
        if has_mesh:
            try:
                tri_mesh = _build_o3d_mesh(mesh_verts, mesh_faces)
                mesh_verts_np = np.asarray(tri_mesh.vertices, dtype=np.float64)
                mesh_colour = np.tile(
                    np.array(_MESH_COLOUR_DARK_BLUE, dtype=np.float64) / 255.0,
                    (mesh_verts_np.shape[0], 1),
                )
                all_verts.append(mesh_verts_np)
                all_colours.append(mesh_colour)
            except Exception as exc:  # mesh build failure is non-fatal
                warnings.warn(
                    f"[SamplingVisualizer] Mesh build failed for '{name}': {exc}. Writing points-only PLY.",
                    stacklevel=2,
                )
                has_mesh = False

        # --- Point component ---
        pt_colour = np.tile(
            np.array(points_colour, dtype=np.float64) / 255.0,
            (pts.shape[0], 1),
        )
        all_verts.append(pts)
        all_colours.append(pt_colour)

        # --- Secondary point component (domain / eikonal) ---
        if secondary_points_mm is not None:
            sec_pts = _to_numpy_f64(secondary_points_mm)
            if sec_pts.shape[0] > max_points:
                rng2 = np.random.default_rng()
                idx2 = rng2.choice(sec_pts.shape[0], max_points, replace=False)
                sec_pts = sec_pts[idx2]
            sec_colour = np.tile(
                np.array(secondary_colour, dtype=np.float64) / 255.0,
                (sec_pts.shape[0], 1),
            )
            all_verts.append(sec_pts)
            all_colours.append(sec_colour)

        # --- Merge into a single PointCloud for export ---
        combined.points = o3d.utility.Vector3dVector(np.vstack(all_verts))
        combined.colors = o3d.utility.Vector3dVector(np.vstack(all_colours))

        out_path = self.output_dir / f"{name}.ply"
        o3d.io.write_point_cloud(str(out_path), combined, write_ascii=False)

        # If mesh is present, also write the triangle mesh alongside
        # (some viewers can only show the mesh if triangles are present)
        if has_mesh and tri_mesh is not None:
            mesh_path = self.output_dir / f"{name}_mesh_only.ply"
            o3d.io.write_triangle_mesh(str(mesh_path), tri_mesh, write_ascii=False)

        print(f"[SamplingVisualizer] Saved → {out_path}")
        self._already_written = True
        return out_path

    def write_legend(self, text: str) -> Path:
        """Write a ``legend.txt`` into ``output_dir`` explaining the colour coding.

        Overwrites any existing legend file.  Safe to call multiple times.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "legend.txt"
        path.write_text(text, encoding="utf-8")
        print(f"[SamplingVisualizer] Legend → {path}")
        return path

    def reset(self) -> None:
        """Allow a new write even when ``is_static=True``."""
        self._already_written = False


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _to_numpy_f64(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert tensor or ndarray to (N, 3) float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _build_o3d_mesh(
    verts: np.ndarray,
    faces: np.ndarray,
) -> o3d.geometry.TriangleMesh:  # type: ignore[name-defined]
    """Construct a coloured open3d TriangleMesh from numpy arrays."""
    import open3d as o3d  # noqa: PLC0415

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(verts, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    mesh.compute_vertex_normals()
    colour = np.array(_MESH_COLOUR_DARK_BLUE, dtype=np.float64) / 255.0
    mesh.paint_uniform_color(colour)
    return mesh
