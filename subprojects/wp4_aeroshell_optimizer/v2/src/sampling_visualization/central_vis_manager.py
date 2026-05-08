"""Central Visualization Manager for AISCO aeroshell optimizer.

Consolidates all 2D visualization writes into a single API that produces
epoch-indexed PLY files under ``<run_dir>/<visual_type>/epoch_XXXXX.ply``
and auto-generates convergence GIFs via ``finalize()``.

Plan 012 / v0.7.0
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (
    generate_gifs_flat,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.scalar_color_mapping import (
    map_scalar_to_rgb_mpl,
)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "viz_config.yaml"

# Uniform dark-blue used when no colormap is specified (RGB, float [0,1])
_DARK_BLUE_RGB: tuple[float, float, float] = (0.0 / 255.0, 0.0 / 255.0, 139.0 / 255.0)


def _log(event: dict[str, Any]) -> None:
    """Write a JSON telemetry event to stderr (tqdm-safe, always visible in SLURM logs)."""
    sys.stderr.write(json.dumps(event) + "\n")


def face_to_vertex_average(
    face_values: np.ndarray,
    faces: np.ndarray,
    verts: np.ndarray,
) -> np.ndarray:
    """Area-weighted average of face scalar values at each vertex.

    Args:
        face_values: Per-face scalar values, shape ``(F,)``.
        faces: Face index array, shape ``(F, 3)``.
        verts: Vertex position array, shape ``(V, 3)``.

    Returns:
        Per-vertex scalar values, shape ``(V,)`` as ``float32``.

    Raises:
        ValueError: If ``len(face_values) != len(faces)``.
    """
    face_values = np.asarray(face_values, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    verts = np.asarray(verts, dtype=np.float64)

    if len(face_values) != len(faces):
        raise ValueError(
            f"face_values length ({len(face_values)}) does not match faces length ({len(faces)}).  Pass one scalar per face."
        )

    n_verts = len(verts)

    # Compute per-face triangle areas
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    face_areas = 0.5 * np.linalg.norm(cross, axis=1).astype(np.float32)  # (F,)

    # Accumulate area-weighted values and total areas per vertex
    weighted_sum = np.zeros(n_verts, dtype=np.float64)
    area_sum = np.zeros(n_verts, dtype=np.float64)

    weighted_vals = face_values * face_areas  # (F,)
    for i in range(3):
        np.add.at(weighted_sum, faces[:, i], weighted_vals)
        np.add.at(area_sum, faces[:, i], face_areas)

    # Avoid division by zero for isolated vertices
    safe_area = np.where(area_sum > 0.0, area_sum, 1.0)
    return (weighted_sum / safe_area).astype(np.float32)


def _probe_alpha_support() -> bool:
    """Write a tiny 3-vertex PLY with RGBA vertex colours and read back its header.

    Returns ``True`` if the round-trip preserves the alpha channel
    (i.e. the header contains ``property uchar alpha``).
    """
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    rgba = np.array([[255, 0, 0, 200], [0, 255, 0, 200], [0, 0, 255, 200]], dtype=np.uint8)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    # Open3D uses [0, 1] float colors, so normalise
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgba[:, :3].astype(np.float64) / 255.0)

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        o3d.io.write_triangle_mesh(str(tmp_path), mesh, write_ascii=True)
        header = tmp_path.read_bytes()[:512]
    except Exception:
        return False
    else:
        return b"property uchar alpha" in header
    finally:
        tmp_path.unlink(missing_ok=True)


def _nested_get(mapping: dict[str, Any], key_path: str) -> dict[str, Any] | None:
    """Navigate a nested dict using a slash-separated key path.

    Example::

        _nested_get(cfg, "aero/cp_field")  # → cfg["aero"]["cp_field"]
        _nested_get(cfg, "mesh_shape")      # → cfg["mesh_shape"]

    Returns ``None`` if any segment is missing.
    """
    parts = key_path.split("/")
    node: Any = mapping
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


class CentralVisManager:
    """Centralised writer for per-epoch PLY visualizations.

    Wraps all 2D visualization writes under a single API.  Produces
    ``<run_dir>/<visual_type>/epoch_XXXXX.ply`` files; ``finalize()``
    auto-renders convergence GIFs from those sequences.

    Args:
        run_dir: Root directory for all visualization output.
            Created if it does not exist.
        run_id: Human-readable run identifier for structured log entries.
        config_path: Optional path to ``viz_config.yaml``.
            Defaults to the ``viz_config.yaml`` file in the same package.

    Raises:
        OSError / NotADirectoryError: If ``run_dir`` cannot be created.
    """

    def __init__(
        self,
        run_dir: Path | str,
        run_id: str,
        config_path: Path | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

        cfg_path = config_path or _DEFAULT_CONFIG_PATH
        self._cfg = yaml.safe_load(Path(cfg_path).read_text())
        self._visual_types: dict[str, Any] = self._cfg.get("visual_types", {})
        self._extras: dict[str, Any] = self._cfg.get("extras", {})

        # Alpha probe: done once at init; result cached for all writes
        self._alpha_use_struct_pack: bool = _probe_alpha_support()
        # Track every output directory written via save_ply / save_ply_points so that
        # finalize() can GIF slot-specific sub-paths (e.g. latent/sampling/slot00_foo)
        # even when they have no direct entry in visual_types config.
        self._written_dirs: set[Path] = set()
        # Active population slot for diversity training.  When set, save_ply() and
        # save_ply_points() automatically append "/slot_XX" to the visual_type so
        # that each population member accumulates its own PLY sequence.
        self._active_slot: int | None = None
        _log(
            {
                "event": "central_vis_manager_init",
                "run_id": run_id,
                "run_dir": str(self.run_dir),
                "alpha_support": self._alpha_use_struct_pack,
            }
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _type_config(self, visual_type: str) -> dict[str, Any] | None:
        """Return config dict for ``visual_type``, or ``None`` if missing/disabled.

        When the exact path is not in config, walks up the path hierarchy to find
        the nearest ancestor that has an ``enabled`` key (leaf config node).  This
        allows slot-specific visual types such as ``"latent/sampling/slot00_foo"``
        to inherit the ``"latent/sampling"`` configuration without requiring each
        slot to be listed explicitly in ``viz_config.yaml``.
        """
        cfg = _nested_get(self._visual_types, visual_type)
        if cfg is None:
            # Walk up the path hierarchy looking for the nearest enabled ancestor.
            parts = visual_type.split("/")
            for depth in range(len(parts) - 1, 0, -1):
                parent_path = "/".join(parts[:depth])
                parent_cfg = _nested_get(self._visual_types, parent_path)
                if parent_cfg is not None and "enabled" in parent_cfg:
                    cfg = parent_cfg
                    break
        if cfg is None:
            return None
        if not cfg.get("enabled", False):
            return None
        return cfg

    def _resolve_bool(self, per_type_val: bool | None, global_val: bool | None) -> bool:
        """Return per-type override if not None, else global, else False."""
        if per_type_val is not None:
            return bool(per_type_val)
        if global_val is not None:
            return bool(global_val)
        return False

    def _output_dir(self, visual_type: str) -> Path:
        """Return (and create) the output directory for ``visual_type``.

        Also records the directory in ``_written_dirs`` so that ``finalize()``
        can generate GIFs for paths not explicitly listed in config (e.g. slot
        sub-paths).
        """
        out_dir = self.run_dir / visual_type
        out_dir.mkdir(parents=True, exist_ok=True)
        self._written_dirs.add(out_dir)
        return out_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_active_slot(self, slot_idx: int | None) -> None:
        """Set (or clear) the active population slot.

        When set to an integer, ``save_ply()`` and ``save_ply_points()``
        automatically append ``/slot_XX`` to the ``visual_type`` argument,
        keeping each diversity-training population member in its own PLY
        sequence.  Call with ``None`` after the per-slot loop finishes.

        Args:
            slot_idx: Zero-based population index, or ``None`` to disable.
        """
        self._active_slot = slot_idx

    def save_ply(
        self,
        visual_type: str,
        epoch: int,
        verts: Any,
        faces: np.ndarray | None = None,
        scalar_values: np.ndarray | None = None,
    ) -> Path | None:
        """Write one epoch PLY for a mesh or point-cloud visual type.

        Args:
            visual_type: Key matching an entry in ``visual_types`` in the
                config (e.g. ``"mesh_shape"``, ``"aero/cp_field"``).
            epoch: Current training epoch (used to name the file).
            verts: Vertex positions, shape ``(V, 3)``.  Accepts numpy arrays
                or PyTorch tensors.
            faces: Optional face indices, shape ``(F, 3)``.
            scalar_values: Optional per-vertex or per-face scalar values,
                shape ``(V,)`` or ``(F,)``.  When ``shape[0] == F`` and
                ``faces`` is provided, values are area-averaged to vertices.
                When ``None``, a uniform dark-blue color is used.

        Returns:
            Path to the written PLY, or ``None`` when the type is disabled or
            writing fails.
        """
        if self._active_slot is not None and "/slot_" not in visual_type:
            visual_type = f"{visual_type}/slot_{self._active_slot:02d}"

        type_cfg = self._type_config(visual_type)
        if type_cfg is None:
            return None

        interval: int | None = type_cfg.get("interval")
        if interval is not None and epoch % interval != 0:
            return None

        _log(
            {
                "event": "ply_write_started",
                "run_id": self.run_id,
                "visual_type": visual_type,
                "epoch": epoch,
            }
        )

        try:
            return self._do_save_ply(visual_type, epoch, verts, faces, scalar_values, type_cfg)
        except Exception as exc:
            _log(
                {
                    "event": "vis_write_failed",
                    "run_id": self.run_id,
                    "visual_type": visual_type,
                    "epoch": epoch,
                    "error": str(exc),
                }
            )
            return None

    def _resolve_rgb_colors(
        self,
        scalar_values: np.ndarray | None,
        n_verts: int,
        n_faces: int,
        faces_np: np.ndarray | None,
        verts_np: np.ndarray,
        type_cfg: dict[str, Any],
        visual_type: str,
        epoch: int,
    ) -> np.ndarray:
        """Return (n_verts, 3) float32 RGB colors for a PLY write."""
        if scalar_values is None:
            return np.tile(_DARK_BLUE_RGB, (n_verts, 1)).astype(np.float32)

        sv = np.asarray(scalar_values, dtype=np.float32).reshape(-1)

        if sv.shape[0] == n_verts:
            vertex_scalars: np.ndarray | None = sv
        elif sv.shape[0] == n_faces and faces_np is not None:
            vertex_scalars = face_to_vertex_average(sv, faces_np, verts_np)
        else:
            _log(
                {
                    "event": "scalar_shape_mismatch_warning",
                    "run_id": self.run_id,
                    "visual_type": visual_type,
                    "epoch": epoch,
                    "scalar_shape": list(sv.shape),
                    "n_verts": n_verts,
                    "n_faces": n_faces,
                    "message": (
                        f"scalar values shape {sv.shape} matches neither "
                        f"n_verts={n_verts} nor n_faces={n_faces}; "
                        "falling back to uniform dark-blue color"
                    ),
                }
            )
            return np.tile(_DARK_BLUE_RGB, (n_verts, 1)).astype(np.float32)

        colormap: str | None = type_cfg.get("colormap")
        if colormap is None:
            return np.tile(_DARK_BLUE_RGB, (n_verts, 1)).astype(np.float32)

        vmin: float | None = type_cfg.get("vmin")
        vmax: float | None = type_cfg.get("vmax")
        # vertex_scalars is always set when sv.shape[0] matches n_verts or n_faces
        assert vertex_scalars is not None
        return map_scalar_to_rgb_mpl(vertex_scalars, colormap, vmin=vmin, vmax=vmax)

    def _do_save_ply(
        self,
        visual_type: str,
        epoch: int,
        verts: Any,
        faces: np.ndarray | None,
        scalar_values: np.ndarray | None,
        type_cfg: dict[str, Any],
    ) -> Path:
        # --- Convert verts -----------------------------------------------
        if hasattr(verts, "detach"):
            verts_np = verts.detach().cpu().numpy().astype(np.float64)
        else:
            verts_np = np.asarray(verts, dtype=np.float64)

        faces_np: np.ndarray | None = None
        if faces is not None:
            faces_np = np.asarray(faces, dtype=np.int32)

        n_verts = len(verts_np)
        n_faces = len(faces_np) if faces_np is not None else 0

        # --- Resolve scalar values → RGB colors ---------------------------
        rgb_colors = self._resolve_rgb_colors(
            scalar_values,
            n_verts,
            n_faces,
            faces_np,
            verts_np,
            type_cfg,
            visual_type,
            epoch,
        )

        # --- Optionally merge coordinate frame ----------------------------
        coord_sys = self._resolve_bool(
            type_cfg.get("coordinate_system"),
            self._extras.get("coordinate_system"),
        )

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        if faces_np is not None:
            mesh.triangles = o3d.utility.Vector3iVector(faces_np)

        # Assign vertex colors (open3d uses float [0, 1])
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_colors.astype(np.float64))

        if coord_sys:
            bbox_pts = verts_np
            diag = float(np.linalg.norm(bbox_pts.max(axis=0) - bbox_pts.min(axis=0)))
            if diag < 1e-6:
                diag = 1.0
            scale = float(self._extras.get("coordinate_system_scale", 0.1))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale * diag, origin=[0.0, 0.0, 0.0])
            mesh = mesh + frame

        # --- Write PLY ----------------------------------------------------
        out_dir = self._output_dir(visual_type)
        out_path = out_dir / f"epoch_{epoch:05d}.ply"
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)

        _log(
            {
                "event": "ply_write_completed",
                "run_id": self.run_id,
                "visual_type": visual_type,
                "epoch": epoch,
                "path": str(out_path),
            }
        )

        return out_path

    def save_ply_points(
        self,
        visual_type: str,
        epoch: int,
        points: Any,
        secondary_points: Any | None = None,
        tertiary_points: Any | None = None,
        mesh_verts: np.ndarray | None = None,
        mesh_faces: np.ndarray | None = None,
    ) -> Path | None:
        """Write one epoch PLY for a latent-space / point-cloud visual type.

        Delegates to ``SamplingVisualizer`` (preserves its colour coding) and
        writes to ``<run_dir>/<visual_type>/epoch_XXXXX.ply``.  Does **not**
        apply interval filtering — the caller controls cadence.

        Args:
            visual_type: Key in ``visual_types`` config (e.g.
                ``"latent/sampling"``).
            epoch: Current training epoch.
            points: Primary point cloud, shape ``(N, 3)``.
            secondary_points: Optional secondary point cloud.
            tertiary_points: Optional tertiary point cloud.
            mesh_verts: Optional background mesh vertices.
            mesh_faces: Optional background mesh face indices.

        Returns:
            Path to the written PLY, or ``None`` if the type is disabled or
            writing fails.
        """
        if self._active_slot is not None and "/slot_" not in visual_type:
            visual_type = f"{visual_type}/slot_{self._active_slot:02d}"

        type_cfg = self._type_config(visual_type)
        if type_cfg is None:
            return None

        interval: int | None = type_cfg.get("interval")
        if interval is not None and epoch % interval != 0:
            return None

        try:
            from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.points_SDFmesh_to_ply import (  # noqa: PLC0415
                SamplingVisualizer,
            )

            out_dir = self._output_dir(visual_type)
            sv = SamplingVisualizer(output_dir=out_dir, is_static=False)
            path = sv.save_snapshot(
                points_mm=points,
                name=f"epoch_{epoch:05d}",
                secondary_points_mm=secondary_points,
                tertiary_points_mm=tertiary_points,
                mesh_verts=mesh_verts,
                mesh_faces=mesh_faces,
            )

            _log(
                {
                    "event": "ply_points_write_completed",
                    "run_id": self.run_id,
                    "visual_type": visual_type,
                    "epoch": epoch,
                    "path": str(path) if path else None,
                }
            )
        except Exception as exc:
            _log(
                {
                    "event": "vis_write_failed",
                    "run_id": self.run_id,
                    "visual_type": visual_type,
                    "epoch": epoch,
                    "error": str(exc),
                }
            )
            return None
        else:
            return path
        return None

    def finalize(self) -> None:
        """Auto-generate convergence GIFs for all enabled visual types.

        Scans each enabled visual type directory for ``epoch_XXXXX.ply``
        files.  When at least one file is found, calls
        ``generate_gifs_flat()`` to render ``animation.gif``.  Failures in
        individual types are caught and logged; other types continue.
        """
        flat_pattern = re.compile(r"^epoch_(\d{5})\.ply$")

        # Collect all enabled visual types from config
        def _collect_types(node: dict[str, Any], prefix: str) -> list[str]:
            types: list[str] = []
            for key, val in node.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(val, dict):
                    if "enabled" in val:
                        # Leaf node
                        if val.get("enabled", False):
                            types.append(path)
                    else:
                        # Namespace node — recurse
                        types.extend(_collect_types(val, path))
            return types

        enabled_types = _collect_types(self._visual_types, "")

        # Merge config-enabled dirs with any dirs written at runtime (slot sub-paths etc.)
        all_dirs: list[Path] = [self.run_dir / vt for vt in enabled_types]
        for written_dir in self._written_dirs:
            if written_dir not in all_dirs:
                all_dirs.append(written_dir)

        for type_dir in all_dirs:
            # Derive a label for telemetry from the directory path relative to run_dir
            visual_type = str(type_dir.relative_to(self.run_dir))
            if not type_dir.is_dir():
                continue

            ply_files = [f for f in type_dir.iterdir() if flat_pattern.match(f.name)]
            if not ply_files:
                continue

            _log(
                {
                    "event": "gif_render_started",
                    "run_id": self.run_id,
                    "visual_type": visual_type,
                    "n_frames": len(ply_files),
                }
            )

            try:
                generate_gifs_flat(type_dir)
                _log(
                    {
                        "event": "gif_render_completed",
                        "run_id": self.run_id,
                        "visual_type": visual_type,
                    }
                )
            except Exception as exc:
                _log(
                    {
                        "event": "gif_render_failed",
                        "run_id": self.run_id,
                        "visual_type": visual_type,
                        "error": str(exc),
                    }
                )
