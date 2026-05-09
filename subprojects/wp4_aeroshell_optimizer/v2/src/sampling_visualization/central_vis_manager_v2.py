"""CentralVisManagerV2 — infrastructure base + named visualization methods.

_BaseVisManager owns:
  - Config loading and lookup helpers
  - ThreadPoolExecutor for fire-and-forget PLY writes
  - finalize() — drains the PLY pool then renders GIFs in a subprocess
  - __del__ + atexit registration so finalize() runs even on crash

CentralVisManagerV2(_BaseVisManager) adds one named method per visual type.
Anyone who needs a new visual type subclasses _BaseVisManager (or extends
CentralVisManagerV2) and adds their own method — the base provides all infra.

Switch via ``vis_manager: v2`` in general_config.yaml.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.coloring import (
    map_scalar_to_rgb_mpl,
    solid_rgb,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.mesh_ops import (
    face_scalars_to_vertex,
)
from subprojects.wp5.unified_mesh_extraction.sdf2mesh import MeshSDFExtractor

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "viz_config.yaml"
_DARK_BLUE_RGB: tuple[float, float, float] = (0.0, 0.0, 139.0 / 255.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_f64(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_i32(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.int32)
    return np.asarray(x, dtype=np.int32)


def _log(event: dict[str, Any]) -> None:
    sys.stderr.write(json.dumps(event) + "\n")


def _nested_get(mapping: dict[str, Any], key_path: str) -> dict[str, Any] | None:
    parts = key_path.split("/")
    node: Any = mapping
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, dict) else None


# ---------------------------------------------------------------------------
# Module-level task functions (must be picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------


def _write_ply_task(
    out_path: Path,
    verts_np: np.ndarray,
    faces_np: np.ndarray | None,
    rgb_np: np.ndarray,
    add_coord_frame: bool,
    coord_frame_scale: float,
) -> tuple[bool, str, str | None]:
    try:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_np)
        if faces_np is not None:
            mesh.triangles = o3d.utility.Vector3iVector(faces_np)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb_np.astype(np.float64))
        if add_coord_frame:
            diag = float(np.linalg.norm(verts_np.max(axis=0) - verts_np.min(axis=0)))
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=coord_frame_scale * max(diag, 1e-6), origin=[0.0, 0.0, 0.0]
            )
            mesh = mesh + frame
        o3d.io.write_triangle_mesh(str(out_path), mesh, write_ascii=False)
        return (True, str(out_path), None)
    except Exception as exc:
        return (False, str(out_path), str(exc))


def _render_gif_task(type_dir: Path) -> tuple[bool, str, str | None]:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (  # noqa: PLC0415
            generate_gifs_flat,
        )

        generate_gifs_flat(type_dir)
        return (True, str(type_dir), None)
    except Exception as exc:
        return (False, str(type_dir), str(exc))


# ---------------------------------------------------------------------------
# Base class — infrastructure only
# ---------------------------------------------------------------------------


class _BaseVisManager:
    """Async PLY write infrastructure. Subclass and add named vis methods."""

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
        self._cfg: dict[str, Any] = yaml.safe_load(Path(cfg_path).read_text())
        self._visual_types: dict[str, Any] = self._cfg.get("visual_types", {})
        self._extras: dict[str, Any] = self._cfg.get("extras", {})

        self._written_dirs: set[Path] = set()
        self._pending_ply: list[concurrent.futures.Future[tuple[bool, str, str | None]]] = []
        self._ply_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._finalized = False
        self._active_slot: int | None = None
        self._assembly_meshes: tuple | None = None
        atexit.register(self.finalize)

        _log({"event": "cvm_v2_init", "run_id": run_id, "run_dir": str(self.run_dir)})
        self._load_assembly_meshes()

    # --- Config helpers -----------------------------------------------------

    def _type_config(self, visual_type: str) -> dict[str, Any] | None:
        cfg = _nested_get(self._visual_types, visual_type)
        if cfg is None:
            parts = visual_type.split("/")
            for depth in range(len(parts) - 1, 0, -1):
                parent_cfg = _nested_get(self._visual_types, "/".join(parts[:depth]))
                if parent_cfg is not None and "enabled" in parent_cfg:
                    cfg = parent_cfg
                    break
        if cfg is None or not cfg.get("enabled", False):
            return None
        return cfg

    def _resolve_bool(self, per_type_val: bool | None, global_val: bool | None) -> bool:
        if per_type_val is not None:
            return bool(per_type_val)
        if global_val is not None:
            return bool(global_val)
        return False

    def _is_enabled(self, visual_type: str) -> bool:
        return self._type_config(visual_type) is not None

    def _slotted(self, visual_type: str) -> str:
        """Return the slot-qualified visual_type when a slot is active."""
        if self._active_slot is not None:
            return f"{visual_type}/slot_{self._active_slot:02d}"
        return visual_type

    def set_active_slot(self, slot_idx: int | None) -> None:
        """Tag all subsequent writes with a population-member suffix.

        In diversity training the optimiser loops over N latent vectors (z_0 …
        z_{N-1}), calling the same loss — and therefore the same vis methods —
        once per member per epoch.  Without slot tagging every member would
        overwrite the same PLY file and only the last write would survive.

        Call ``set_active_slot(i)`` before computing member i's loss; every vis
        method called while a slot is active writes to
        ``<visual_type>/slot_{i:02d}/epoch_XXXXX.ply`` instead of
        ``<visual_type>/epoch_XXXXX.ply``.  The config entry for the parent
        type (e.g. ``mesh_shape``) is inherited automatically via the ancestor
        lookup in ``_type_config``.

        Call ``set_active_slot(None)`` after the population loop to return to
        non-slotted writes.

        Args:
            slot_idx: Zero-based population index, or None to clear.
        """
        self._active_slot = slot_idx

    def _load_assembly_meshes(self) -> None:
        """Auto-tessellate the inner/outer bbox solids from regulations and store them.

        Called once from __init__.  Skipped silently if mesh_assembly_v4 is disabled
        in viz_config.yaml, or if WP2 / the assembly_id cannot be resolved.
        """
        if not self._is_enabled("mesh_assembly_v4"):
            return
        try:
            from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.custom_modes.bounding_box_mode_utils import (  # noqa: PLC0415
                load_bb_assembly_id,
                load_inner_outer_solids,
                merge_meshes,
                solid_to_kaolin_mesh,
            )

            assembly_id = load_bb_assembly_id()
            inner_solids, outer_solid = load_inner_outer_solids(assembly_id)

            inner_meshes = [solid_to_kaolin_mesh(s, device="cpu", tolerance=2.0, angular_tolerance=0.4) for s in inner_solids]
            inner_verts, inner_faces, _ = merge_meshes(inner_meshes)
            outer_verts, outer_faces = solid_to_kaolin_mesh(outer_solid, device="cpu", tolerance=2.0, angular_tolerance=0.4)

            self.set_assembly_meshes(
                inner_verts, inner_faces,
                outer_verts, outer_faces,
                sdf_color=(0.2, 1.0, 1.0),
                inner_color=(1.0, 0.5, 0.0),
                outer_color=(1.0, 0.5, 0.0),
                sdf_alpha=0.5,
                outer_alpha=0.2,
            )
            _log({"event": "assembly_meshes_loaded", "run_id": self.run_id, "assembly_id": assembly_id})
        except Exception as exc:
            _log({"event": "assembly_meshes_load_failed", "run_id": self.run_id, "error": str(exc),
                  "message": "Call set_assembly_meshes() manually if the assembly cannot be loaded automatically."})

    def set_assembly_meshes(
        self,
        inner_verts: Any,
        inner_faces: Any,
        outer_verts: Any,
        outer_faces: Any,
        *,
        sdf_color: tuple[float, float, float] = (0.2, 1.0, 1.0),
        inner_color: tuple[float, float, float] = (1.0, 0.55, 0.0),
        outer_color: tuple[float, float, float] = (1.0, 0.35, 0.15),
        sdf_alpha: float = 0.5,
        outer_alpha: float = 0.2,
    ) -> None:
        """Store constant bbox meshes and display colours for vis_mesh_assembly_v4.

        Call once after tessellating the inner/outer bounding-box solids (the
        expensive kaolin step).  The tensors are converted to CPU immediately so
        the training thread never has to wait on subsequent vis calls.

        Args:
            inner_verts / inner_faces: Pre-tessellated inner bounding-box solid.
            outer_verts / outer_faces: Pre-tessellated outer bounding-box solid.
            sdf_color: RGB colour for the evolving SDF mesh.
            inner_color: RGB colour for the inner assembly.
            outer_color: RGB colour for the outer bounding box.
            sdf_alpha: Opacity of the SDF mesh in [0, 1].
            outer_alpha: Opacity of the outer bbox in [0, 1].
        """
        import torch  # noqa: PLC0415

        def _cpu(x: Any) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return torch.from_numpy(np.asarray(x))

        self._assembly_meshes = (
            _cpu(inner_verts), _cpu(inner_faces),
            _cpu(outer_verts), _cpu(outer_faces),
            sdf_color, inner_color, outer_color, sdf_alpha, outer_alpha,
        )
        _log({"event": "assembly_meshes_set", "run_id": self.run_id})

    def make_vis_extractor(self, device: str = "cuda") -> MeshSDFExtractor:
        """Create a low-resolution MeshSDFExtractor for visualization.

        The bbox is derived automatically from the outer-prism regulations, scaled
        by ``vis_extraction.bbox_scale_factor`` (default 1.1).  Resolution is read
        from ``vis_extraction.resolution`` (default 64).  Other extraction
        parameters (isovalue, batch_size, etc.) are inherited from the unified
        extractor's own YAML config.

        Use the returned extractor to cheaply extract the SDF mesh once per epoch
        and reuse it across all vis methods that need one.

        Args:
            device: Torch device for grid evaluation (``"cpu"`` is fine for vis).

        Returns:
            A configured MeshSDFExtractor ready to call ``.extract_mesh(model)``.
        """
        from subprojects.wp2_assembler.version4.utils.reg_imp import (  # noqa: PLC0415
            get_outer_prism_bounds_mm,
            scale_outer_prism,
        )
        from subprojects.wp5.unified_mesh_extraction.utils import (  # noqa: PLC0415
            MeshExtractionConfig,
            load_mesh_extraction_config,
        )

        vis_cfg = self._cfg.get("vis_extraction") or {}
        resolution = int(vis_cfg.get("resolution", 64))
        bbox_scale = float(vis_cfg.get("bbox_scale_factor", 1.1))

        prism = get_outer_prism_bounds_mm()
        scaled = scale_outer_prism(prism, bbox_scale)
        bbox = [
            [scaled["x"][0], scaled["x"][1]],
            [scaled["y"][0], scaled["y"][1]],
            [scaled["z"][0], scaled["z"][1]],
        ]

        base_cfg = load_mesh_extraction_config()
        vis_config = MeshExtractionConfig(
            device=device,
            isovalue=base_cfg.isovalue,
            base_resolution=resolution,
            subdivision_factor=base_cfg.subdivision_factor,
            max_refine_levels=base_cfg.max_refine_levels,
            batch_size=base_cfg.batch_size,
            bbox=bbox,
            target_num_vertices=None,
            vertex_count_tolerance=base_cfg.vertex_count_tolerance,
        )
        return MeshSDFExtractor(config=vis_config)

    def _output_dir(self, visual_type: str) -> Path:
        out_dir = self.run_dir / visual_type
        out_dir.mkdir(parents=True, exist_ok=True)
        self._written_dirs.add(out_dir)
        return out_dir

    # --- Color resolution ---------------------------------------------------

    def _resolve_rgb(
        self,
        scalars: np.ndarray | None,
        n_verts: int,
        n_faces: int,
        faces_np: np.ndarray | None,
        verts_np: np.ndarray,
        type_cfg: dict[str, Any],
        visual_type: str,
        epoch: int,
    ) -> np.ndarray:
        if scalars is None:
            return solid_rgb(n_verts, _DARK_BLUE_RGB)

        sv = np.asarray(scalars, dtype=np.float32).reshape(-1)

        if sv.shape[0] == n_verts:
            vertex_scalars = sv
        elif sv.shape[0] == n_faces and faces_np is not None:
            vertex_scalars = face_scalars_to_vertex(sv, faces_np, verts_np)
        else:
            _log({"event": "scalar_shape_mismatch", "run_id": self.run_id,
                  "visual_type": visual_type, "epoch": epoch,
                  "scalar_shape": list(sv.shape), "n_verts": n_verts, "n_faces": n_faces})
            return solid_rgb(n_verts, _DARK_BLUE_RGB)

        colormap: str | None = type_cfg.get("colormap")
        if colormap is None:
            return solid_rgb(n_verts, _DARK_BLUE_RGB)

        return map_scalar_to_rgb_mpl(vertex_scalars, colormap, vmin=type_cfg.get("vmin"), vmax=type_cfg.get("vmax"))

    # --- Submit helpers (call from named methods) ---------------------------

    def _submit_ply(
        self,
        visual_type: str,
        epoch: int,
        verts_np: np.ndarray,
        faces_np: np.ndarray | None,
        rgb_np: np.ndarray,
        type_cfg: dict[str, Any],
    ) -> None:
        add_coord = self._resolve_bool(type_cfg.get("coordinate_system"), self._extras.get("coordinate_system"))
        coord_scale = float(self._extras.get("coordinate_system_scale", 0.1))
        out_path = self._output_dir(visual_type) / f"epoch_{epoch:05d}.ply"
        future = self._ply_pool.submit(_write_ply_task, out_path, verts_np, faces_np, rgb_np, add_coord, coord_scale)
        self._pending_ply.append(future)
        _log({"event": "ply_submitted", "run_id": self.run_id, "visual_type": visual_type, "epoch": epoch})

    def _submit_ply_points(
        self,
        visual_type: str,
        epoch: int,
        points: Any,
        secondary_points: Any | None = None,
        tertiary_points: Any | None = None,
        mesh_verts: np.ndarray | None = None,
        mesh_faces: np.ndarray | None = None,
    ) -> None:
        out_dir = self._output_dir(visual_type)

        def _task() -> tuple[bool, str, str | None]:
            try:
                from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.points_SDFmesh_to_ply import (  # noqa: PLC0415
                    SamplingVisualizer,
                )

                sv = SamplingVisualizer(output_dir=out_dir, is_static=False)
                path = sv.save_snapshot(
                    points_mm=points,
                    name=f"epoch_{epoch:05d}",
                    secondary_points_mm=secondary_points,
                    tertiary_points_mm=tertiary_points,
                    mesh_verts=mesh_verts,
                    mesh_faces=mesh_faces,
                )
                return (True, str(path) if path else "", None)
            except Exception as exc:
                return (False, str(out_dir), str(exc))

        future = self._ply_pool.submit(_task)
        self._pending_ply.append(future)
        _log({"event": "ply_points_submitted", "run_id": self.run_id, "visual_type": visual_type, "epoch": epoch})

    # --- Lifecycle ----------------------------------------------------------

    def finalize(self) -> None:
        """Drain pending PLY writes then generate convergence GIFs. Idempotent."""
        if self._finalized:
            return
        self._finalized = True

        _log({"event": "finalize_started", "run_id": self.run_id, "pending_ply": len(self._pending_ply)})

        done, _ = concurrent.futures.wait(self._pending_ply) if self._pending_ply else (set(), set())
        for f in done:
            ok, path, err = f.result()
            if not ok:
                _log({"event": "ply_write_failed", "run_id": self.run_id, "path": path, "error": err})
        self._ply_pool.shutdown(wait=False)

        flat_pattern = re.compile(r"^epoch_(\d{5})\.ply$")

        def _collect_enabled(node: dict[str, Any], prefix: str) -> list[str]:
            types: list[str] = []
            for key, val in node.items():
                path = f"{prefix}/{key}" if prefix else key
                if isinstance(val, dict):
                    if "enabled" in val:
                        if val.get("enabled", False):
                            types.append(path)
                    else:
                        types.extend(_collect_enabled(val, path))
            return types

        all_dirs = [self.run_dir / vt for vt in _collect_enabled(self._visual_types, "")]
        for d in self._written_dirs:
            if d not in all_dirs:
                all_dirs.append(d)

        gif_dirs = [d for d in all_dirs if d.is_dir() and any(flat_pattern.match(f.name) for f in d.iterdir())]
        if not gif_dirs:
            _log({"event": "finalize_completed", "run_id": self.run_id, "gif_count": 0})
            return

        gif_workers: int = self._cfg.get("gif", {}).get("pool_workers", 1)
        gif_futures: list[concurrent.futures.Future[tuple[bool, str, str | None]]] = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=gif_workers) as pool:
            for d in gif_dirs:
                n = sum(1 for f in d.iterdir() if flat_pattern.match(f.name))
                _log({"event": "gif_submitted", "run_id": self.run_id,
                      "visual_type": str(d.relative_to(self.run_dir)), "n_frames": n})
                gif_futures.append(pool.submit(_render_gif_task, d))

            for f in concurrent.futures.as_completed(gif_futures):
                ok, path, err = f.result()
                _log({"event": "gif_completed" if ok else "gif_failed",
                      "run_id": self.run_id, "path": path, "error": err})

        _log({"event": "finalize_completed", "run_id": self.run_id, "gif_count": len(gif_dirs)})

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.finalize()


# ---------------------------------------------------------------------------
# Concrete visualization manager — add named methods here
# ---------------------------------------------------------------------------


class CentralVisManagerV2(_BaseVisManager):
    """Named visualization methods for the AISCO training loop.

    Each method:
      1. Returns immediately if the visual type is disabled in viz_config.yaml.
      2. Converts data to numpy on the calling thread (safe to do while the
         training loop still owns the tensors).
      3. Submits the actual file write fire-and-forget to the thread pool.
    """

    def vis_mesh_shape(self, epoch: int, verts: Any, faces: Any) -> None:
        vt = self._slotted("mesh_shape")
        type_cfg = self._type_config(vt)
        if type_cfg is None:
            return
        verts_np = _to_f64(verts)
        self._submit_ply(vt, epoch, verts_np, _to_i32(faces), solid_rgb(len(verts_np), _DARK_BLUE_RGB), type_cfg)

    def vis_aero_cp_field(self, epoch: int, verts: Any, faces: Any, cp_values: Any) -> None:
        vt = self._slotted("aero/cp_field")
        type_cfg = self._type_config(vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        sv = np.asarray(cp_values, dtype=np.float32).reshape(-1)
        rgb = self._resolve_rgb(sv, len(verts_np), len(faces_np), faces_np, verts_np, type_cfg, vt, epoch)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_solar_selection(self, epoch: int, verts: Any, faces: Any, weights: Any) -> None:
        vt = self._slotted("solar/selection")
        type_cfg = self._type_config(vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        sv = np.asarray(weights, dtype=np.float32).reshape(-1)
        rgb = self._resolve_rgb(sv, len(verts_np), len(faces_np), faces_np, verts_np, type_cfg, vt, epoch)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_solar_shading(self, epoch: int, verts: Any, faces: Any, shading: Any) -> None:
        vt = self._slotted("solar/shading")
        type_cfg = self._type_config(vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        sv = np.asarray(shading, dtype=np.float32).reshape(-1)
        rgb = self._resolve_rgb(sv, len(verts_np), len(faces_np), faces_np, verts_np, type_cfg, vt, epoch)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_solar_penalty(self, epoch: int, verts: Any, faces: Any, penalty: Any) -> None:
        vt = self._slotted("solar/penalty")
        type_cfg = self._type_config(vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        sv = np.asarray(penalty, dtype=np.float32).reshape(-1)
        rgb = self._resolve_rgb(sv, len(verts_np), len(faces_np), faces_np, verts_np, type_cfg, vt, epoch)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_bb_constraint(self, epoch: int, verts: Any, distances: Any) -> None:
        """Bounding-box constraint violations. verts is a point cloud (no faces)."""
        vt = self._slotted("bb_constraint")
        type_cfg = self._type_config(vt)
        if type_cfg is None:
            return
        verts_np = _to_f64(verts)
        sv = np.asarray(distances, dtype=np.float32).reshape(-1)
        rgb = self._resolve_rgb(sv, len(verts_np), 0, None, verts_np, type_cfg, vt, epoch)
        self._submit_ply(vt, epoch, verts_np, None, rgb, type_cfg)

    def vis_latent_sampling(
        self,
        epoch: int,
        points: Any,
        slot_idx: int | None = None,
        slot_name: str = "",
        secondary_points: Any | None = None,
        tertiary_points: Any | None = None,
        mesh_verts: np.ndarray | None = None,
        mesh_faces: np.ndarray | None = None,
    ) -> None:
        """Latent-space point-cloud snapshot.

        slot_idx/slot_name come from the pretraining loop and set the output
        sub-path directly (independent of set_active_slot, which is for the
        diversity population loop).
        """
        vt = f"latent/sampling/slot{slot_idx:02d}_{slot_name}" if slot_idx is not None else "latent/sampling"
        if not self._is_enabled(vt):
            return
        self._submit_ply_points(vt, epoch, points, secondary_points, tertiary_points, mesh_verts, mesh_faces)

    def vis_mesh_assembly_v4(
        self,
        epoch: int,
        sdf_verts: Any,
        sdf_faces: Any,
    ) -> None:
        """Export one epoch of the full mesh assembly as a combined PLY and named-component GLB.

        Call ``set_assembly_meshes`` once before training to supply the constant
        inner/outer bounding-box meshes and colours.  This method only needs the
        evolving SDF mesh, which changes every epoch.

        PLY goes to  <run_dir>/mesh_assembly_v4/epoch_XXXXX.ply  (for GIF generation).
        GLB goes to  <run_dir>/mesh_assembly_v4/epoch_XXXXX.glb  (for the three.js viewer).

        Args:
            epoch: Current training epoch.
            sdf_verts / sdf_faces: The optimised SDF mesh for this epoch.
        """
        vt = self._slotted("mesh_assembly_v4")
        if not self._is_enabled(vt):
            return

        if self._assembly_meshes is None:
            _log({"event": "assembly_meshes_missing", "run_id": self.run_id,
                  "message": "vis_mesh_assembly_v4 called before assembly meshes were loaded — skipping."})
            return

        inn_v, inn_f, out_v, out_f, sdf_color, inner_color, outer_color, sdf_alpha, outer_alpha = (
            self._assembly_meshes
        )

        import torch  # noqa: PLC0415
        sdf_v = sdf_verts.detach().cpu() if isinstance(sdf_verts, torch.Tensor) else torch.from_numpy(np.asarray(sdf_verts))
        sdf_f = sdf_faces.detach().cpu() if isinstance(sdf_faces, torch.Tensor) else torch.from_numpy(np.asarray(sdf_faces))

        out_dir = self._output_dir(vt)
        ply_path = out_dir / f"epoch_{epoch:05d}.ply"
        glb_path = out_dir / f"epoch_{epoch:05d}.glb"

        def _task() -> tuple[bool, str, str | None]:
            try:
                from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.file_io import (  # noqa: PLC0415
                    save_scene_glb,
                )
                from subprojects.wp4_aeroshell_optimizer.v2.src.utils.combined_ply_export import (  # noqa: PLC0415
                    export_combined_bbox_mesh_ply,
                )

                export_combined_bbox_mesh_ply(
                    inner_verts=inn_v, inner_faces=inn_f,
                    outer_verts=out_v, outer_faces=out_f,
                    sdf_verts=sdf_v, sdf_faces=sdf_f,
                    output_path=ply_path,
                    inner_color=inner_color, outer_color=outer_color, sdf_color=sdf_color,
                    inner_alpha=1.0, outer_alpha=outer_alpha, sdf_alpha=sdf_alpha,
                )

                sdf_rgb = np.tile(np.array(sdf_color, dtype=np.float32), (len(sdf_v), 1))
                inn_rgb = np.tile(np.array(inner_color, dtype=np.float32), (len(inn_v), 1))
                out_rgb = np.tile(np.array(outer_color, dtype=np.float32), (len(out_v), 1))
                save_scene_glb(glb_path, [
                    ("sdf_mesh",       sdf_v.numpy(), sdf_f.numpy(), sdf_rgb),
                    ("inner_assembly", inn_v.numpy(), inn_f.numpy(), inn_rgb),
                    ("outer_bbox",     out_v.numpy(), out_f.numpy(), out_rgb),
                ])
                return (True, str(ply_path), None)
            except Exception as exc:
                return (False, str(ply_path), str(exc))

        future = self._ply_pool.submit(_task)
        self._pending_ply.append(future)
        _log({"event": "ply_submitted", "run_id": self.run_id, "visual_type": vt, "epoch": epoch})

    # --- Backward-compat shims (old callers still use save_ply; log and drop) -

    def save_ply(self, visual_type: str, epoch: int, *_args: Any, **_kwargs: Any) -> None:
        _log({"event": "deprecated_save_ply", "run_id": self.run_id,
              "visual_type": visual_type, "epoch": epoch,
              "message": f"Caller still uses save_ply('{visual_type}', ...) — migrate to the named vis_* method."})

    def save_ply_points(self, visual_type: str, epoch: int, *_args: Any, **_kwargs: Any) -> None:
        _log({"event": "deprecated_save_ply_points", "run_id": self.run_id,
              "visual_type": visual_type, "epoch": epoch,
              "message": f"Caller still uses save_ply_points('{visual_type}', ...) — migrate to vis_latent_sampling()."})

