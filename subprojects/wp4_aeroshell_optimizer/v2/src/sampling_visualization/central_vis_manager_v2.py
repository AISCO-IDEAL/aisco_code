"""CentralVisManagerV2 — async PLY writes + named visualization methods.

_BaseVisManager owns the infrastructure: thread pool for PLY writes, process pool
for GIF generation in finalize(), crash-safe shutdown via atexit/__del__, and the
public setup API (set_active_slot, set_assembly_meshes, make_vis_extractor).

CentralVisManagerV2(_BaseVisManager) adds one named vis_* method per visual type.
To add a new visual: subclass or extend CentralVisManagerV2 and add a method.

Switch via ``vis_manager: v2`` in general_config.yaml.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.assembly_loader import (
    load_assembly_meshes,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.coloring import (
    resolve_mesh_rgb,
    solid_rgb,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.ply_tasks import (
    render_gif_task,
    write_ply_task,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.utils.vis_config_helpers import (
    collect_gif_dirs,
    resolve_bool,
    vis_is_enabled,
    vis_type_config,
)
from subprojects.wp5.unified_mesh_extraction.sdf2mesh import MeshSDFExtractor

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "viz_config.yaml"
_DARK_BLUE_RGB: tuple[float, float, float] = (0.0, 0.0, 139.0 / 255.0)


def _log(event: dict[str, Any]) -> None:
    sys.stderr.write(json.dumps(event) + "\n")


def _to_f64(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def _to_i32(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.int32)
    return np.asarray(x, dtype=np.int32)


# ---------------------------------------------------------------------------
# Base class — infrastructure only
# ---------------------------------------------------------------------------


class _BaseVisManager:
    """Async PLY write infrastructure. Subclass and add named vis_* methods."""

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

    def is_enabled(self, visual_type: str) -> bool:
        """Check if a visual type is enabled in the config, with ancestor lookup.

        E.g. 'mesh_shape/slot_00' inherits the 'mesh_shape' config.
        """
        return vis_is_enabled(self._visual_types, self._slotted(visual_type))

    def _load_assembly_meshes(self) -> None:
        if not vis_is_enabled(self._visual_types, "mesh_assembly_v4"):
            return
        try:
            self._assembly_meshes = load_assembly_meshes()
            _log({"event": "assembly_meshes_loaded", "run_id": self.run_id})
        except Exception as exc:
            _log(
                {
                    "event": "assembly_meshes_load_failed",
                    "run_id": self.run_id,
                    "error": str(exc),
                    "message": "Call set_assembly_meshes() manually if the assembly cannot be loaded automatically.",
                }
            )

    def _slotted(self, visual_type: str) -> str:
        if self._active_slot is not None:
            return f"{visual_type}/slot_{self._active_slot:02d}"
        return visual_type

    def _output_dir(self, visual_type: str) -> Path:
        out_dir = self.run_dir / visual_type
        out_dir.mkdir(parents=True, exist_ok=True)
        self._written_dirs.add(out_dir)
        return out_dir

    def _submit_ply(
        self,
        visual_type: str,
        epoch: int,
        verts_np: np.ndarray,
        faces_np: np.ndarray | None,
        rgb_np: np.ndarray,
        type_cfg: dict[str, Any],
    ) -> None:
        add_coord = resolve_bool(type_cfg.get("coordinate_system"), self._extras.get("coordinate_system"))
        coord_scale = float(self._extras.get("coordinate_system_scale", 0.1))
        out_path = self._output_dir(visual_type) / f"epoch_{epoch:05d}.ply"
        future = self._ply_pool.submit(write_ply_task, out_path, verts_np, faces_np, rgb_np, add_coord, coord_scale)
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

        self._pending_ply.append(self._ply_pool.submit(_task))
        _log({"event": "ply_points_submitted", "run_id": self.run_id, "visual_type": visual_type, "epoch": epoch})

    # --- Public API ---------------------------------------------------------

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
        lookup in ``vis_type_config``.

        Call ``set_active_slot(None)`` after the population loop to return to
        non-slotted writes.

        Args:
            slot_idx: Zero-based population index, or None to clear.
        """
        self._active_slot = slot_idx

    def set_assembly_meshes(
        self,
        inner_verts: Any,
        inner_faces: Any,
        outer_verts: Any,
        outer_faces: Any,
        *,
        sdf_color: tuple[float, float, float] = (0.2, 1.0, 1.0),
        inner_color: tuple[float, float, float] = (1.0, 0.5, 0.0),
        outer_color: tuple[float, float, float] = (1.0, 0.5, 0.0),
        sdf_alpha: float = 0.5,
        outer_alpha: float = 0.2,
    ) -> None:
        """Store constant bbox meshes and display colours for vis_mesh_assembly_v4.

        Called automatically from __init__ via load_assembly_meshes(); call this
        manually to override colours or supply meshes when WP2 is unavailable.

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
            return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x))

        self._assembly_meshes = (
            _cpu(inner_verts),
            _cpu(inner_faces),
            _cpu(outer_verts),
            _cpu(outer_faces),
            sdf_color,
            inner_color,
            outer_color,
            sdf_alpha,
            outer_alpha,
        )
        _log({"event": "assembly_meshes_set", "run_id": self.run_id})

    def make_vis_extractor(self, device: str = "cuda") -> MeshSDFExtractor:
        """Create a low-resolution MeshSDFExtractor for visualization.

        The bbox is derived from the outer-prism regulations, scaled by
        ``vis_extraction.bbox_scale_factor`` (default 1.3).  Resolution is read
        from ``vis_extraction.resolution`` (default 64).  All other extraction
        parameters inherit from the unified extractor's own YAML config.

        Args:
            device: Torch device for grid evaluation.

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
        bbox_scale = float(vis_cfg.get("bbox_scale_factor", 1.3))

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

        gif_dirs = collect_gif_dirs(self._visual_types, self.run_dir, self._written_dirs)
        if not gif_dirs:
            _log({"event": "finalize_completed", "run_id": self.run_id, "gif_count": 0})
            return

        for d in gif_dirs:
            n = sum(1 for f in d.iterdir() if f.name.startswith("epoch_") and f.name.endswith(".ply"))
            _log({"event": "gif_started", "run_id": self.run_id, "visual_type": str(d.relative_to(self.run_dir)), "n_frames": n})
            ok, path, err = render_gif_task(d)
            _log({"event": "gif_completed" if ok else "gif_failed", "run_id": self.run_id, "path": path, "error": err})

        _log({"event": "finalize_completed", "run_id": self.run_id, "gif_count": len(gif_dirs)})

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.finalize()


# ---------------------------------------------------------------------------
# Concrete visualization manager — add named methods here
# ---------------------------------------------------------------------------


class CentralVisManagerV2(_BaseVisManager):
    """Named visualization methods for the AISCO training loop.

    Each method checks if the visual type is enabled in viz_config.yaml,
    converts data to numpy on the calling thread, and submits file writes
    fire-and-forget to the thread pool.
    """

    def vis_mesh_shape(self, epoch: int, verts: Any, faces: Any) -> None:
        vt = self._slotted("mesh_shape")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np = _to_f64(verts)
        self._submit_ply(vt, epoch, verts_np, _to_i32(faces), solid_rgb(len(verts_np), _DARK_BLUE_RGB), type_cfg)

    def vis_aero_cp_field(self, epoch: int, verts: Any, faces: Any, cp_values: Any) -> None:
        vt = self._slotted("aero/cp_field")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        cp_np = _to_f64(cp_values).astype(np.float32).reshape(-1)
        rgb = resolve_mesh_rgb(cp_np, len(verts_np), len(faces_np), faces_np, verts_np, type_cfg)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)
    
    def vis_solar_quality(self, epoch: int, verts: Any, faces: Any, quality_values: Any) -> None:
        
        vt = self._slotted("solar/quality")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        quality_np = _to_f64(quality_values).astype(np.float32).reshape(-1)
        #scale quality from -1,1 to 0,1 for better colormap resolution
        quality_np = (quality_np + 1.0) / 2.0
        rgb = resolve_mesh_rgb(quality_np, len(verts_np), len(faces_np), faces_np, verts_np, type_cfg)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    


    def vis_solar_selection(self, epoch: int, verts: Any, faces: Any, weights: Any) -> None:
        vt = self._slotted("solar/selection")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        rgb = resolve_mesh_rgb(
            np.asarray(_to_f64(weights), np.float32).reshape(-1), len(verts_np), len(faces_np), faces_np, verts_np, type_cfg
        )
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_solar_shading(self, epoch: int, verts: Any, faces: Any, shading: Any) -> None:
        vt = self._slotted("solar/shading")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        rgb = resolve_mesh_rgb(
            np.asarray(_to_f64(shading), np.float32).reshape(-1), len(verts_np), len(faces_np), faces_np, verts_np, type_cfg
        )
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_solar_penalty(self, epoch: int, verts: Any, faces: Any, penalty: Any) -> None:
        vt = self._slotted("solar/penalty")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np, faces_np = _to_f64(verts), _to_i32(faces)
        rgb = resolve_mesh_rgb(
            np.asarray(_to_f64(penalty), np.float32).reshape(-1), len(verts_np), len(faces_np), faces_np, verts_np, type_cfg
        )
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

    def vis_bb_constraint(self, epoch: int, verts: Any, distances: Any) -> None:
        """Bounding-box constraint violations. verts is a point cloud (no faces)."""
        vt = self._slotted("bb_constraint")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return
        verts_np = _to_f64(verts)
        rgb = resolve_mesh_rgb(np.asarray(distances, np.float32).reshape(-1), len(verts_np), 0, None, verts_np, type_cfg)
        self._submit_ply(vt, epoch, verts_np, None, rgb, type_cfg)

    def vis_mesh_gradient(
        self,
        epoch: int,
        loss_key: str,
        verts: Any,
        faces: Any,
        dL_dv: Any,
    ) -> None:
        """Render per-vertex `dL/dv · n` for one loss as a coloured PLY.

        The per-vertex outward normal is computed from face normals; the dot
        product with `dL/dv` is normalised to a symmetric range and mapped
        through the configured diverging colormap (vmin/vmax are inferred
        per-epoch by default — see ``visual_types.mesh_gradients``).

        Args:
            epoch:    Training epoch index — embedded in the PLY filename.
            loss_key: Name of the loss whose gradient is being visualised.
                      Becomes the sub-folder name under ``mesh_gradients/``.
            verts:    (V, 3) mesh vertices (tensor or array).
            faces:    (F, 3) triangle indices (tensor or array).
            dL_dv:    (V, 3) gradient of the loss w.r.t. `verts`.
        """
        vt = self._slotted(f"mesh_gradients/{loss_key}")
        type_cfg = vis_type_config(self._visual_types, vt)
        if type_cfg is None:
            return

        verts_np = _to_f64(verts)
        faces_np = _to_i32(faces)
        grads_np = _to_f64(dL_dv)
        if verts_np.shape[0] == 0 or faces_np.shape[0] == 0:
            return
        if grads_np.shape != verts_np.shape:
            _log(
                {
                    "event": "vis_mesh_gradient_shape_mismatch",
                    "run_id": self.run_id,
                    "loss_key": loss_key,
                    "verts": list(verts_np.shape),
                    "grads": list(grads_np.shape),
                }
            )
            return

        # Vertex normals = sum of incident face normals (no need to weight by
        # area; the projection onto dL/dv only needs a direction).
        e1 = verts_np[faces_np[:, 1]] - verts_np[faces_np[:, 0]]
        e2 = verts_np[faces_np[:, 2]] - verts_np[faces_np[:, 0]]
        face_n = np.cross(e1, e2)
        face_n_norm = np.linalg.norm(face_n, axis=1, keepdims=True)
        face_n /= np.maximum(face_n_norm, 1e-12)
        vert_n = np.zeros_like(verts_np)
        np.add.at(vert_n, faces_np[:, 0], face_n)
        np.add.at(vert_n, faces_np[:, 1], face_n)
        np.add.at(vert_n, faces_np[:, 2], face_n)
        vert_n_norm = np.linalg.norm(vert_n, axis=1, keepdims=True)
        vert_n /= np.maximum(vert_n_norm, 1e-12)

        dot = np.einsum("ij,ij->i", vert_n, grads_np).astype(np.float32)
        dot = np.nan_to_num(dot, nan=0.0, posinf=0.0, neginf=0.0)

        # Symmetric range around zero so a diverging colormap (e.g. RdBu_r,
        # coolwarm) is centred on "no force". vmin/vmax in the YAML override
        # this; null in YAML means "use the data-driven symmetric range".
        cfg_vmin = type_cfg.get("vmin")
        cfg_vmax = type_cfg.get("vmax")
        if cfg_vmin is None or cfg_vmax is None:
            amp = float(np.max(np.abs(dot))) if dot.size else 0.0
            amp = amp if amp > 0.0 else 1.0
            vmin_eff = -amp if cfg_vmin is None else float(cfg_vmin)
            vmax_eff = amp if cfg_vmax is None else float(cfg_vmax)
        else:
            vmin_eff, vmax_eff = float(cfg_vmin), float(cfg_vmax)

        # Build an effective per-type config with the resolved bounds so
        # resolve_mesh_rgb honours them without mutating the cached YAML dict.
        eff_cfg = dict(type_cfg)
        eff_cfg["vmin"] = vmin_eff
        eff_cfg["vmax"] = vmax_eff

        rgb = resolve_mesh_rgb(dot, len(verts_np), len(faces_np), faces_np, verts_np, eff_cfg)
        self._submit_ply(vt, epoch, verts_np, faces_np, rgb, type_cfg)

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
        sub-path directly (independent of set_active_slot for the diversity loop).
        """
        vt = f"latent/sampling/slot{slot_idx:02d}_{slot_name}" if slot_idx is not None else "latent/sampling"
        if not vis_is_enabled(self._visual_types, vt):
            return
        self._submit_ply_points(vt, epoch, points, secondary_points, tertiary_points, mesh_verts, mesh_faces)

    def vis_mesh_assembly_v4(self, epoch: int, sdf_verts: Any, sdf_faces: Any) -> None:
        """Export one epoch of the full mesh assembly as a combined PLY and named-component GLB.

        Requires set_assembly_meshes() to have been called (happens automatically
        from __init__ when mesh_assembly_v4 is enabled in viz_config.yaml).

        PLY goes to  <run_dir>/mesh_assembly_v4/epoch_XXXXX.ply  (for GIF generation).
        GLB goes to  <run_dir>/mesh_assembly_v4/epoch_XXXXX.glb  (for the three.js viewer).

        Args:
            epoch: Current training epoch.
            sdf_verts / sdf_faces: The optimised SDF mesh for this epoch.
        """
        vt = self._slotted("mesh_assembly_v4")
        if not vis_is_enabled(self._visual_types, vt):
            return

        if self._assembly_meshes is None:
            _log(
                {
                    "event": "assembly_meshes_missing",
                    "run_id": self.run_id,
                    "message": "vis_mesh_assembly_v4 called before assembly meshes were loaded — skipping.",
                }
            )
            return

        inn_v, inn_f, out_v, out_f, sdf_color, inner_color, outer_color, sdf_alpha, outer_alpha = self._assembly_meshes

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
                    inner_verts=inn_v,
                    inner_faces=inn_f,
                    outer_verts=out_v,
                    outer_faces=out_f,
                    sdf_verts=sdf_v,
                    sdf_faces=sdf_f,
                    output_path=ply_path,
                    inner_color=inner_color,
                    outer_color=outer_color,
                    sdf_color=sdf_color,
                    inner_alpha=1.0,
                    outer_alpha=outer_alpha,
                    sdf_alpha=sdf_alpha,
                )
                sdf_rgb = np.tile(np.array(sdf_color, dtype=np.float32), (len(sdf_v), 1))
                inn_rgb = np.tile(np.array(inner_color, dtype=np.float32), (len(inn_v), 1))
                out_rgb = np.tile(np.array(outer_color, dtype=np.float32), (len(out_v), 1))
                save_scene_glb(
                    glb_path,
                    [
                        ("sdf_mesh", sdf_v.numpy(), sdf_f.numpy(), sdf_rgb),
                        ("inner_assembly", inn_v.numpy(), inn_f.numpy(), inn_rgb),
                        ("outer_bbox", out_v.numpy(), out_f.numpy(), out_rgb),
                    ],
                )
                return (True, str(ply_path), None)
            except Exception as exc:
                return (False, str(ply_path), str(exc))

        self._pending_ply.append(self._ply_pool.submit(_task))
        _log({"event": "ply_submitted", "run_id": self.run_id, "visual_type": vt, "epoch": epoch})

    # --- Backward-compat shims (old callers; log and drop) ------------------

    def save_ply(self, visual_type: str, epoch: int, *_args: Any, **_kwargs: Any) -> None:
        _log(
            {
                "event": "deprecated_save_ply",
                "run_id": self.run_id,
                "visual_type": visual_type,
                "epoch": epoch,
                "message": f"Caller still uses save_ply('{visual_type}', ...) — migrate to the named vis_* method.",
            }
        )

    def save_ply_points(self, visual_type: str, epoch: int, *_args: Any, **_kwargs: Any) -> None:
        _log(
            {
                "event": "deprecated_save_ply_points",
                "run_id": self.run_id,
                "visual_type": visual_type,
                "epoch": epoch,
                "message": f"Caller still uses save_ply_points('{visual_type}', ...) — migrate to vis_latent_sampling().",
            }
        )
