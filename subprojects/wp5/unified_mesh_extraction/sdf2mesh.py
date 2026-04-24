"""Minimal SDF->mesh extractor with differentiable surrogate and cross-loss mesh cache."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import mcubes as _mcubes
import numpy as np
import torch

from subprojects.wp5.unified_mesh_extraction.cache import (
    _MESH_CACHE,
    _MESH_CACHE_MAX_ENTRIES,
    clear_mesh_cache,  # noqa: F401 — re-exported for backward compat
)
from subprojects.wp5.unified_mesh_extraction.utils import (
    MeshExtractionConfig,
    _model_version,
    _normalise_bbox,
    load_mesh_extraction_config,
)

try:
    from fast_simplification import simplify as _fast_simplify
except Exception:  # pragma: no cover
    _fast_simplify = None

SDFModel = Callable[[torch.Tensor], torch.Tensor]
_LOGGER = logging.getLogger("sdf2mesh")


class MeshSDFExtractor:
    """SDF -> triangle mesh via marching cubes, with differentiable surrogate.

    Speed: sparse coarse-to-fine marching cubes evaluates fine voxels only near
    the zero-crossing. Memory: fp16 SDF eval on CUDA. Mesh reuse: a global
    cache keyed on (model id, param version, bbox, resolution, isovalue,
    target vertex count) lets multiple losses share the same extraction.
    """

    def __init__(self, config: MeshExtractionConfig | None = None, gradient_max: None | float = None) -> None:
        """Initialise an extractor bound to a fixed bbox/resolution.

        Steps:
          1) Use the provided `config` or fall back to YAML defaults.
          2) Downgrade `device` to CPU if CUDA was requested but is unavailable.
          3) Cache hot config fields (isovalue, batch_size, target vertex count).
          4) Enable fp16 autocast when running on CUDA for memory-efficient SDF eval.
          5) Initialise slots for the cross-loss shared-mesh injection API.
          6) Call `set_grid` to compute the effective grid resolution.
        """
        self.gradient_max = gradient_max
        self.config = config if config is not None else load_mesh_extraction_config()
        self.device = self.config.device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            _LOGGER.warning("CUDA unavailable, falling back to CPU.")
            self.device = "cpu"
        self.isovalue = float(self.config.isovalue)
        self.batch_size = int(self.config.batch_size)
        self.target_num_vertices = self.config.target_num_vertices
        self.vertex_count_tolerance = float(self.config.vertex_count_tolerance)
        self._use_amp = self.device.startswith("cuda")
        self._cached_resolution: int | None = None
        self._shared_vertices: torch.Tensor | None = None
        self._shared_faces: torch.Tensor | None = None
        self._stats: dict[str, int | float] = {"calls": 0}
        # `max_refine_levels` is treated as a resolution multiplier on the
        # uniform grid; it yields equivalent quality to adaptive refinement
        # at a fraction of the complexity.
        self.set_grid(self.config.bbox, self.config.base_resolution)

    def set_grid(self, bbox: Any, resolution: int) -> None:
        """Update bbox and base resolution.

        Steps:
          1) Normalise the bbox to `[[xlo,xhi],[ylo,yhi],[zlo,zhi]]`.
          2) Compute `effective_resolution = base_resolution * subdiv**refine`,
             clamped to [2, 512]. This replaces adaptive octree refinement with
             an equivalent uniform-grid multiplier (simpler and faster).
          3) Invalidate the warm-started resolution used by `_extract_to_target`.
        """
        self.bbox = _normalise_bbox(bbox)
        self.base_resolution = int(resolution)
        subdiv = max(2, int(self.config.subdivision_factor))
        refine = max(0, int(self.config.max_refine_levels))
        self.effective_resolution = min(512, max(2, self.base_resolution * (subdiv**refine)))
        self._cached_resolution = None

    def extract_mesh(self, sdf_model: SDFModel) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract a mesh from `sdf_model`, reusing cached result when possible.

        Steps:
          1) Fast path: if `TotalLoss` injected a shared mesh via
             `set_shared_mesh`, return it verbatim (zero re-extraction).
          2) Move the model to `self.device` if it is an `nn.Module`.
          3) Build a cache key from `(id(model), param_version, bbox, resolution,
             isovalue, target vertex count, tolerance, device)`.
          4) Return immediately on cache hit.
          5) On miss, purge stale entries for the same model whose param
             version has bumped (keeps the cache small and coherent).
          6) Run either the fixed-resolution `_extract` or the target-count
             adaptive `_extract_to_target`, depending on config.
          7) FIFO-evict if the cache is full, then store the new (v, f) pair.
        """
        # Instance-level shared mesh injected by cross-loss sharing (TotalLoss)
        if self._shared_vertices is not None and self._shared_faces is not None:
            return self._shared_vertices, self._shared_faces

        if isinstance(sdf_model, torch.nn.Module):
            sdf_model = sdf_model.to(self.device)

        model_id = id(sdf_model)
        version = _model_version(sdf_model)
        key = (
            model_id,
            version,
            tuple(tuple(ax) for ax in self.bbox),
            self.effective_resolution,
            self.isovalue,
            self.target_num_vertices,
            round(self.vertex_count_tolerance, 4),
            str(torch.device(self.device)),
        )
        hit = _MESH_CACHE.get(key)
        if hit is not None:
            return hit

        for k in [k for k in _MESH_CACHE if k[0] == model_id and k[1] != version]:
            _MESH_CACHE.pop(k, None)

        if self.target_num_vertices is None:
            verts, faces = self._extract(sdf_model, self.effective_resolution)
        else:
            verts, faces = self._extract_to_target(sdf_model)

        if len(_MESH_CACHE) >= _MESH_CACHE_MAX_ENTRIES:
            _MESH_CACHE.pop(next(iter(_MESH_CACHE)))
        _MESH_CACHE[key] = (verts, faces)
        return verts, faces

    def extract_and_compute_loss(
        self,
        sdf_model: SDFModel,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        enable_backward: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract mesh, compute `loss_fn(v, f)`, and a surrogate whose backward
        pass routes gradients into the SDF parameters.

        Returns `(vertices, faces, loss, surrogate_loss, dL_dvertices)`.

        Steps:
          1) Extract mesh via the cached `extract_mesh`.
          2) If empty, return a large penalty loss (prevents NaN training steps).
          3) Detach vertices and re-enable grad so `loss_fn` can backprop to them.
          4) Evaluate `loss = loss_fn(v, f)`.
          5) If `enable_backward=False`, return zero surrogate (inference only).
          6) Otherwise compute `dL/dv` with `torch.autograd.grad`, then build the
             surrogate loss via `compute_gradient_flow` (normal-direction
             projection). Its `.backward()` yields correct gradients w.r.t. the
             SDF parameters even though marching cubes itself is not differentiable.
        """
        verts, faces = self.extract_mesh(sdf_model)
        if verts.numel() == 0 or faces.numel() == 0:
            _LOGGER.warning("Empty mesh; returning penalty loss.")
            ev = torch.zeros((1, 3), device=self.device)
            ef = torch.zeros((1, 3), dtype=torch.int64, device=self.device)
            return (
                ev,
                ef,
                torch.tensor(1e4, device=self.device),
                torch.zeros((), device=self.device, requires_grad=True),
                torch.zeros_like(ev),
            )

        v_for_loss = verts.detach().requires_grad_(True)
        loss = loss_fn(v_for_loss, faces)
        if not enable_backward:
            return verts, faces, loss, torch.zeros((), device=self.device), torch.zeros_like(verts)

        (dL_dv,) = torch.autograd.grad(loss, v_for_loss, torch.ones_like(loss))
        surrogate = self.compute_gradient_flow(verts, dL_dv, sdf_model)

        if self.gradient_max is not None:
            norms = torch.norm(dL_dv, dim=1, keepdim=True)
            scale = self.gradient_max / (norms + 1e-8)
            scale = torch.minimum(scale, torch.ones_like(scale))
            dL_dv = dL_dv * scale

        return verts, faces, loss, surrogate, dL_dv

    def compute_gradient_flow(self, vertices: torch.Tensor, dL_dvertices: torch.Tensor, sdf_model: SDFModel) -> torch.Tensor:
        """Build a scalar whose grad w.r.t. SDF params equals the normal-direction projection of dL/dv.

        Background: for an implicit surface {s(x; theta)=c}, moving vertex v
        along the SDF normal n(v) by epsilon changes s(v) by epsilon. Hence
        `dL/dtheta = sum_v -<dL/dv, n(v)> * d s(v)/dtheta`. This function
        constructs that scalar so that `.backward()` produces the theta gradient.

        Steps:
          1) Re-evaluate `s = sdf_model(v)` with grad to get the per-vertex
             SDF gradient `g = d s/dv`.
          2) Normalise `g` to unit-length surface normals `n`.
          3) Project `dL/dv` onto `-n` to obtain the scalar `dL/ds` per vertex
             (detached; it acts as a fixed coefficient).
          4) Re-evaluate `s2 = sdf_model(v)` (fresh graph, no create_graph)
             so that `theta` still participates in autograd.
          5) Return `sum(dL_ds * s2)`: its backward is exactly the desired
             gradient w.r.t. `theta`.
        """
        v = vertices.detach().requires_grad_(True)
        s = sdf_model(v)
        if s.dim() > 1:
            s = s.squeeze(-1)
        (g,) = torch.autograd.grad(s, v, torch.ones_like(s), create_graph=False)
        normals = g / (g.norm(dim=1, keepdim=True) + 1e-8)
        dL_ds = -(dL_dvertices * normals).sum(dim=1).detach()

        v2 = vertices.detach().requires_grad_(True)
        s2 = sdf_model(v2)
        if s2.dim() > 1:
            s2 = s2.squeeze(-1)
        return (dL_ds * s2).sum()

    # ---- cross-loss mesh sharing + stats (API compat) ------------------------
    def extraction_key(self) -> tuple:
        """Hashable key for grouping extractors whose extracted meshes are interchangeable.

        Steps:
          1) Pack all fields that affect the extracted geometry (bbox,
             effective resolution, isovalue, target vertex count, tolerance).
          2) `TotalLoss` groups extractors by this key, extracts once, and
             replays the result into siblings via `set_shared_mesh`.
        """
        return (
            tuple(tuple(ax) for ax in self.bbox),
            self.effective_resolution,
            self.isovalue,
            self.target_num_vertices,
            round(self.vertex_count_tolerance, 4),
        )

    def set_shared_mesh(self, vertices: torch.Tensor, faces: torch.Tensor) -> None:
        """Inject a pre-extracted mesh so subsequent `extract_mesh` calls reuse it.

        Steps:
          1) Store `(vertices, faces)` on the instance.
          2) `extract_mesh` short-circuits to these tensors until cleared.
        """
        self._shared_vertices = vertices
        self._shared_faces = faces

    def clear_shared_mesh(self) -> None:
        """Remove any injected shared mesh (re-enables normal cache lookup)."""
        self._shared_vertices = None
        self._shared_faces = None

    def get_stats(self) -> dict[str, int | float]:
        """Return a shallow copy of the per-extractor stats dict (API compat)."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset per-extractor stats (API compat; stats collection is minimal)."""
        self._stats = {"calls": 0}

    def log_stats_summary(self, level: int = logging.INFO) -> None:  # noqa: ARG002
        """No-op retained for API compat with the older timing-heavy extractor."""
        return None

    def _empty(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (empty vertices, empty faces) tensors on the correct device."""
        return (
            torch.empty((0, 3), dtype=torch.float32, device=self.device),
            torch.empty((0, 3), dtype=torch.int64, device=self.device),
        )

    def _eval_points(self, sdf_model: SDFModel, points: torch.Tensor) -> torch.Tensor:
        """Evaluate the SDF at an arbitrary point cloud in memory-bounded chunks.

        Steps:
          1) Iterate through `points` in slices of `self.batch_size`.
          2) Disable grad (`torch.no_grad`); forward pass only.
          3) On CUDA, wrap the forward in fp16 autocast to halve activation memory.
          4) Squeeze singleton output dims and cast back to fp32 for marching cubes.
          5) Concatenate chunk outputs into a single 1-D tensor.
        """
        if points.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=self.device)
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, points.shape[0], self.batch_size):
                chunk = points[i : i + self.batch_size]
                if self._use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        v = sdf_model(chunk)
                else:
                    v = sdf_model(chunk)
                if v.dim() > 1:
                    v = v.squeeze(-1)
                out.append(v.detach().to(torch.float32))
        return torch.cat(out)

    def _eval_grid(self, sdf_model: SDFModel, res: int) -> np.ndarray:
        """Evaluate the SDF on a dense `res^3` uniform grid spanning `self.bbox`.

        Steps:
          1) Build per-axis linspaces of length `res` inside the bbox.
          2) Form the Cartesian product via `torch.meshgrid(..., indexing='ij')`.
          3) Flatten to an (N, 3) point array and evaluate via `_eval_points`.
          4) Reshape the result into a `(res, res, res)` numpy volume for
             consumption by marching cubes.
        """
        axes = [torch.linspace(self.bbox[a][0], self.bbox[a][1], res, device=self.device) for a in range(3)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        pts = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        return self._eval_points(sdf_model, pts).cpu().numpy().reshape(res, res, res)

    def _marching_cubes(self, volume: np.ndarray, res: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Run PyMCubes marching cubes on a volume and lift vertices to world coordinates.

        Steps:
          1) Reject volumes whose value range does not bracket the isovalue
             (nothing to extract). Return an empty mesh.
          2) Compute per-axis `spacing` so each cell has physical size
             (extent / (res - 1)).
          3) Run the PyMCubes C++ backend; vertices come back in index space.
          4) Rescale vertices by `spacing` and shift by the bbox origin to
             obtain world coordinates.
          5) Convert to torch tensors on `self.device`.
        """
        mn, mx = float(volume.min()), float(volume.max())
        if not (mn <= self.isovalue <= mx):
            _LOGGER.warning("SDF does not bracket isovalue %.4f (range [%.4f, %.4f]).", self.isovalue, mn, mx)
            return self._empty()
        bbox = self.bbox
        spacing = tuple((bbox[a][1] - bbox[a][0]) / (res - 1) for a in range(3))
        origin = (bbox[0][0], bbox[1][0], bbox[2][0])
        v, f = _mcubes.marching_cubes(volume, self.isovalue)
        v = np.asarray(v, dtype=np.float32)
        f = np.asarray(f, dtype=np.int64)
        v[:, 0] = v[:, 0] * spacing[0] + origin[0]
        v[:, 1] = v[:, 1] * spacing[1] + origin[1]
        v[:, 2] = v[:, 2] * spacing[2] + origin[2]
        return torch.from_numpy(v).to(self.device), torch.from_numpy(f).to(self.device)

    def _extract(self, sdf_model: SDFModel, res: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform marching cubes; sparse coarse-to-fine when `res` is large.

        Steps:
          1) For small grids (res <= 64), evaluate the full dense volume and
             run marching cubes directly (overhead of sparse path not worth it).
          2) For larger grids, dispatch to `_extract_sparse` which only
             evaluates the SDF in cells adjacent to the zero-crossing.
        """
        sparse_threshold = 64
        if res > sparse_threshold:
            return self._extract_sparse(sdf_model, fine_res=res, coarse_res=sparse_threshold)
        return self._marching_cubes(self._eval_grid(sdf_model, res), res)

    def _extract_sparse(self, sdf_model: SDFModel, fine_res: int, coarse_res: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse coarse-to-fine marching cubes: O(surface) SDF evals instead of O(res^3).

        Steps:
          1) Evaluate a dense `coarse_res^3` volume (cheap even on CPU).
          2) Reject if the coarse volume does not bracket the isovalue.
          3) For every coarse cell, stack its 8 corner values and flag cells
             where `min < 0 < max` (cells containing the zero-crossing).
          4) Dilate the active mask by one cell (3x3x3 neighbourhood) so the
             fine evaluation includes a safety margin around the surface.
          5) Map each fine voxel back to its containing coarse cell via
             `f2c = floor(i * (coarse-1) / (fine-1))`; voxels in inactive
             coarse cells inherit the (already evaluated) nearest coarse value.
          6) Upsample the coarse volume via nearest-neighbour indexing (`ci`).
          7) Compute flat indices of fine voxels that are active, decode them
             into (ix, iy, iz), and build their world-space coordinates using
             bbox-scaled steps.
          8) Evaluate the SDF on only those surface-adjacent points and write
             the results back into the upsampled volume in place.
          9) Run marching cubes on the patched fine volume.
        Complexity: ~O(coarse^3 + surface * (fine/coarse)^2), typically an
        order of magnitude fewer SDF evaluations than a dense fine grid.
        """
        coarse = self._eval_grid(sdf_model, coarse_res)
        if not (float(coarse.min()) <= self.isovalue <= float(coarse.max())):
            _LOGGER.warning("Coarse grid does not bracket isovalue.")
            return self._empty()
        shifted = coarse - self.isovalue
        c = coarse_res - 1
        corners = np.stack([shifted[dx : c + dx, dy : c + dy, dz : c + dz] for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)])
        active = (corners.min(axis=0) < 0) & (corners.max(axis=0) > 0)
        padded = np.pad(active, 1)
        dilated = np.zeros_like(active)
        for dx in range(3):
            for dy in range(3):
                for dz in range(3):
                    dilated |= padded[dx : c + dx, dy : c + dy, dz : c + dz]

        f2c = np.minimum((np.arange(fine_res) * c / (fine_res - 1)).astype(np.intp), c - 1)
        fine_active = dilated[f2c[:, None, None], f2c[None, :, None], f2c[None, None, :]]
        ci = np.clip(np.round(np.arange(fine_res) * (coarse_res - 1) / (fine_res - 1)).astype(np.intp), 0, coarse_res - 1)
        vol = coarse[np.ix_(ci, ci, ci)].copy()

        flat_idx = np.flatnonzero(fine_active.ravel())
        if flat_idx.size == 0:
            return self._empty()

        bbox = self.bbox
        plane = fine_res * fine_res
        flat_t = torch.from_numpy(flat_idx.astype(np.int64)).to(self.device)
        ix = torch.div(flat_t, plane, rounding_mode="floor")
        rem = flat_t - ix * plane
        iy = torch.div(rem, fine_res, rounding_mode="floor")
        iz = rem - iy * fine_res
        sx = (bbox[0][1] - bbox[0][0]) / (fine_res - 1)
        sy = (bbox[1][1] - bbox[1][0]) / (fine_res - 1)
        sz = (bbox[2][1] - bbox[2][0]) / (fine_res - 1)
        pts = torch.stack([bbox[0][0] + ix.float() * sx, bbox[1][0] + iy.float() * sy, bbox[2][0] + iz.float() * sz], dim=1)
        vol.ravel()[flat_idx] = self._eval_points(sdf_model, pts).cpu().numpy()
        return self._marching_cubes(vol, fine_res)

    def _extract_to_target(self, sdf_model: SDFModel) -> tuple[torch.Tensor, torch.Tensor]:
        """Adaptively extract a mesh whose vertex count is within tolerance of a target.

        Steps:
          1) Compute `[lo, hi] = target * (1 +/- tolerance)`.
          2) Warm-start the resolution from `_cached_resolution` (success
             resolution from the previous call) or the configured default.
          3) Loop up to 6 times:
             a) Clamp `res` into [8, 512] and extract at that resolution.
             b) If the mesh is empty, cache `res` and return.
             c) If `lo <= n <= hi`, cache `res` and return (target hit).
             d) If `n > hi`, cache a smaller predicted `res` for next call
                and decimate the current mesh down to `target` via
                `fast_simplification`.
             e) Otherwise (`n < lo`) scale `res` up by `sqrt(target / n)`
                (area scales with res^2) and retry.
          4) If the loop exits without convergence, decimate if still over
             budget; else return as-is.
        """
        target = int(self.target_num_vertices)  # type: ignore[arg-type]
        tol = self.vertex_count_tolerance
        lo, hi = int(target * (1 - tol)), int(target * (1 + tol))
        res = self._cached_resolution or self.effective_resolution
        v, f = self._empty()
        for _ in range(6):
            res = max(8, min(512, res))
            v, f = self._extract(sdf_model, res)
            n = int(v.shape[0])
            if n == 0:
                self._cached_resolution = res
                return v, f
            if lo <= n <= hi:
                self._cached_resolution = res
                return v, f
            if n > hi:
                self._cached_resolution = max(8, int(res * (target / n) ** 0.5))
                return self._decimate(v, f, target)
            res = max(res + 2, int(res * (target / n) ** 0.5))
        self._cached_resolution = res
        return self._decimate(v, f, target) if int(v.shape[0]) > hi else (v, f)

    def _decimate(self, v: torch.Tensor, f: torch.Tensor, target: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Quadric-edge-collapse decimation down to (approximately) `target` vertices.

        Steps:
          1) No-op if `fast_simplification` is unavailable or vertex count
             already within target.
          2) Scale the current face count by `target / current_vertices` to
             estimate the equivalent face budget (edge-collapse removes two
             faces per vertex).
          3) Run `fast_simplification.simplify` on float64 vertices + int64
             faces (its expected dtypes).
          4) Convert the simplified arrays back to torch tensors on
             `self.device`.
        """
        if _fast_simplify is None or int(v.shape[0]) <= target:
            return v, f
        target_faces = max(4, int(f.shape[0] * (target / max(1, int(v.shape[0])))))
        v2, f2 = _fast_simplify(
            np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float64),
            np.ascontiguousarray(f.detach().cpu().numpy(), dtype=np.int64),
            target_count=target_faces,
        )
        return (
            torch.from_numpy(np.asarray(v2, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.asarray(f2, dtype=np.int64)).to(self.device),
        )
