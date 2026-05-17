"""Assembly-sourced interface points and distance masking for the diversity loss.

The diversity loss must not pull SDF values around at hard-constraint regions
(wheel contact patches, inner-prism / seat region). This module rebuilds those
constraint point sets directly from the assembly + regulations — independent of
which constraint losses happen to be active — and provides a small utility to
drop diversity query points within configurable cutoffs.

Wheel-patch construction mirrors `WheelLossV3._setup`; inner-bb sampling mirrors
`BbV44._setup`. They are duplicated here on purpose so the diversity pipeline
stays self-contained and untouched by loss-side refactors.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import trimesh
import yaml
from torch import Tensor

from subprojects.wp2_assembler.version4.components.nbz import call_assembly, get_training_frame_offset
from subprojects.wp2_assembler.version4.utils.reg_imp import get_regulation


_ASSEMBLY_MAP_PATH = (
    Path(__file__).parents[6] / "subprojects/wp2_assembler/version4/config/assembly_map.yaml"
)


def build_wheel_interface_points(
    assembly_id: int,
    sampling_density: int = 10,
    manual_z_shift: float = 0.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Wheel contact-patch grid in mm, one rectangle per wheel.

    Args:
        assembly_id: assembly_map.yaml key.
        sampling_density: points per edge of each rectangular contact patch.
        manual_z_shift: optional vertical offset matching `wheel_loss_config.manual_z_shift`.
        device: target torch device.

    Returns:
        (num_wheels * sampling_density^2, 3) tensor of points in mm.
    """
    with _ASSEMBLY_MAP_PATH.open("r") as f:
        assembly_map = yaml.safe_load(f) or {}
    params = assembly_map[int(assembly_id)]
    wheel_coords = [(p["x"], p["y"], p["z"]) for key, p in params.items() if key.startswith("wheel_")]

    # Match WheelInterfaceV0 / WheelAlignmentV0: raw assembly_map coords must be
    # shifted into the SDF training frame, otherwise samples land in the wrong
    # spatial location and don't match what the model sees. See
    # `get_training_frame_offset` docstring.
    cx, cy, cz = get_training_frame_offset(int(assembly_id))

    width = get_regulation("wheel.tire_width")
    radius = get_regulation("wheel.tire_diameter") / 2
    gc = get_regulation("no_build_zones.outer_prism.ground_clearance")
    C = radius - gc
    L = math.sqrt(radius**2 - C**2)

    dx_vals = torch.linspace(-1, 1, sampling_density)
    dy_vals = torch.linspace(-1, 1, sampling_density)
    grid_dx, grid_dy = torch.meshgrid(dx_vals, dy_vals, indexing="ij")
    grid_dx = grid_dx.flatten()
    grid_dy = grid_dy.flatten()

    grids: list[Tensor] = []
    for wx, wy, wz in wheel_coords:
        xs = (wx - cx) + grid_dx * L
        ys = (wy - cy) + grid_dy * (width / 2)
        zs = torch.full_like(xs, (wz - cz) + radius - C + manual_z_shift)
        grids.append(torch.stack([xs, ys, zs], dim=-1))
    return torch.cat(grids, dim=0).to(device)


def build_bb_inner_interface_points(
    assembly_id: int,
    target_point_count: int = 20000,
    tess_tolerance: float = 2.0,
    tess_angular_tolerance: float = 0.4,
    min_points_per_inner_mesh: int = 256,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Area-weighted surface samples on the inner-prism (seat + future components).

    Args:
        assembly_id: assembly_map.yaml key.
        target_point_count: total inner-surface samples (single budget — there is
            no outer split here unlike `BbV44`).
        tess_tolerance, tess_angular_tolerance: CadQuery tessellation params.
        min_points_per_inner_mesh: minimum sample count per individual inner solid
            so tiny components are still represented.
        device: target torch device.

    Returns:
        (target_point_count_approx, 3) tensor of points in mm sampled on the
        union of all inner-prism solids' zero-level surfaces.
    """
    inner_assembly, _outer_solid, _vision_nbz = call_assembly(id_number=int(assembly_id))
    inner_solids = list(inner_assembly.toCompound().Solids())

    inner_meshes: list[trimesh.Trimesh] = []
    for solid in inner_solids:
        vertices_cq, faces_cq = solid.tessellate(tess_tolerance, tess_angular_tolerance)
        vertices_np = np.array([[v.x, v.y, v.z] for v in vertices_cq], dtype=np.float32)
        faces_np = np.array(faces_cq, dtype=np.int64)
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()
        if len(mesh.faces) == 0:
            continue
        trimesh.repair.fix_normals(mesh, multibody=False)
        inner_meshes.append(mesh)

    if not inner_meshes:
        return torch.zeros((0, 3), dtype=torch.float32, device=device)

    areas = np.array([max(float(m.area), 1e-6) for m in inner_meshes], dtype=np.float64)
    weights = areas / areas.sum()
    counts = np.maximum(min_points_per_inner_mesh, np.floor(target_point_count * weights).astype(int))
    deficit = target_point_count - int(counts.sum())
    if deficit > 0:
        for idx in np.argsort(-weights)[:deficit]:
            counts[idx] += 1

    parts: list[np.ndarray] = []
    for mesh, count in zip(inner_meshes, counts, strict=False):
        if int(count) <= 0:
            continue
        pts, _face_idx = trimesh.sample.sample_surface(mesh, int(count))
        parts.append(pts.astype(np.float32, copy=False))
    points_np = np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), dtype=np.float32)
    return torch.from_numpy(points_np).to(device)


class InterfaceSource(NamedTuple):
    """One mask source: a point cloud and the keep-out radius around its points."""

    name: str
    points: Tensor  # (M, 3) mm
    cutoff_mm: float
    tag: int  # 0=kept, positive integers per-source for visualization


def build_default_sources(
    assembly_id: int,
    wheel_cutoff_mm: float,
    bb_inner_cutoff_mm: float,
    wheel_sampling_density: int = 10,
    wheel_manual_z_shift: float = 0.0,
    bb_inner_target_point_count: int = 20000,
    device: str | torch.device = "cpu",
) -> list[InterfaceSource]:
    """Convenience: build wheel + bb_inner sources at once with their cutoffs.

    Tag convention: 1 = wheel, 2 = bb_inner. Future sources keep counting up.
    Returning a list (not a dict) keeps the tag-to-source order stable.
    """
    sources: list[InterfaceSource] = []
    if wheel_cutoff_mm > 0:
        sources.append(
            InterfaceSource(
                name="wheel",
                points=build_wheel_interface_points(
                    assembly_id=assembly_id,
                    sampling_density=wheel_sampling_density,
                    manual_z_shift=wheel_manual_z_shift,
                    device=device,
                ),
                cutoff_mm=float(wheel_cutoff_mm),
                tag=1,
            )
        )
    if bb_inner_cutoff_mm > 0:
        sources.append(
            InterfaceSource(
                name="bb_inner",
                points=build_bb_inner_interface_points(
                    assembly_id=assembly_id,
                    target_point_count=bb_inner_target_point_count,
                    device=device,
                ),
                cutoff_mm=float(bb_inner_cutoff_mm),
                tag=2,
            )
        )
    return sources


def mask_points_by_distance(
    query_pts: Tensor,
    sources: list[InterfaceSource],
) -> tuple[Tensor, Tensor]:
    """Drop query points within `cutoff_mm` of any source's points.

    For each source, computes min-distance from each query to the source's point
    set and marks queries inside the cutoff as dropped. First-hit source wins for
    the tag assignment so the visualizer can color the reason a point was dropped.

    Args:
        query_pts: (N, 3) tensor in mm. Lives on whatever device.
        sources: list of `InterfaceSource`. Their `points` are moved to
            `query_pts.device` on the fly.

    Returns:
        keep_mask: (N,) bool tensor — True for retained points.
        tag: (N,) int64 tensor — 0 if kept, otherwise the tag of the first
            source that dropped it (matches `InterfaceSource.tag`).
    """
    n = query_pts.shape[0]
    keep_mask = torch.ones(n, dtype=torch.bool, device=query_pts.device)
    tag = torch.zeros(n, dtype=torch.int64, device=query_pts.device)

    for src in sources:
        if src.points.numel() == 0 or src.cutoff_mm <= 0:
            continue
        src_pts = src.points.to(query_pts.device)
        # cdist returns (N, M); take min over M for per-query nearest interface point.
        min_dist = torch.cdist(query_pts, src_pts).min(dim=1).values
        dropped_by_this = (min_dist <= src.cutoff_mm) & keep_mask
        tag = torch.where(dropped_by_this, torch.full_like(tag, src.tag), tag)
        keep_mask = keep_mask & ~(min_dist <= src.cutoff_mm)

    return keep_mask, tag
