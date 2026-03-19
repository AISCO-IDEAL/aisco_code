"""
Sparse Differentiable Dual Marching Cubes.

Octree-based adaptive SDF evaluation + sparse DMC mesh extraction.
Pure PyTorch — works on CPU, MPS, and CUDA. Zero TorchCAD imports.

Public API:
    octree_sdf_eval()          — adaptive SDF evaluation via octree
    sparse_dual_marching_cubes() — sparse DMC mesh extraction
    sparse_sdf_to_mesh()       — top-level entry chaining both
"""

import math
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from .tables import get_tables

if TYPE_CHECKING:
    @runtime_checkable
    class MeshTimer(Protocol):
        """Protocol for mesh extraction phase timing.

        Any object with a ``start(name)`` method satisfies this protocol.
        Each ``start()`` call implicitly ends the previous phase.
        """

        def start(self, name: str) -> None: ...


def _chunked_sdf_eval(
    sdf_fn: Callable[[Tensor], Tensor],
    points: Tensor,
    chunk_size: int | None,
) -> Tensor:
    """
    Evaluate SDF in chunks to bound neural net activation memory.

    Args:
        sdf_fn: SDF function. [N, 3] -> [B, N].
        points: [N, 3] query points.
        chunk_size: Max points per chunk. None = no chunking.

    Returns:
        [B, N] SDF values.
    """
    N = points.shape[0]
    if chunk_size is None or chunk_size >= N:
        return sdf_fn(points)
    chunks = [sdf_fn(points[i : i + chunk_size]) for i in range(0, N, chunk_size)]
    return torch.cat(chunks, dim=1)  # [B, N]


def _unique_int_coords(
    coords: Tensor,
    max_coord: int,
) -> tuple[Tensor, Tensor]:
    """
    Deduplicate rows of integer coordinates using 1D scalar keys.

    Replaces torch.unique(dim=0) which is broken on MPS
    (aten::unique_dim falls back to CPU with incorrect inverse indices).

    Args:
        coords:    [N, 3] int tensor with values in [0, max_coord].
        max_coord: Maximum coordinate value (inclusive).

    Returns:
        unique_coords: [M, 3] unique rows (same dtype as input).
        inverse_idx:   [N] int64 mapping from input rows to unique rows.
    """
    R = max_coord + 1
    keys = coords[:, 0].long() * (R * R) + coords[:, 1].long() * R + coords[:, 2].long()
    unique_keys, inverse_idx = torch.unique(keys, return_inverse=True)
    unique_coords = torch.stack(
        [
            unique_keys // (R * R),
            (unique_keys // R) % R,
            unique_keys % R,
        ],
        dim=1,
    ).to(coords.dtype)
    return unique_coords, inverse_idx


# ---------------------------------------------------------------------------
# Boundary clipping detection (sparse grids)
# ---------------------------------------------------------------------------


def _check_boundary_clipping_sparse(
    grid_coords: Tensor,
    sdf_values: Tensor,
    resolution: int,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
) -> None:
    """
    Check if SDF values on boundary faces of the bounding box are negative.

    For sparse octree grids, boundary points are those where any grid
    coordinate equals 0 or equals the resolution. Only points present in
    the sparse grid are checked (octree culling may have removed some
    boundary cells, which is fine — those are far from the surface).

    Args:
        grid_coords: [M, 3] int tensor — unique grid coordinates.
        sdf_values:  [B, M] float tensor — SDF at each grid point.
        resolution:  Grid resolution (coords range from 0 to resolution).
        xyz_min:     Bounding box min corner.
        xyz_max:     Bounding box max corner.
    """
    axis_names = ["x", "y", "z"]
    clipped_faces: list[str] = []

    for axis in range(3):
        coord = grid_coords[:, axis]

        # Check min face (coord == 0)
        min_mask = coord == 0
        if min_mask.any():
            boundary_sdf = sdf_values[:, min_mask]  # [B, N_face]
            if (boundary_sdf < 0).any():
                clipped_faces.append(f"{axis_names[axis]}_min")

        # Check max face (coord == resolution)
        max_mask = coord == resolution
        if max_mask.any():
            boundary_sdf = sdf_values[:, max_mask]  # [B, N_face]
            if (boundary_sdf < 0).any():
                clipped_faces.append(f"{axis_names[axis]}_max")

    if clipped_faces:
        faces_str = ", ".join(clipped_faces)
        warnings.warn(
            f"Shape extends beyond mesh bounds. "
            f"Clipped faces: {faces_str}. "
            f"Current bounds: x=[{xyz_min[0]}, {xyz_max[0]}], "
            f"y=[{xyz_min[1]}, {xyz_max[1]}], "
            f"z=[{xyz_min[2]}, {xyz_max[2]}]. "
            f"Increase bounds to capture the full shape.",
            stacklevel=4,
        )


def _normalize_device(device: torch.device | str | None) -> torch.device:
    """Return a concrete torch.device, defaulting to CPU."""
    if device is None:
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _phase_starter(timer: "MeshTimer | None") -> Callable[[str], None]:
    """Return a callable that starts timer phases, or a no-op."""
    if timer is None:
        return lambda _: None
    return timer.start


def _refine_octree_cells(
    level_cells: Tensor,
    cur_res: int,
    num_levels: int,
    corner_offsets: Tensor,
    bb_min: Tensor,
    bb_extent: Tensor,
    lipschitz: float,
    sdf_fn: Callable[[Tensor], Tensor],
    max_points_per_chunk: int | None,
    start_phase: Callable[[str], None],
) -> Tensor:
    """Refine sparse octree cells from coarse resolution to target."""
    with torch.no_grad():
        for level in range(num_levels):
            start_phase(f"octree_level_{level}_sdf_eval")

            cell_size_world = bb_extent / cur_res  # [3] world-space cell size
            cell_diag = torch.norm(cell_size_world)

            # Compute unique corners for current cells
            # Each cell has 8 corners at coords + corner_offsets
            # cell_corners_grid: [K, 8, 3] int coords at cur_res+1 grid
            cell_corners_grid = level_cells.unsqueeze(1) + corner_offsets.unsqueeze(0)  # [K, 8, 3]

            # Flatten to [K*8, 3], find unique (1D keys — avoids broken MPS unique_dim)
            flat_corners = cell_corners_grid.reshape(-1, 3).long()
            unique_corners, inverse_idx = _unique_int_coords(flat_corners, max_coord=cur_res)  # [M, 3], [K*8]

            # Convert unique corners to world-space positions
            unique_pos = bb_min + (unique_corners.float() / cur_res) * bb_extent  # [M, 3]

            # Evaluate SDF at unique corners
            sdf_vals = _chunked_sdf_eval(sdf_fn, unique_pos, max_points_per_chunk)  # [B, M]
            if sdf_vals.dim() == 1:
                sdf_vals = sdf_vals.unsqueeze(0)

            start_phase(f"octree_level_{level}_cull_subdivide")

            # Gather per-cell corner SDF: [B, K, 8]
            cell_corner_idx = inverse_idx.reshape(-1, 8)  # [K, 8]
            cell_sdf = sdf_vals[:, cell_corner_idx.long()]  # [B, K, 8] — advanced indexing on cols

            # Lipschitz near-surface test: min(|sdf|) across corners AND batch < L * diag
            abs_sdf = cell_sdf.abs()  # [B, K, 8]
            min_abs_per_cell = abs_sdf.min(dim=-1).values.min(dim=0).values  # [K] — min over batch and corners
            near_surface = min_abs_per_cell < lipschitz * cell_diag

            # Keep only near-surface cells
            active_cells = level_cells[near_surface]  # [K', 3]

            # Subdivide: each cell becomes 8 children at 2x resolution.
            cur_res *= 2
            parent_2x = active_cells * 2  # [K', 3] scaled to new res
            child_offsets = corner_offsets.unsqueeze(0)  # [1, 8, 3]
            children = parent_2x.unsqueeze(1) + child_offsets  # [K', 8, 3]
            level_cells = children.reshape(-1, 3)  # [K'*8, 3]

            # Deduplicate children (cells can overlap from different parents)
            level_cells, _ = _unique_int_coords(level_cells, max_coord=cur_res - 1)

    return level_cells


def _evaluate_final_octree_cells(
    final_cells: Tensor,
    resolution: int,
    corner_offsets: Tensor,
    bb_min: Tensor,
    bb_extent: Tensor,
    sdf_fn: Callable[[Tensor], Tensor],
    max_points_per_chunk: int | None,
    lipschitz: float,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    start_phase: Callable[[str], None],
) -> tuple[Tensor, Tensor, Tensor]:
    """Evaluate final octree corners and cull non-surface cells."""
    start_phase("octree_final_sdf")

    cell_size_final_world = bb_extent / resolution  # [3]
    cell_diag_final = torch.norm(cell_size_final_world)

    # Compute corners for final cells
    final_corners_grid = final_cells.unsqueeze(1) + corner_offsets.unsqueeze(0)  # [K, 8, 3]
    flat_final = final_corners_grid.reshape(-1, 3).long()
    unique_final, inv_final = _unique_int_coords(flat_final, max_coord=resolution)  # [M, 3], [K*8]

    # World-space positions for unique corners
    unique_pos_final = bb_min + (unique_final.float() / resolution) * bb_extent  # [M, 3]

    # Evaluate with gradients
    sdf_final = _chunked_sdf_eval(sdf_fn, unique_pos_final, max_points_per_chunk)  # [B, M]
    if sdf_final.dim() == 1:
        sdf_final = sdf_final.unsqueeze(0)

    # --- Boundary clipping check ---
    # Check if any SDF values on the 6 boundary faces of the bounding box are
    # negative, meaning the shape extends beyond the mesh bounds.
    with torch.no_grad():
        _check_boundary_clipping_sparse(unique_final, sdf_final, resolution, xyz_min, xyz_max)

    # Gather per-cell
    cell_corner_idx_final = inv_final.reshape(-1, 8)  # [K, 8]
    corner_sdf = sdf_final[:, cell_corner_idx_final.long()]  # [B, K, 8]

    # Corner world positions: [K, 8, 3]
    corner_positions = unique_pos_final[cell_corner_idx_final.long()]  # [K, 8, 3]

    # One more Lipschitz cull — remove cells that are definitely not near surface
    # at the final resolution
    start_phase("octree_final_cull")
    with torch.no_grad():
        abs_final = corner_sdf.abs()
        min_abs_final = abs_final.min(dim=-1).values.min(dim=0).values  # [K]
        near_final = min_abs_final < lipschitz * cell_diag_final

    final_cells = final_cells[near_final]
    corner_sdf = corner_sdf[:, near_final]
    corner_positions = corner_positions[near_final]

    return final_cells, corner_sdf, corner_positions


# ---------------------------------------------------------------------------
# Phase 1: Octree-based adaptive SDF evaluation
# ---------------------------------------------------------------------------


def octree_sdf_eval(
    sdf_fn: Callable[[Tensor], Tensor],
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int,
    _batch_size: int,
    lipschitz: float = 1.1,
    coarse_factor: int = 8,
    device: torch.device | None = None,
    max_points_per_chunk: int | None = None,
    timer: "MeshTimer | None" = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Adaptive SDF evaluation via octree refinement.

    Starts at a coarse grid, evaluates SDF, keeps only cells near the
    surface (Lipschitz test), subdivides those, and repeats until the
    target resolution is reached.

    Args:
        sdf_fn: SDF function. [N, 3] -> [B, N].
        xyz_min: Bounding box min corner.
        xyz_max: Bounding box max corner.
        resolution: Target grid resolution (must be power of 2).
        _batch_size: Number of batch elements.
        lipschitz: Lipschitz constant of the SDF (default 1.1).
        coarse_factor: Ratio of target to starting resolution (default 8).
        device: Torch device.
        max_points_per_chunk: Max query points per SDF eval chunk (None = no chunking).

    Returns:
        cell_coords:      [K, 3]    int32  — active cell grid coords at target resolution
        corner_sdf:       [B, K, 8] float  — SDF at 8 corners (with gradients)
        corner_positions: [K, 8, 3] float  — world-space corner positions
    """
    start_phase = _phase_starter(timer)
    start_phase("octree_init")

    device = _normalize_device(device)

    print(__file__)
    print(type(device),device)

    # Validate resolution is power of 2
    if resolution < 2 or (resolution & (resolution - 1)) != 0:
        raise ValueError(f"resolution must be power of 2, got {resolution}")

    # MPS torch.unique silently produces wrong results for int64 keys > 2^24.
    # At resolution > 128, coordinate keys exceed this limit and corrupt the
    # octree deduplication (wrong inverse indices → wrong SDF gathering).
    if device.type == "mps" and resolution > 128:
        raise ValueError(
            f"resolution={resolution} exceeds MPS limit of 128. "
            f"MPS torch.unique produces wrong results for large int64 keys. "
            f"Use resolution <= 128 on MPS, or use device='cpu'."
        )

    coarse_res = resolution // coarse_factor
    coarse_res = max(coarse_res, 1)

    num_levels = int(math.log2(coarse_factor))
    if 2**num_levels != coarse_factor:
        raise ValueError(f"coarse_factor must be power of 2, got {coarse_factor}")

    # Bounding box as tensors
    bb_min = torch.tensor(xyz_min, dtype=torch.float32, device=device)
    bb_max = torch.tensor(xyz_max, dtype=torch.float32, device=device)
    bb_extent = bb_max - bb_min

    tables = get_tables(device)
    corner_offsets = tables[0]  # [8, 3] int32

    # Start: all cells at coarse level
    cur_res = coarse_res
    ci = torch.arange(cur_res, device=device, dtype=torch.int32)
    gx, gy, gz = torch.meshgrid(ci, ci, ci, indexing="ij")
    level_cells = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=1)  # [K, 3]

    level_cells = _refine_octree_cells(
        level_cells,
        cur_res,
        num_levels,
        corner_offsets,
        bb_min,
        bb_extent,
        lipschitz,
        sdf_fn,
        max_points_per_chunk,
        start_phase,
    )

    final_cells, corner_sdf, corner_positions = _evaluate_final_octree_cells(
        level_cells,
        resolution,
        corner_offsets,
        bb_min,
        bb_extent,
        sdf_fn,
        max_points_per_chunk,
        lipschitz,
        xyz_min,
        xyz_max,
        start_phase,
    )

    return final_cells, corner_sdf, corner_positions


# ---------------------------------------------------------------------------
# Phase 2: Sparse Dual Marching Cubes
# ---------------------------------------------------------------------------


def _resolve_ambiguous_cases(
    case_idx: Tensor,
    cell_coords: Tensor,
    lookup_cells,
    problematic_configs: Tensor,
    resolution: int,
    R: int,
) -> Tensor:
    """
    Resolve ambiguous C16/C19 marching cubes cases for manifold output.

    For each problematic cell, checks the specific neighbor sharing the
    ambiguous face. If that neighbor is also problematic, XORs the case
    index with 0xFF (swaps C16<->C19).

    Args:
        case_idx:    [K] int32 — case index per cell
        cell_coords: [K, 3] int32 — cell grid coordinates
        lookup_cells: callable — maps [N] int64 keys to [N] int64 cell indices (-1 if missing)
        problematic_configs: [256] int32 — ambiguity direction encoding
        resolution: int — grid resolution (cells in [0, resolution-1])
        R: int — hash key range (resolution + 1)

    Returns:
        resolved_case: [K] int32 — case indices with ambiguities resolved
    """

    prob = problematic_configs[case_idx.long()]  # [K]
    is_problematic = prob != 255  # [K]

    if not is_problematic.any():
        return case_idx  # fast path — common for convex shapes

    prob_indices = torch.where(is_problematic)[0]  # [P]
    direction = prob[prob_indices]  # [P] — 3-bit encoding

    # Decode: bit0=sign (0=neg delta, 1=pos delta), bits1-2=axis (0=X, 1=Y, 2=Z)
    sign_bit = direction & 1  # [P]
    axis = (direction >> 1) & 3  # [P]
    delta = torch.where(sign_bit == 1, torch.ones_like(sign_bit), -torch.ones_like(sign_bit))  # [P]

    # Compute neighbor coordinates
    neighbor_coords = cell_coords[prob_indices].clone()  # [P, 3]
    for a in range(3):
        mask_a = axis == a
        if mask_a.any():
            neighbor_coords[mask_a, a] += delta[mask_a]

    # Bounds check
    in_bounds = ((neighbor_coords >= 0) & (neighbor_coords < resolution)).all(dim=1)  # [P]

    # Look up neighbor cells
    neighbor_keys = neighbor_coords[:, 0].long() * R * R + neighbor_coords[:, 1].long() * R + neighbor_coords[:, 2].long()  # [P]

    neighbor_cell_idx = torch.full_like(neighbor_keys, -1)
    if in_bounds.any():
        neighbor_cell_idx[in_bounds] = lookup_cells(neighbor_keys[in_bounds])

    # Check if neighbor exists and is also problematic
    neighbor_exists = neighbor_cell_idx >= 0  # [P]
    should_xor = torch.zeros_like(neighbor_exists)  # [P]

    if neighbor_exists.any():
        neighbor_cases = case_idx[neighbor_cell_idx[neighbor_exists]]  # [V]
        neighbor_prob = problematic_configs[neighbor_cases.long()]  # [V]
        should_xor[neighbor_exists] = neighbor_prob != 255

    # Apply XOR to resolve ambiguity
    if should_xor.any():
        resolved = case_idx.clone()
        xor_cells = prob_indices[should_xor]
        resolved[xor_cells] = resolved[xor_cells] ^ 0xFF
        return resolved

    return case_idx


def _resolve_ambiguous_cases_batched(
    case_idx: Tensor,
    cell_coords: Tensor,
    lookup_cells,
    problematic_configs: Tensor,
    resolution: int,
    R: int,
) -> Tensor:
    """
    Resolve ambiguous C16/C19 marching cubes cases for manifold output (batched).

    Processes all B batch elements simultaneously with zero GPU-CPU sync points.

    Args:
        case_idx:    [B, K] int32 — case index per cell per batch
        cell_coords: [K, 3] int32 — cell grid coordinates (shared)
        lookup_cells: callable — maps [N] int64 keys to [N] int64 cell indices (-1 if missing)
        problematic_configs: [256] int32 — ambiguity direction encoding
        resolution: int — grid resolution
        R: int — hash key range (resolution + 1)

    Returns:
        resolved_case: [B, K] int32
    """
    device = case_idx.device
    B, K = case_idx.shape

    # [B, K] — direction encoding per cell per batch
    prob = problematic_configs[case_idx.long()]  # [B, K]
    is_problematic = prob != 255  # [B, K]

    if not is_problematic.any():
        return case_idx  # fast path

    # Precompute ALL 6 neighbor directions for ALL K cells: [K, 6, 3]
    # Direction encoding: 0=-X, 1=+X, 2=-Y, 3=+Y, 4=-Z, 5=+Z
    # bit0=sign (0=neg, 1=pos), bits1-2=axis
    deltas = torch.tensor(
        [
            [-1, 0, 0],  # dir 0: -X
            [1, 0, 0],  # dir 1: +X
            [0, -1, 0],  # dir 2: -Y
            [0, 1, 0],  # dir 3: +Y
            [0, 0, -1],  # dir 4: -Z
            [0, 0, 1],  # dir 5: +Z
        ],
        dtype=torch.int32,
        device=device,
    )  # [6, 3]

    all_neighbor_coords = cell_coords.unsqueeze(1) + deltas.unsqueeze(0)  # [K, 6, 3]

    # Bounds check: [K, 6]
    in_bounds = ((all_neighbor_coords >= 0) & (all_neighbor_coords < resolution)).all(dim=2)  # [K, 6]

    # Compute keys for all 6 neighbors: [K, 6]
    all_neighbor_keys = (
        all_neighbor_coords[:, :, 0].long() * R * R
        + all_neighbor_coords[:, :, 1].long() * R
        + all_neighbor_coords[:, :, 2].long()
    )  # [K, 6]

    # Single flattened lookup: [K*6] -> [K*6]
    flat_keys = all_neighbor_keys.reshape(-1)  # [K*6]
    flat_in_bounds = in_bounds.reshape(-1)  # [K*6]
    flat_cell_idx = torch.full_like(flat_keys, -1)
    if flat_in_bounds.any():
        flat_cell_idx[flat_in_bounds] = lookup_cells(flat_keys[flat_in_bounds])
    all_neighbor_cell_idx = flat_cell_idx.reshape(K, 6)  # [K, 6]

    # For each batch element, gather the correct neighbor based on direction
    # prob contains the direction (0-5) for problematic cells, 255 for non-problematic
    # Clamp to [0, 5] for safe indexing (non-problematic cells get dir 0, but are masked out)
    direction_safe = prob.long().clamp(0, 5)  # [B, K]

    # Gather one neighbor index per batch/cell by selecting from the six precomputed neighbor slots using the per-batch direction code.
    neighbor_cell_idx = all_neighbor_cell_idx[torch.arange(K, device=device).unsqueeze(0).expand(B, -1), direction_safe]  # [B, K]

    # Check if neighbor exists
    neighbor_exists = neighbor_cell_idx >= 0  # [B, K]

    # Get neighbor's case index: [B, K]
    # Safe gather — use 0 for missing neighbors (masked out later)
    safe_neighbor_idx = neighbor_cell_idx.clamp(min=0)  # [B, K]
    neighbor_cases = torch.gather(case_idx, 1, safe_neighbor_idx)  # [B, K]
    neighbor_prob = problematic_configs[neighbor_cases.long()]  # [B, K]

    # Should XOR: problematic AND neighbor exists AND neighbor is also problematic
    should_xor = is_problematic & neighbor_exists & (neighbor_prob != 255)  # [B, K]

    # torch.where: single kernel, no GPU→CPU sync (vs .any() + clone + masked assign)
    return torch.where(should_xor, case_idx ^ 0xFF, case_idx)


def _emit_union_quads(
    cell_coords: Tensor,
    corner_sdf: Tensor,
    quad_cell_offsets: Tensor,
    quad_local_edges: Tensor,
    edge_to_slot: Tensor,
    cell_to_union: Tensor,
    union_indices: Tensor,
    lookup_cells: Callable[[Tensor], Tensor],
    R: int,
) -> tuple[list[Tensor], list[Tensor]]:
    """Emit quads for all axes and return per-batch faces with validity masks."""
    B = corner_sdf.shape[0]
    union_coords = cell_coords[union_indices]  # [K_union, 3]
    all_quads_list: list[Tensor] = []
    all_face_valid_list: list[Tensor] = []

    for axis in range(3):
        offsets = quad_cell_offsets[axis]  # [4, 3]
        neighbor_coords = union_coords.unsqueeze(1) + offsets.unsqueeze(0)  # [K_union, 4, 3]

        neighbor_keys = (
            neighbor_coords[:, :, 0].long() * R * R + neighbor_coords[:, :, 1].long() * R + neighbor_coords[:, :, 2].long()
        )  # [K_union, 4]

        neighbor_cell_idx = torch.stack([lookup_cells(neighbor_keys[:, j]) for j in range(4)], dim=1)  # [K_union, 4]

        # All 4 neighbors must exist in the sparse grid
        all_exist = (neighbor_cell_idx >= 0).all(dim=1)  # [K_union]
        if not all_exist.any():
            continue

        # All 4 neighbors must be in the union set
        cand_idx = torch.where(all_exist)[0]
        cand_cell_idx = neighbor_cell_idx[cand_idx]  # [C, 4]
        cand_union = cell_to_union[cand_cell_idx]  # [C, 4]
        all_in_union = (cand_union >= 0).all(dim=1)

        if not all_in_union.any():
            continue

        valid = cand_idx[all_in_union]
        valid_cell_idx = neighbor_cell_idx[valid]  # [V, 4]
        valid_union_idx = cell_to_union[valid_cell_idx]  # [V, 4]

        # Sign change check on anchor cell edge — per batch
        anchor_cell_idx = valid_cell_idx[:, 0]  # [V]
        if axis == 0:
            c0, c1 = 0, 1
        elif axis == 1:
            c0, c1 = 0, 3
        else:
            c0, c1 = 0, 4

        sdf_c0 = corner_sdf[:, anchor_cell_idx, c0]  # [B, V]
        sdf_c1 = corner_sdf[:, anchor_cell_idx, c1]  # [B, V]
        sign_change = (sdf_c0 <= 0) != (sdf_c1 <= 0)  # [B, V]

        # Union sign change: any batch element has sign change
        union_sign_change = sign_change.any(dim=0)  # [V]
        if not union_sign_change.any():
            continue

        # Filter to quads with union sign change
        sc_union_idx = valid_union_idx[union_sign_change]  # [Q, 4]
        sc_sign_change = sign_change[:, union_sign_change]  # [B, Q]
        sc_sdf_c0 = sdf_c0[:, union_sign_change]  # [B, Q]

        # Per-batch vertex routing via edge_to_slot: [B, Q, 4]
        local_edges = quad_local_edges[axis]  # [4]
        quad_vert_idx = torch.stack(
            [
                sc_union_idx[:, j].unsqueeze(0).expand(B, -1) * 4
                + edge_to_slot[:, sc_union_idx[:, j], local_edges[j].long()]
                for j in range(4)
            ],
            dim=2,
        )  # [B, Q, 4]

        # Consistent winding: flip where corner 0 is outside (per batch)
        flip = sc_sdf_c0 > 0  # [B, Q]
        flipped = quad_vert_idx[:, :, [0, 3, 2, 1]]
        quad_vert_idx = torch.where(
            flip.unsqueeze(-1).expand_as(quad_vert_idx),
            flipped,
            quad_vert_idx,
        )  # [B, Q, 4]

        all_quads_list.append(quad_vert_idx)
        all_face_valid_list.append(sc_sign_change)  # [B, Q]

    return all_quads_list, all_face_valid_list


def _compact_batch_mesh_outputs(
    flat_verts: Tensor,
    all_faces: Tensor,
    all_face_valid: Tensor,
    device: torch.device,
) -> tuple[list[Tensor], list[Tensor]]:
    """Filter per-batch faces and compact vertices to referenced indices."""
    B = flat_verts.shape[0]
    vertices_list = []
    faces_list = []

    for b in range(B):
        b_valid = all_face_valid[b]  # [Q_total]
        b_faces = all_faces[b][b_valid]  # [Q_b, 4]

        if b_faces.shape[0] == 0:
            vertices_list.append(torch.zeros(0, 3, dtype=flat_verts.dtype, device=device))
            faces_list.append(torch.zeros(0, 4, dtype=torch.long, device=device))
            continue

        # Compact vertices: keep only those referenced by valid faces
        used_idx = torch.unique(b_faces.reshape(-1))
        remap = torch.full((flat_verts.shape[1],), -1, dtype=torch.long, device=device)
        remap[used_idx] = torch.arange(used_idx.shape[0], device=device, dtype=torch.long)

        vertices_list.append(flat_verts[b][used_idx])
        faces_list.append(remap[b_faces])

    return vertices_list, faces_list


def _build_cell_lookup(
    cell_coords: Tensor,
    resolution: int,
    device: torch.device,
) -> tuple[int, Callable[[Tensor], Tensor]]:
    """Build cell lookup function keyed by integer grid hash."""
    R = resolution + 1
    cell_keys = cell_coords[:, 0].long() * R * R + cell_coords[:, 1].long() * R + cell_coords[:, 2].long()
    cell_idx_map = torch.full((R * R * R,), -1, dtype=torch.long, device=device) if R * R * R < 50_000_000 else None

    if cell_idx_map is not None:
        cell_idx_map[cell_keys] = torch.arange(cell_coords.shape[0], device=device, dtype=torch.long)
    else:
        sorted_keys, sort_perm = cell_keys.sort()

    def lookup_cells(keys: Tensor) -> Tensor:
        """Look up cell indices for given keys. Returns -1 for missing."""
        if cell_idx_map is not None:
            valid = (keys >= 0) & (keys < R * R * R)
            result = torch.full_like(keys, -1, dtype=torch.long)
            result[valid] = cell_idx_map[keys[valid]]
            return result
        pos = torch.searchsorted(sorted_keys, keys)
        pos = pos.clamp(max=sorted_keys.shape[0] - 1)
        found = sorted_keys[pos] == keys
        result = torch.full_like(keys, -1, dtype=torch.long)
        result[found] = sort_perm[pos[found]]
        return result

    return R, lookup_cells


def _resolve_union_active_cells(
    corner_sdf: Tensor,
    cell_coords: Tensor,
    lookup_cells: Callable[[Tensor], Tensor],
    problematic_configs: Tensor,
    resolution: int,
    R: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Resolve cases and return active mask and union-active indices."""
    device = corner_sdf.device
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.int32, device=device)
    case_idx = ((corner_sdf <= 0).int() * powers).sum(dim=-1).int()
    resolved = _resolve_ambiguous_cases_batched(
        case_idx,
        cell_coords,
        lookup_cells,
        problematic_configs,
        resolution,
        R,
    )
    active_mask = (resolved != 0) & (resolved != 255)
    union_indices = torch.where(active_mask.any(dim=0))[0]
    return resolved, active_mask, union_indices


def _compute_union_vertices(
    corner_sdf: Tensor,
    corner_positions: Tensor,
    union_indices: Tensor,
    resolved: Tensor,
    active_mask: Tensor,
    edge_corners_table: Tensor,
    dual_points_list: Tensor,
    case_edge_to_slot_table: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute per-slot dual vertices and edge-slot routing for union-active cells."""
    device = corner_sdf.device
    K_union = union_indices.shape[0]

    active_sdf = corner_sdf[:, union_indices]  # [B, K_union, 8]
    active_pos = corner_positions[union_indices]  # [K_union, 8, 3]
    ec = edge_corners_table.long()  # [12, 2]

    # Gather SDF at edge endpoints
    sdf_e0 = active_sdf[:, :, ec[:, 0]]
    sdf_e1 = active_sdf[:, :, ec[:, 1]]
    pos_e0 = active_pos[:, ec[:, 0]]
    pos_e1 = active_pos[:, ec[:, 1]]

    denom = sdf_e0 - sdf_e1
    eps = torch.where(denom >= 0, 1e-8, -1e-8)
    t = (sdf_e0 / (denom + eps)).clamp(0.0, 1.0)
    crossing_pts = pos_e0.unsqueeze(0) + t.unsqueeze(-1) * (pos_e1.unsqueeze(0) - pos_e0.unsqueeze(0))

    active_cases = resolved[:, union_indices]
    edge_to_slot = case_edge_to_slot_table[active_cases.long()]
    dp_masks = dual_points_list[active_cases.long()]

    vertices = torch.zeros(corner_sdf.shape[0], K_union, 4, 3, dtype=corner_sdf.dtype, device=device)
    edge_bits = torch.arange(12, device=device)
    for s in range(4):
        mask_s = ((dp_masks[:, :, s : s + 1] >> edge_bits) & 1).float()
        count = mask_s.sum(dim=-1, keepdim=True).clamp(min=1.0)
        vertices[:, :, s] = (crossing_pts * mask_s.unsqueeze(-1)).sum(dim=2) / count

    inactive = ~active_mask[:, union_indices]
    cell_center = active_pos.mean(dim=1)
    vertices = torch.where(
        inactive[:, :, None, None].expand_as(vertices),
        cell_center[None, :, None, :].expand_as(vertices),
        vertices,
    )
    return vertices, edge_to_slot


def _combine_face_tensors(
    all_quads_list: list[Tensor],
    all_face_valid_list: list[Tensor],
    batch_size: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Combine per-axis face chunks into dense per-batch tensors."""
    if len(all_quads_list) == 0:
        return (
            torch.zeros(batch_size, 0, 4, dtype=torch.long, device=device),
            torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        )
    return torch.cat(all_quads_list, dim=1), torch.cat(all_face_valid_list, dim=1)


def sparse_dual_marching_cubes(
    cell_coords: Tensor,
    corner_sdf: Tensor,
    corner_positions: Tensor,
    resolution: int,
    timer: "MeshTimer | None" = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """
    Sparse Manifold Dual Marching Cubes.

    Multiple dual vertices per cell (1-4) for manifold output. Resolves
    ambiguous C16/C19 cases via neighbor checks, places vertices per
    surface component based on edge partitioning, and routes quad edges
    to the correct vertex slot.

    Args:
        cell_coords:      [K, 3]    int32  — active cell grid coords
        corner_sdf:       [B, K, 8] float  — SDF at 8 corners
        corner_positions: [K, 8, 3] float  — world-space corner positions
        resolution:       int              — grid resolution (for hash keys)

    Returns:
        vertices_list: List of B tensors, each [V_b, 3]
        faces_list:    List of B tensors, each [F_b, 4] (quads)
    """

    device = cell_coords.device
    tables = get_tables(device)
    (
        _corner_offsets,
        edge_corners_table,
        _edge_table,
        _edge_axis,
        quad_cell_offsets,
        dual_points_list,
        problematic_configs,
        _num_dual_points,
        quad_local_edges,
        case_edge_to_slot_table,
    ) = tables

    B = corner_sdf.shape[0]
    K = cell_coords.shape[0]

    if K == 0:
        empty_v = torch.zeros(0, 3, device=device)
        empty_f = torch.zeros(0, 4, dtype=torch.long, device=device)
        return [empty_v] * B, [empty_f] * B

    R, lookup_cells = _build_cell_lookup(cell_coords, resolution, device)

    start_phase = _phase_starter(timer)
    start_phase("dmc_case_resolution")

    resolved, active_mask, union_indices = _resolve_union_active_cells(
        corner_sdf,
        cell_coords,
        lookup_cells,
        problematic_configs,
        resolution,
        R,
    )
    K_union = union_indices.shape[0]

    if K_union == 0:
        empty_v = torch.zeros(0, 3, device=device)
        empty_f = torch.zeros(0, 4, dtype=torch.long, device=device)
        return [empty_v] * B, [empty_f] * B

    start_phase("dmc_vertex_placement")

    vertices, edge_to_slot = _compute_union_vertices(
        corner_sdf,
        corner_positions,
        union_indices,
        resolved,
        active_mask,
        edge_corners_table,
        dual_points_list,
        case_edge_to_slot_table,
    )

    start_phase("dmc_quad_emission")

    # ===================================================================
    # 3h. Union quad emission with per-batch routing
    # ===================================================================
    # Map: cell index (0..K-1) -> union index (0..K_union-1)
    cell_to_union = torch.full((K,), -1, dtype=torch.long, device=device)
    cell_to_union[union_indices] = torch.arange(K_union, device=device, dtype=torch.long)

    all_quads_list, all_face_valid_list = _emit_union_quads(
        cell_coords,
        corner_sdf,
        quad_cell_offsets,
        quad_local_edges,
        edge_to_slot,
        cell_to_union,
        union_indices,
        lookup_cells,
        R,
    )

    start_phase("dmc_per_batch_compact")

    # ===================================================================
    # 3i. Output (backward compatible List[Tensor])
    # ===================================================================
    # Flatten vertices: [B, K_union, 4, 3] -> [B, K_union*4, 3]
    flat_verts = vertices.reshape(B, K_union * 4, 3)

    all_faces, all_face_valid = _combine_face_tensors(
        all_quads_list,
        all_face_valid_list,
        B,
        device,
    )

    vertices_list, faces_list = _compact_batch_mesh_outputs(
        flat_verts,
        all_faces,
        all_face_valid,
        device,
    )

    return vertices_list, faces_list


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def sparse_sdf_to_mesh(
    sdf_fn: Callable[[Tensor], Tensor],
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int,
    batch_size: int,
    lipschitz: float = 1.1,
    coarse_factor: int = 8,
    device: torch.device | None = None,
    triangulate: bool = True,
    max_points_per_chunk: int | None = None,
    timer: "MeshTimer | None" = None,
) -> tuple[list[Tensor], list[Tensor], int]:
    """
    Sparse differentiable SDF-to-mesh via octree + dual marching cubes.

    Chains octree_sdf_eval() and sparse_dual_marching_cubes().

    Args:
        sdf_fn: SDF function. [N, 3] -> [B, N].
        xyz_min: Bounding box min corner.
        xyz_max: Bounding box max corner.
        resolution: Target grid resolution (must be power of 2).
        batch_size: Number of batch elements.
        lipschitz: Lipschitz constant of the SDF (default 1.1).
        coarse_factor: Ratio of target to starting resolution (default 8).
        device: Torch device (default CPU).
        triangulate: If True, split quads into triangles (faces [F, 3]).
        max_points_per_chunk: Max query points per SDF eval chunk (None = no chunking).

    Returns:
        (vertices_list, faces_list, batch_size)
        vertices_list: List of B tensors, each [V_b, 3]
        faces_list: List of B tensors, each [F_b, 3] or [F_b, 4]
    """
    cell_coords, corner_sdf, corner_positions = octree_sdf_eval(
        sdf_fn,
        xyz_min,
        xyz_max,
        resolution,
        batch_size,
        lipschitz=lipschitz,
        coarse_factor=coarse_factor,
        device=device,
        max_points_per_chunk=max_points_per_chunk,
        timer=timer,
    )

    vertices_list, faces_list = sparse_dual_marching_cubes(
        cell_coords,
        corner_sdf,
        corner_positions,
        resolution,
        timer=timer,
    )

    if timer:
        timer.start("triangulate")

    if triangulate:
        tri_faces_list = []
        for faces in faces_list:
            if faces.shape[0] == 0:
                tri_faces_list.append(torch.zeros(0, 3, dtype=torch.long, device=faces.device))
            else:
                # Split each quad [v0, v1, v2, v3] into two triangles
                tri1 = faces[:, [0, 1, 2]]
                tri2 = faces[:, [0, 2, 3]]
                tri_faces_list.append(torch.cat([tri1, tri2], dim=0))
        faces_list = tri_faces_list

    return vertices_list, faces_list, batch_size
