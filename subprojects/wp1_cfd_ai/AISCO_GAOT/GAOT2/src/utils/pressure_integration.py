import torch
from torch import Tensor


def compute_vertex_areas(pos: Tensor, face: Tensor) -> Tensor:
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    face_areas = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1)

    vertex_areas = torch.zeros(pos.shape[0], device=pos.device, dtype=pos.dtype)
    third = face_areas / 3.0
    vertex_areas.scatter_add_(0, face[0], third)
    vertex_areas.scatter_add_(0, face[1], third)
    vertex_areas.scatter_add_(0, face[2], third)
    return vertex_areas


def compute_face_normals(pos: Tensor, face: Tensor) -> Tensor:
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return cross / (cross.norm(dim=-1, keepdim=True) + 1e-12)


def pressure_to_drag(
    pred_pressure: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int = 0,
) -> Tensor:
    """
    _validate_pressure_drag_inputs(
        pred_pressure=pred_pressure,
        normals=normals,
        face=face,
        pos=pos,
        drag_direction=drag_direction,
    )

    raise_if_normals_locally_inconsistent(normals, pos, k=12, cosine_threshold=-0.2)
    """

    p = pred_pressure.squeeze(-1) if pred_pressure.dim() > 1 else pred_pressure

    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    # 1. Compute unnormalized face normal vectors
    area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F, 3]

    # 2. ROBUST ORIENTATION: Match the dataset's outward normals
    outward_n = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3.0
    dot_prod = (area_normals * outward_n).sum(dim=-1, keepdim=True)

    # If dot product is negative, our normal is pointing inward. Flip it.
    area_normals = torch.where(dot_prod < 0, -area_normals, area_normals)

    # 3. Integrate: Pressure pushes INWARD, so we use -p_face
    p_face = (p[face[0]] + p[face[1]] + p[face[2]]) / 3.0
    n_dir_area = area_normals[:, drag_direction]

    drag = (-p_face * n_dir_area).sum()
    print(f"Area Normals: {area_normals}")
    print(f"Outward Normals: {outward_n}")
    print(f"Pressure per Face: {p_face}")
    print(f"Directional Area Normals: {n_dir_area}")
    print(f"Computed Drag: {drag}")
    return drag


def pressure_shear_to_drag(
    pred_pressure: Tensor,
    pred_shear: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int = 0,
) -> Tensor:
    p = pred_pressure.squeeze(-1) if pred_pressure.dim() > 1 else pred_pressure

    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]

    # 1. Compute unnormalized face normal vectors
    area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)  # [F, 3]

    # 2. ROBUST ORIENTATION: Match the dataset's outward normals
    outward_n = (normals[face[0]] + normals[face[1]] + normals[face[2]]) / 3.0
    dot_prod = (area_normals * outward_n).sum(dim=-1, keepdim=True)
    area_normals = torch.where(dot_prod < 0, -area_normals, area_normals)

    # 3. --- PRESSURE DRAG ---
    p_face = (p[face[0]] + p[face[1]] + p[face[2]]) / 3.0
    n_dir_area = area_normals[:, drag_direction]

    # Pressure pushes INWARD, so we use -p_face
    pressure_drag = (-p_face * n_dir_area).sum()

    # 4. --- FRICTION DRAG ---
    # OpenFOAM's wallShearStress is already the exact force vector (kinematic).
    # We just average it to the face and multiply by the exact area!
    tau_face = (pred_shear[face[0]] + pred_shear[face[1]] + pred_shear[face[2]]) / 3.0
    face_areas = area_normals.norm(dim=-1)  # [F]

    tau_dir = tau_face[:, drag_direction]
    friction_drag = (-tau_dir * face_areas).sum()

    return pressure_drag + friction_drag


def drag_to_cd(
    drag: Tensor,
    rho: float = 1.225,
    u_inf: float = 38.889,
    A_ref: float = 2.16,
) -> Tensor:
    q = 0.5 * rho * u_inf**2 * A_ref
    return drag / q


def _validate_pressure_drag_inputs( #Noqa: PLR0912, PLR0915
    pred_pressure: Tensor,
    normals: Tensor,
    face: Tensor,
    pos: Tensor,
    drag_direction: int,
) -> None:
    # --- type checks ---
    for name, x in {
        "pred_pressure": pred_pressure,
        "normals": normals,
        "face": face,
        "pos": pos,
    }.items():
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}.")

    if not isinstance(drag_direction, int):
        raise TypeError(f"drag_direction must be an int in {{0,1,2}}, got {type(drag_direction).__name__}.")

    # --- device checks ---
    devices = {pred_pressure.device, normals.device, face.device, pos.device}
    if len(devices) != 1:
        raise ValueError(
            "All tensors must be on the same device, but got:\n"
            f"  pred_pressure: {pred_pressure.device}\n"
            f"  normals:       {normals.device}\n"
            f"  face:          {face.device}\n"
            f"  pos:           {pos.device}"
        )

    # --- drag direction ---
    if drag_direction not in (0, 1, 2):
        raise ValueError(f"drag_direction must be 0, 1, or 2, got {drag_direction}.")

    # --- pos shape ---
    if pos.dim() != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}.")
    n_vertices = pos.shape[0]
    if n_vertices == 0:
        raise ValueError("pos is empty: expected at least one vertex.")

    # --- normals shape ---
    if normals.dim() != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must have shape [N, 3], got {tuple(normals.shape)}.")
    if normals.shape[0] != n_vertices:
        raise ValueError(
            "normals and pos must refer to the same number of vertices, but got:\n"
            f"  pos.shape[0]     = {n_vertices}\n"
            f"  normals.shape[0] = {normals.shape[0]}"
        )

    # --- pred_pressure shape ---
    if pred_pressure.dim() == 1:
        if pred_pressure.shape[0] != n_vertices:
            raise ValueError(
                "pred_pressure with shape [N] must match pos.shape[0], but got:\n"
                f"  pred_pressure.shape = {tuple(pred_pressure.shape)}\n"
                f"  pos.shape           = {tuple(pos.shape)}"
            )
    elif pred_pressure.dim() == 2:
        if pred_pressure.shape[1] != 1:
            raise ValueError(f"pred_pressure with 2 dims must have shape [N, 1], got {tuple(pred_pressure.shape)}.")
        if pred_pressure.shape[0] != n_vertices:
            raise ValueError(
                "pred_pressure with shape [N,1] must match pos.shape[0], but got:\n"
                f"  pred_pressure.shape = {tuple(pred_pressure.shape)}\n"
                f"  pos.shape           = {tuple(pos.shape)}"
            )
    else:
        raise ValueError(f"pred_pressure must have shape [N] or [N,1], got {tuple(pred_pressure.shape)}.")

    # --- face shape ---
    if face.dim() != 2:
        raise ValueError(f"face must be a rank-2 tensor with shape [3, F], got {tuple(face.shape)}.")

    if face.shape[0] != 3:
        if face.shape[1] == 3:
            raise ValueError(
                "face appears to have shape [F, 3], but this function expects [3, F].\n"
                f"Got face.shape = {tuple(face.shape)}.\n"
                "You probably need to transpose it before calling:\n"
                "    face = face.T"
            )
        raise ValueError(f"face must have shape [3, F], got {tuple(face.shape)}.")

    n_faces = face.shape[1]
    if n_faces == 0:
        raise ValueError("face is empty: expected at least one triangle.")

    # --- face dtype ---
    if face.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError(f"face must contain integer indices, got dtype {face.dtype}.")

    # --- index range checks ---
    face_min = int(face.min().item())
    face_max = int(face.max().item())

    if face_min < 0:
        raise ValueError(f"face contains negative vertex indices. Minimum index found: {face_min}.")

    if face_max >= n_vertices:
        raise ValueError(
            "face contains out-of-range vertex indices, but valid range is "
            f"[0, {n_vertices - 1}]. Maximum index found: {face_max}."
        )

    # --- dtype sanity for geometry tensors ---
    for name, x in {
        "pred_pressure": pred_pressure,
        "normals": normals,
        "pos": pos,
    }.items():
        if not x.dtype.is_floating_point:
            raise TypeError(f"{name} must be a floating-point tensor, got dtype {x.dtype}.")

    # --- optional NaN / Inf checks ---
    for name, x in {
        "pred_pressure": pred_pressure,
        "normals": normals,
        "pos": pos,
    }.items():
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains NaN or Inf values.")

    # --- degenerate face warning as error ---
    v0 = pos[face[0]]
    v1 = pos[face[1]]
    v2 = pos[face[2]]
    area_normals = 0.5 * torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_areas = torch.linalg.norm(area_normals, dim=-1)

    if (face_areas == 0).any():
        bad = torch.nonzero(face_areas == 0, as_tuple=False).flatten()[:10].tolist()
        raise ValueError(f"Degenerate triangle(s) detected: at least one face has zero area. Example face indices: {bad}")


def check_normal_orientation_consistency(
    normals: Tensor,
    pos: Tensor,
    k: int = 12,
    cosine_threshold: float = -0.2,
    max_report: int = 20,
) -> dict:
    """
    Check whether each normal is similarly oriented to nearby normals.

    Since nearby vertices are not assumed to be adjacent in the tensor ordering,
    neighbors are found geometrically using Euclidean distance in `pos`.

    Parameters
    ----------
    normals : Tensor
        Shape [N, 3]. Vertex normals.
    pos : Tensor
        Shape [N, 3]. Vertex positions.
    k : int
        Number of geometric neighbors to inspect for each vertex.
    cosine_threshold : float
        A pair of normals is considered suspicious if their cosine similarity
        is below this threshold. Typical values:
            0.0   -> flag anything more than 90° apart
           -0.2   -> slightly tolerant
           -0.5   -> only flag strongly opposing normals
    max_report : int
        Maximum number of inconsistent vertices/pairs to include in the report.

    Returns
    -------
    dict with keys:
        ok : bool
            True if no suspicious orientations were found.
        n_vertices : int
        n_checked_pairs : int
        n_bad_pairs : int
        bad_fraction : float
        n_bad_vertices : int
        bad_vertices : Tensor
            Indices of vertices that participate in at least one suspicious pair.
        sample_bad_pairs : list[dict]
            Small human-readable sample of problematic pairs.

    Notes
    -----
    - This checks *local consistency*, not whether normals point outward globally.
    - It works best on smooth surfaces.
    - On sharp creases, corners, or intentionally discontinuous normals,
      some flagged pairs may be expected.
    """

    _validate_normal_checker_inputs(normals, pos, k, cosine_threshold, max_report)

    # Normalize normals to unit length
    n = normals / torch.linalg.norm(normals, dim=1, keepdim=True).clamp_min(1e-12)

    N = pos.shape[0]
    k_eff = min(k, N - 1)

    # Pairwise distances in geometry space, not tensor order
    dists = torch.cdist(pos, pos)  # [N, N]
    dists.fill_diagonal_(float("inf"))

    # Indices of k nearest geometric neighbors for each vertex
    neighbor_idx = torch.topk(dists, k=k_eff, dim=1, largest=False).indices  # [N, k]

    # Gather neighbor normals
    neighbor_normals = n[neighbor_idx]  # [N, k, 3]

    # Cosine similarity between each normal and its local neighbors
    cos_sim = (n.unsqueeze(1) * neighbor_normals).sum(dim=-1)  # [N, k]

    # Suspicious if a nearby normal points too much the other way
    bad_mask = cos_sim < cosine_threshold  # [N, k]

    # Count stats
    n_checked_pairs = int(N * k_eff)
    n_bad_pairs = int(bad_mask.sum().item())
    bad_fraction = n_bad_pairs / max(n_checked_pairs, 1)

    bad_vertices_mask = bad_mask.any(dim=1)
    bad_vertices = torch.nonzero(bad_vertices_mask, as_tuple=False).flatten()
    n_bad_vertices = int(bad_vertices.numel())

    # Build a small report of bad pairs
    bad_locs = torch.nonzero(bad_mask, as_tuple=False)
    sample_bad_pairs = []
    for row in bad_locs[:max_report]:
        i = int(row[0].item())
        j_local = int(row[1].item())
        j = int(neighbor_idx[i, j_local].item())
        sample_bad_pairs.append(
            {
                "vertex_i": i,
                "vertex_j": j,
                "distance": float(dists[i, j].item()),
                "cosine_similarity": float(cos_sim[i, j_local].item()),
            }
        )

    return {
        "ok": n_bad_pairs == 0,
        "n_vertices": N,
        "n_checked_pairs": n_checked_pairs,
        "n_bad_pairs": n_bad_pairs,
        "bad_fraction": bad_fraction,
        "n_bad_vertices": n_bad_vertices,
        "bad_vertices": bad_vertices,
        "sample_bad_pairs": sample_bad_pairs,
    }


def _validate_normal_checker_inputs( #Noqa: PLR0912
    normals: Tensor,
    pos: Tensor,
    k: int,
    cosine_threshold: float,
    max_report: int,
) -> None:
    if not isinstance(normals, torch.Tensor):
        raise TypeError(f"normals must be a torch.Tensor, got {type(normals).__name__}.")
    if not isinstance(pos, torch.Tensor):
        raise TypeError(f"pos must be a torch.Tensor, got {type(pos).__name__}.")

    if normals.dim() != 2 or normals.shape[1] != 3:
        raise ValueError(f"normals must have shape [N, 3], got {tuple(normals.shape)}.")
    if pos.dim() != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape [N, 3], got {tuple(pos.shape)}.")

    if normals.shape[0] != pos.shape[0]:
        raise ValueError(f"normals and pos must have the same number of rows, got {normals.shape[0]} and {pos.shape[0]}.")

    if normals.shape[0] < 2:
        raise ValueError("Need at least 2 vertices to compare nearby normals.")

    if not normals.dtype.is_floating_point:
        raise TypeError(f"normals must be floating point, got {normals.dtype}.")
    if not pos.dtype.is_floating_point:
        raise TypeError(f"pos must be floating point, got {pos.dtype}.")

    if normals.device != pos.device:
        raise ValueError(f"normals and pos must be on the same device, got {normals.device} and {pos.device}.")

    if not torch.isfinite(normals).all():
        raise ValueError("normals contains NaN or Inf.")
    if not torch.isfinite(pos).all():
        raise ValueError("pos contains NaN or Inf.")

    normal_norms = torch.linalg.norm(normals, dim=1)
    if (normal_norms < 1e-12).any():
        idx = torch.nonzero(normal_norms < 1e-12, as_tuple=False).flatten()[:10].tolist()
        raise ValueError(f"normals contains near-zero vectors at indices {idx}.")

    if not isinstance(k, int):
        raise TypeError(f"k must be an int, got {type(k).__name__}.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    if not isinstance(cosine_threshold, (int, float)):
        raise TypeError(f"cosine_threshold must be a number, got {type(cosine_threshold).__name__}.")
    if cosine_threshold < -1.0 or cosine_threshold > 1.0:
        raise ValueError(f"cosine_threshold must lie in [-1, 1], got {cosine_threshold}.")

    if not isinstance(max_report, int):
        raise TypeError(f"max_report must be an int, got {type(max_report).__name__}.")
    if max_report < 1:
        raise ValueError(f"max_report must be >= 1, got {max_report}.")


def raise_if_normals_locally_inconsistent(
    normals: Tensor,
    pos: Tensor,
    k: int = 12,
    cosine_threshold: float = -0.2,
    max_report: int = 10,
) -> None:
    """
    Convenience wrapper: runs the checker and raises ValueError if suspicious
    local orientation mismatches are found.
    """
    result = check_normal_orientation_consistency(
        normals=normals,
        pos=pos,
        k=k,
        cosine_threshold=cosine_threshold,
        max_report=max_report,
    )

    if not result["ok"]:
        lines = [
            "Normals are not consistently oriented with nearby normals.",
            f"Vertices: {result['n_vertices']}",
            f"Checked neighbor pairs: {result['n_checked_pairs']}",
            f"Bad pairs: {result['n_bad_pairs']}",
            f"Bad fraction: {result['bad_fraction']:.6f}",
            f"Bad vertices: {result['n_bad_vertices']}",
            "Example inconsistent pairs:",
        ]
        for item in result["sample_bad_pairs"]:
            lines.append(
                f"  i={item['vertex_i']}, j={item['vertex_j']}, "
                f"distance={item['distance']:.6g}, "
                f"cos={item['cosine_similarity']:.6f}"
            )
        raise ValueError("\n".join(lines))
