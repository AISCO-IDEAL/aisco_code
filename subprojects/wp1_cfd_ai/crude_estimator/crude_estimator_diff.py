import torch

# --- Default weights from the original script ---
# These are defined from MIN to MAX axis
_DEFAULT_WEIGHTS_LIST = [
    8.0, 7.39, 6.78, 2.07, 1.2, 1.2, 1.1, 0.75, 0.75, 0.9,
    1.05, 1.7, 2.0, 2.3, 3.0, 3.5, 3.7, 4.8, 6.25, 8.0,
]
# Epsilon for numerical stability (avoiding div/sqrt of zero)
EPSILON = 1e-10

def calculateCdProxy_differentiable(
    vertices: torch.Tensor,
    slicing_axis: str = "x",
    slicing_direction: str = "-",
    weights: list[float] | None = None,
    N_SLICES: int = 20,
    slice_bounds: tuple[float, float] | None = None,
    num_std_devs: float = 3.0,
    k_steepness: float = 50.0,
) -> torch.Tensor | None:
    """
    Calculates a fully differentiable Cd proxy from a vertex point cloud.

    This function can operate in two modes:
    1. FIXED (slice_bounds is provided): Slices within a fixed box.
    2. DYNAMIC (slice_bounds is None): Differentiably calculates the
       object's bounds using mean and standard deviation.

    Args:
        vertices: The input mesh vertices. Shape: `[N_verts, 3]`.
        slicing_axis: The axis to slice along ('x', 'y', or 'z').
        slicing_direction: The direction to slice ('+' or '-').
        weights: A list of floats for slice weights.
        N_SLICES: The number of slices to divide the mesh into.
        slice_bounds: A tuple (min, max) defining the FIXED coordinate
            range to slice over. If None, bounds are calculated
            dynamically. Defaults to None.
        num_std_devs: Used only if slice_bounds is None. Defines the
            bounds as (center +/- num_std_devs * std_dev).
        k_steepness: Controls the "steepness" of the soft slice masks.

    Returns:
        The calculated normalized loss as a single-item PyTorch tensor.
    """

    # --- 0. Setup ---
    device = vertices.device
    dtype = vertices.dtype

    # --- Handle weights ---
    manual_weights_to_use = weights
    if manual_weights_to_use is None:
        manual_weights_to_use = _DEFAULT_WEIGHTS_LIST

    if len(manual_weights_to_use) != N_SLICES:
        print(f"FATAL ERROR: Weights list has {len(manual_weights_to_use)} items, but N_SLICES is {N_SLICES}.")
        return None

    weights_to_use_list = manual_weights_to_use
    if slicing_direction == "-":
        weights_to_use_list = manual_weights_to_use[::-1]
    
    weights_tensor = torch.tensor(
        weights_to_use_list, dtype=dtype, device=device
    )

    # --- 1. Set up axis indices ---
    slicing_axis_lower = slicing_axis.lower()
    if slicing_axis_lower == "x":
        axis_idx, area_idx_1, area_idx_2 = 0, 1, 2
    elif slicing_axis_lower == "y":
        axis_idx, area_idx_1, area_idx_2 = 1, 0, 2
    elif slicing_axis_lower == "z":
        axis_idx, area_idx_1, area_idx_2 = 2, 0, 1
    else:
        print(f"FATAL ERROR: Slicing axis '{slicing_axis}' is invalid.")
        return None

    # --- 2. Differentiable "Soft" Slicing ---
    
    # Get all vertex coordinates for the slicing axis
    vert_coords = vertices[:, axis_idx] # Shape: [N_verts]

    # --- THIS IS THE STANDARD DEVIATION BOX METHOD ---
    if slice_bounds is not None:
        # 1. FIXED mode: Use the user-provided bounds
        min_bound, max_bound = slice_bounds[0], slice_bounds[1]
    else:
        # 2. DYNAMIC mode: Differentiably calculate bounds
        center = torch.mean(vert_coords)
        spread = torch.std(vert_coords)
        
        # Add epsilon to spread to prevent 0 spread on a flat plane
        min_bound = center - num_std_devs * (spread + EPSILON)
        max_bound = center + num_std_devs * (spread + EPSILON)

    # Create the slicing "ruler" based on the bounds
    split_points = torch.linspace(
        min_bound, max_bound, N_SLICES + 1, device=device, dtype=dtype
    )
    # --- END OF THE METHOD IMPLEMENTATION ---
    
    all_slice_weights = []
    for i in range(N_SLICES):
        a_start = split_points[i]
        a_end = split_points[i + 1]
        min_mask = torch.sigmoid((vert_coords - a_start) * k_steepness)
        max_mask = torch.sigmoid((a_end - vert_coords) * k_steepness)
        in_slice_weight = min_mask * max_mask
        all_slice_weights.append(in_slice_weight)

    # --- 3. Differentiable Frontal Area Calculation ---
    slice_frontal_areas = []
    coords_1 = vertices[:, area_idx_1] # e.g., Y-coords
    coords_2 = vertices[:, area_idx_2] # e.g., Z-coords

    for in_slice_weight in all_slice_weights:
        total_weight = in_slice_weight.sum() + EPSILON

        # Calculate spread on Area_1 axis
        mean_1 = (coords_1 * in_slice_weight).sum() / total_weight
        variance_1 = (((coords_1 - mean_1)**2) * in_slice_weight).sum() / total_weight
        
        # Calculate spread on Area_2 axis
        mean_2 = (coords_2 * in_slice_weight).sum() / total_weight
        variance_2 = (((coords_2 - mean_2)**2) * in_slice_weight).sum() / total_weight

        # Differentiable "area" = product of "spreads" (std devs)
        spread_1 = torch.sqrt(variance_1 + EPSILON)
        spread_2 = torch.sqrt(variance_2 + EPSILON)
        
        proxy_area = spread_1 * spread_2
        slice_frontal_areas.append(proxy_area)

    all_slice_areas = torch.stack(slice_frontal_areas) # Shape: [N_SLICES]

    # --- 4. Final Weighted Loss ---
    final_weighted_loss = torch.dot(all_slice_areas, weights_tensor)

    # --- 5. Differentiable Normalization ---
    # (Weighted Loss * A_max) / (A_norm**2)
    
    A_max_proxy = torch.max(all_slice_areas)
    A_norm_proxy = torch.mean(all_slice_areas)

    numerator = final_weighted_loss * A_max_proxy
    denominator = (A_norm_proxy**2) + EPSILON
    
    normalized_loss = numerator / denominator
    
    return normalized_loss