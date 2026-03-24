import torch


def averaging_solar_quality(
    normals: torch.Tensor, sun_directions: torch.Tensor, batch_size: int | None = None, debug_print: bool = False
) -> torch.Tensor:
    """
    Aggregates the solar quality values along the given dimension by taking the cumulative sum.
    This is a simple way to aggregate the solar quality values, but it can be replaced with a more sophisticated method if needed.

    Eats Memory of order (N x S) if no batching is used, so only use for small N and S (S=M x T).

    Args:
        normals: A tensor of shape (N, 3) OR (N, 3) containing the NORMALISED normal vectors of the mesh vertices.
        sun_directions: A tensor of shape (M, T, 3) or (S, 3) containing the NORMALISED sun direction vectors you want to aggregate for each face
    """

    # if shapes of sun vectors is (N,T,3):
    if sun_directions.ndim == 3:
        # remove excess dimension by putting T into N dimension
        sun_directions = sun_directions.view(-1, 3)
    elif sun_directions.ndim != 2:
        raise ValueError("sun_directions must be either (M,T,3) or (S,3)")

    print("Sun directions shape:", sun_directions.shape) if debug_print else None
    print("Normals shape:", normals.shape) if debug_print else None

    if batch_size is None:
        # simple dot product between normals and all sun directions
        illumination_quality = torch.einsum("ij,kj->ik", normals, sun_directions)
        print("Illumination quality shape:", illumination_quality.shape) if debug_print else None
        return illumination_quality.mean(dim=1)

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer or None")

    total = torch.zeros(normals.shape[0], device=normals.device, dtype=normals.dtype)
    count = 0
    for start in range(0, sun_directions.shape[0], batch_size):
        end = min(start + batch_size, sun_directions.shape[0])
        sun_chunk = sun_directions[start:end]
        illumination_chunk = torch.einsum("ij,kj->ik", normals, sun_chunk)
        total += illumination_chunk.sum(dim=1)
        count += sun_chunk.shape[0]

    return total / count
