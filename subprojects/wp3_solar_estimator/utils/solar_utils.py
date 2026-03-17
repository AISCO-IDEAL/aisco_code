import torch


def averaging_solar_quality(normals: torch.Tensor, sun_directions: torch.Tensor) -> torch.Tensor:
    """
    Aggregates the solar quality values along the given dimension by taking the cumulative sum.
    This is a simple way to aggregate the solar quality values, but it can be replaced with a more sophisticated method if needed.

    Args:
        normals: A tensor of shape (N, 3) OR (N, T, 3) containing the NORMALISED normal vectors of the mesh vertices.
        sun_directions: A tensor of shape (M, 3) containing the NORMALISED sun direction vectors you want to aggregate for each face
    """
    # if shapes of sun vectors is (N,T,3):
    if sun_directions.ndim == 3:
        # remove excess dimension by pputting it T into N dimension
        sun_directions = sun_directions.view(-1, 3)
        # simple dot product between normals and all sun directions
        illumination_quality = torch.einsum("ij,kj->ik", normals, sun_directions)
    elif sun_directions.ndim == 2:
        # simple dot product between normals and all sun directions
        illumination_quality = torch.einsum("ij,kj->ik", normals, sun_directions)

    else:
        raise ValueError("sun_directions must be either (N,T,3) or (N,3)")

    return illumination_quality.mean(dim=1)
