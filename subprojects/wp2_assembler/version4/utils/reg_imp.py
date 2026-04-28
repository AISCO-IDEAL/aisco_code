from pathlib import Path

import yaml


def get_regulation(section: str) -> int | float:
    """
    Fetch regulation data from a YAML configuration file.
    Args:
        section (str): The section of the regulations to retrieve, e.g. dimensions.max_length
    Returns:
        int | float: The value of the specified regulation section, or -1 if not found.
    """
    keys = section.split(".")
    config_path = Path(__file__).parent.parent / "config" / "regulations_and_parameters.yaml"

    with config_path.open() as file:
        data = yaml.safe_load(file)

    # Traverse the nested dictionary structure
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            print("Error in get_regulation: Error 11 with: " + section)
            # raise Exception("Error in get_regulation: Error 1")  # noqa: ERA001
            return -1  # Key not found or current is not a dict

    # Check if we ended up at a numeric value
    if isinstance(current, (int, float)):
        return current

    print("Error in get_regulation: Error 20 with: " + section)
    # raise Exception("Error in get_regulation: Error 2") # noqa: ERA001
    return -1  # Final value is not a number


def get_outer_prism_bounds_mm() -> dict[str, tuple[float, float]]:
    """Return outer-prism spatial bounds in millimetres.

    Reads dimensions directly from ``regulations_and_parameters.yaml`` so that
    any change to the regulation values propagates automatically to all callers.

    Returns:
        Dict with four keys:
          - ``"x"``: ``(lo, hi)`` along the car-length axis (symmetric, ±length/2).
          - ``"y"``: ``(lo, hi)`` along the lateral axis (symmetric, ±width/2).
          - ``"z_sampling"``: ``(lo, hi)`` for domain sampling — includes a
            below-ground buffer equal to ``ground_clearance`` for eikonal
            regularisation.
          - ``"z_mesh"``: ``(lo, hi)`` for mesh extraction — shapes exist above
            the ground plane (Z ≥ 0).
    """
    half_length = float(get_regulation("no_build_zones.outer_prism.length")) / 2.0
    half_width = float(get_regulation("no_build_zones.outer_prism.width")) / 2.0
    height = float(get_regulation("no_build_zones.outer_prism.height"))
    ground_clearance = float(get_regulation("no_build_zones.outer_prism.ground_clearance"))
    return {
        "x": (-half_length, half_length),
        "y": (-half_width, half_width),
        "z_sampling": (-ground_clearance, height),
        "z_mesh": (0.0, height),
    }


def scale_outer_prism(outer_prism: dict[str, tuple[float, float]], scale_factor: float) -> dict[str, tuple[float, float]]:
    """Scale outer_prism bounds symmetrically around each axis centre.

    Each axis half-span is multiplied by *scale_factor*; the centre of each
    axis is preserved.  For axes that are already symmetric about zero (x, y)
    this is equivalent to a simple multiplicative scale.  For asymmetric axes
    (z_sampling, z_mesh) the centre is preserved instead of the origin.

    Args:
        outer_prism: Dict from ``get_outer_prism_bounds_mm()`` with keys
            ``'x'``, ``'y'``, ``'z_sampling'``, ``'z_mesh'``.
        scale_factor: Multiplicative factor for each half-span.  ``1.0`` returns
            the input unchanged.  ``> 1.0`` expands; ``< 1.0`` shrinks.

    Returns:
        New outer_prism dict with scaled bounds.  If *scale_factor* is ``1.0``,
        returns the original dict (no copy).
    """
    if scale_factor == 1.0:
        return outer_prism
    result: dict[str, tuple[float, float]] = {}
    for key, (lo, hi) in outer_prism.items():
        centre = (lo + hi) / 2.0
        half_span = (hi - lo) / 2.0 * scale_factor
        result[key] = (centre - half_span, centre + half_span)
    return result


if __name__ == "__main__":
    # Example usage - get values from nested YAML structure in test - print values 3 2 1
    pass
