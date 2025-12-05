import numpy as np
import scipy as sp

from subprojects.wp2_assembler.version3.utils.reg_imp import get_regulation


def uniform_distribution(low: float, high: float, size: int = 1) -> float | np.ndarray:
    rng = np.random.default_rng()
    samples = rng.uniform(low, high, size=size)
    return samples[0] if size == 1 else samples


def truncated_gamma(
    x: float = 0.0,
    y: float = 100.0,
    k: float = get_regulation("wheel.gamma_distribution.shape1"),
    theta: float = get_regulation("wheel.gamma_distribution.scale1"),
    c: float = get_regulation("wheel.gamma_distribution.cut_off_point"),
    size: int = 1,
) -> float | np.ndarray:
    """
    Generate samples from gamma distribution truncated to [-y + x, 0]. to then shift by a to [x, y].

    Args:
        shape: Shape parameter (k) - controls distribution curve shape
                See: wp2_assembler/version2/data/imagePlots/gamma_distribution_comparison.png
        scale: Scale parameter (theta) - controls distribution spread
                See: wp2_assembler/version2/data/imagePlots/gamma_distribution_comparison.png
        x: Upper truncation bound (samples will be in [0, x])
        size: Number of samples to generate (default: 1)

    Returns:
        Single float if size=1, otherwise numpy array of samples

    Raises:
        ValueError: If parameters are invalid
    """
    # ensure x is smaller than y
    if x > y:
        raise ValueError("x should be smaller than y")  # noqa: TRY003

    gamma_dist = sp.stats.gamma(a=k, scale=theta)
    rng = np.random.default_rng()

    # CDF value at truncation point
    fa = gamma_dist.cdf(c)

    # Generate uniform samples in [-fa, 0] and inverse transform
    u = rng.uniform(0, fa, size=size)
    samples = np.array(-gamma_dist.ppf(u))
    # shift all entries by c and then multiply by (y-x)/c to scale to [x,y]
    samples = (samples + c) * ((y - x) / c) + y

    return samples[0] if size == 1 else samples
