from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from .reg_imp import get_regulation


def distribution_error(type: str) -> None:
    """
    Exception for unknown distribution types.
    Args:
        type: The unknown distribution type
    """
    msg = f"Unknown distribution type: {type}"
    raise ValueError(msg)


def visualizeResults(num_samples: int = 10000, type: str = "truncated_gammas"):  # noqa: N802
    """
    Visualize samples from specified distribution type.
    Args:
        num_samples: Number of samples to generate
        type: Type of distribution to visualize
    Returns: None
    Raises:
        UnknownDistributionTypeError: If distribution type is unknown
    """
    if type == "truncated_gammas":
        samples1 = truncated_gamma("l", size=num_samples)
        samples2 = truncated_gamma("w", size=num_samples)
    else:
        distribution_error(type)
        return

    plt.hist(samples1, bins=30, density=True, alpha=0.5, label="Length")
    plt.hist(samples2, bins=30, density=True, alpha=0.5, label="Width")
    plt.title(f"Histogram of {type} samples")
    plt.xlabel("Value")
    plt.ylabel("Density")

    output_dir = Path(__file__).parent.parent / "data" / "imagePlots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "gamma_distribution_check.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved as: {output_path}")
    plt.close()  # Clean up memory


def _validate_params(a: float, shape: float, scale: float) -> None:
    """
    Validate parameters for truncated gamma distribution.

    Args:
        shape: Shape parameter (k)
        scale: Scale parameter (theta)
        a: Upper truncation bound

    Raises:
        ValueError: If any parameter is invalid
    """
    if shape <= 0:
        msg = f"shape must be positive, got {shape}"
        raise ValueError(msg)
    if scale <= 0:
        msg = f"scale must be positive, got {scale}"
        raise ValueError(msg)
    if a <= 0:
        msg = f"truncation bound 'a' must be positive, got {a}"
        raise ValueError(msg)


def truncated_gamma(a: str, size: int = 1) -> float | np.ndarray:
    """
    Generate samples from gamma distribution truncated to [-a, 0]. to then shift by a to [0, a].

    Args:
        shape: Shape parameter (k) - controls distribution curve shape
               See: wp2_assembler/version2/data/imagePlots/gamma_distribution_comparison.png
        scale: Scale parameter (theta) - controls distribution spread
               See: wp2_assembler/version2/data/imagePlots/gamma_distribution_comparison.png
        a: Upper truncation bound (samples will be in [0, a])
        size: Number of samples to generate (default: 1)

    Returns:
        Single float if size=1, otherwise numpy array of samples

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> truncated_gamma(2.0, 2.0, 100.0, size=5)
        array([1.38, 1.95, 9.06, 14.70, 1.78])
    """

    if a == "l":
        a = (  # value = 4600
            get_regulation("no_build_zones.outer_prism.length")  # value = 5800
            - get_regulation("wheel.base.max_front_margin")  # value = 600
            - get_regulation("wheel.base.max_rear_margin")  # value = 600
        )
        k = get_regulation("wheel.gamma_distribution.shape1")
        theta = get_regulation("wheel.gamma_distribution.scale1")
    elif a == "w":
        a = (  # value = 1052
            get_regulation("no_build_zones.outer_prism.width") / 2  # value = 1150
            - get_regulation("wheel.margin_axial")  # value = 50
            - get_regulation("wheel.tire_width") / 2  # value = 48
        )
        k = get_regulation("wheel.gamma_distribution.shape2")
        theta = get_regulation("wheel.gamma_distribution.scale2")

    _validate_params(a, k, theta)  # Validate parameters

    gamma_dist = sp.stats.gamma(a=k, scale=theta)
    rng = np.random.default_rng()

    # CDF value at truncation point
    fa = gamma_dist.cdf(20)

    # Check if CDF is valid
    if fa <= 0 or not np.isfinite(fa):
        msg = f"Invalid CDF value: fa={fa}. Check parameters: shape={k}, scale={theta}, a={a}"
        raise ValueError(msg)

    # Generate uniform samples in [-fa, 0] and inverse transform
    u = rng.uniform(0, fa, size=size)
    samples = -gamma_dist.ppf(u)
    # shift all entries by 20 and then multiply by a/20 to scale to [0,a]
    samples = (samples + 20) * (a / 20)

    return samples[0] if size == 1 else samples
