import random
from dataclasses import asdict, fields, replace


def randomize(
    base_params,
    constraints: dict[str, tuple[float, float]],
    random_magnitude: float = 0.2,
    lock: list[str] = [],
):
    """
    Varies numeric fields of a @dataclass taking into account __constraints__.

    Args:
    base_params: @dataclass object with optional __constraints__.
    random_magnitude: relative change +/- x%.
    lock: List of locked parameter names.

    Returns:
    New @dataclass object.
    """
    params_dict = asdict(base_params)
    new_params = {}

    for field in fields(base_params):
        name = field.name
        value = params_dict[name]

        if name in lock or not isinstance(value, (int, float)):
            new_params[name] = value
            continue

        delta = value * random.uniform(-random_magnitude, random_magnitude)
        candidate = value + delta
        if abs(value) < 1e-6 and name in constraints:
            min_r, max_r = constraints[name]
            candidate = random.uniform(
                random_magnitude * min_r, random_magnitude * max_r
            )

        if name in constraints:
            min_val, max_val = constraints[name]
            candidate = min(max(candidate, min_val), max_val)

        new_params[name] = candidate

    return replace(base_params, **new_params)
