import math

import numpy as np


def wheel_turning_angle(
    outer_wheel: list[float],
    back_wheel: list[float],
    inner_wheel: list[float],
    turning_radius: float,
    perpendicular_to_rear_wheel: list[float] | None = None,
) -> tuple[float, float]:
    """
    Returns the steering angle of the inner front wheel.

    :param outer_wheel: position of the outermost front wheel
    :type outer_wheel: list[float]
    :param back_wheel: position of the back wheel (on same side as outer_wheel)
    :type back_wheel: list[float]
    :param inner_wheel: position of the inner front wheel
    :type inner_wheel: list[float]
    :param perpendicular_to_rear_wheel: vector along vehicle forward direction
    :type perpendicular_to_rear_wheel: list[float]
    :param turning_radius: turning radius of the outer front wheel
    :type turning_radius: float
    :return: steering angle of the inner wheel (degrees)
    :rtype: float
    """
    if perpendicular_to_rear_wheel is None:
        perpendicular_to_rear_wheel = [1, 0]
    outer_wheel_arr = np.array(outer_wheel, dtype=float)
    back_wheel_arr = np.array(back_wheel, dtype=float)
    inner_wheel_arr = np.array(inner_wheel, dtype=float)
    d = np.array(perpendicular_to_rear_wheel, dtype=float)

    # Find instantaneous center of rotation (ICR) as intersection of:
    #  - line through back_wheel in direction d
    #  - circle centered at outer_wheel with radius turning_radius
    a = np.dot(d, d)
    b = 2 * np.dot(d, back_wheel_arr - outer_wheel_arr)
    c = np.dot(back_wheel_arr - outer_wheel_arr, back_wheel_arr - outer_wheel_arr) - turning_radius**2

    disc = b * b - 4 * a * c
    if disc < 0:
        raise Exception("Problem in turning radius: No intersection, length of car longer than turning radius.")  # noqa: TRY002

    sqrt_disc = np.sqrt(disc)
    t1 = (-b + sqrt_disc) / (2 * a)
    t2 = (-b - sqrt_disc) / (2 * a)

    Q1 = back_wheel_arr + t1 * d  # first possible ICR
    Q2 = back_wheel_arr + t2 * d  # second possible ICR

    # Vectors from inner wheel to possible ICRs
    v1 = Q1 - inner_wheel_arr
    v2 = Q2 - inner_wheel_arr

    def angle(u, v):
        num = np.dot(u, v)
        den = np.linalg.norm(u) * np.linalg.norm(v)
        if den == 0:
            raise ValueError("Zero length vector in angle computation.")
        cos_theta = num / den
        cos_theta = max(-1.0, min(1.0, cos_theta))  # numerical safety
        return math.degrees(math.acos(cos_theta))

    angle1 = angle(d, v1)
    angle2 = angle(d, v2)

    # Pick the smaller magnitude steering angle (physically relevant solution)
    return (angle1, angle2)


if __name__ == "__main__":
    outer = [0.0, 5.0]
    back = [0.0, 0.0]
    inner = [0.0, 4.0]  # for example, inner wheel closer to center line
    radius = 10
    print(wheel_turning_angle(outer, back, inner, radius))
