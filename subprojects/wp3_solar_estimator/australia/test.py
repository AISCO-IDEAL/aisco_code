import numpy as np
from matplotlib import pyplot as plt

from subprojects.wp3_solar_estimator.australia.route import Route


def test_route_loader():
    route = Route()
    points = route.get_route_points()
    print(points)
    # Plot longitude vs latitude, ensure spacing is correct and not distorted by aspect ratio
    plt.scatter(points[:, 1], points[:, 0])  # Plot longitude vs latitude
    plt.axis("equal")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route Points")
    # save plot to file
    plt.savefig("/cluster/home/aalthaus/AISCO_code/subprojects/wp3_solar_estimator/australia/data/route_points.png")
    plt.close()

    # do plot but with heading vectors drawn as arrows (ensure arrows are visible by scaling them and map appropriately)

    plt.figure()
    plt.scatter(points[:, 1], points[:, 0])  # Plot longitude vs latitude
    # Heading is given as degrees, 0 deg is north, goes counterclockwise
    plt.quiver(
        points[:, 1],
        points[:, 0],
        np.sin(np.radians(points[:, 2])),
        np.cos(np.radians(points[:, 2])),
        angles="xy",
        scale_units="xy",
        scale=0.5,
        color="red",
    )
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route Points with Headings")
    plt.savefig("/cluster/home/aalthaus/AISCO_code/subprojects/wp3_solar_estimator/australia/data/route_points_headings.png")


if __name__ == "__main__":
    test_route_loader()
