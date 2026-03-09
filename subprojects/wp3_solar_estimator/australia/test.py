import numpy as np
from matplotlib import pyplot as plt

from subprojects.wp3_solar_estimator.australia.route_v1 import Route


def test_route_loader():
    route = Route()
    route.debug_print()
    points = route.get_route_points()
    print(points)
    # Plot longitude vs latitude, ensure spacing is correct and not distorted by aspect ratio
    plt.scatter(points[:, 1], points[:, 0])  # Plot longitude vs latitude
    plt.axis("equal")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route Points")
    # save plot to file
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/route_points.png")
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
        scale=0.7,  # paradoxically you need to increase for smaller arrows
        color="red",
    )
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route Points with Headings")
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/route_points_headings.png")

    solar_vectors = route.get_route_solar_vecs()  # shape (N, T, 3)
    # Draw solar vectors as arrows from origin, colored by time sample index.
    # Viewed from the +x (forward) axis so the left-up plane is front-facing.
    T = solar_vectors.shape[1]
    norm = plt.Normalize(0, T - 1)
    cmap = plt.colormaps["viridis"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    zeros = np.zeros(solar_vectors.shape[0])
    for i in range(T):
        color = cmap(norm(i))
        ax.quiver(
            zeros,
            zeros,
            zeros,  # origins at (0, 0, 0)
            solar_vectors[:, i, 0],  # forward component
            solar_vectors[:, i, 1],  # left component
            solar_vectors[:, i, 2],  # up component
            color=color,
            alpha=0.4,
            length=1.0,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time sample index", shrink=0.6)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-0.3, 1.0)
    ax.set_xlabel("Forward/X")
    ax.set_ylabel("Left/Y")
    ax.set_zlabel("Up/Z")
    ax.set_title("Solar Vectors in Car Body Frame")
    ax.view_init(elev=10, azim=0)  # look from the +x (forward) axis toward origin
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/solar_vectors_X.png")
    ax.view_init(elev=10, azim=90)  # look from the +y (left) axis toward origin
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/solar_vectors_Y.png")
    ax.view_init(elev=90, azim=0)  # look from the -z (backward) axis toward origin
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/solar_vectors_Z.png")
    plt.close()
    # just test that this runs without error, will verify values in future tests
    return route


# Run to verify route is accurate, see data folder
if __name__ == "__main__":
    route = test_route_loader()
    print("Saved verification plots")
    print("Solar Vectors for route points:")
    print(route.get_route_solar_vecs())
