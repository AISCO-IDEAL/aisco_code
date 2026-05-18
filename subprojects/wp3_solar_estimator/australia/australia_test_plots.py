import numpy as np
from matplotlib import pyplot as plt

from subprojects.wp3_solar_estimator.australia.route_v3 import RouteV3


def test_route_loader():  # noqa: PLR0915
    route_sweep = RouteV3(mode="time_sweep")
    route_sweep.debug_print()
    points = route_sweep.get_route_points()
    print(points)

    plt.scatter(points[:, 1], points[:, 0])
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route Points")
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/route_points.png")
    plt.close()

    plt.figure()
    plt.scatter(points[:, 1], points[:, 0])
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

    # Solar vectors from time_sweep: shape (N, T, 3) — coloured by time sample
    solar_vectors_sweep = route_sweep.solar_vectors()  # (N*T, 3) flat from cache
    T = len(route_sweep.times)
    N = route_sweep.num_points
    solar_vectors_nt3 = solar_vectors_sweep.view(N, T, 3).numpy()

    norm = plt.Normalize(0, T - 1)
    cmap = plt.colormaps["viridis"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    zeros = np.zeros(N)
    for i in range(T):
        color = cmap(norm(i))
        ax.quiver(
            zeros,
            zeros,
            zeros,
            solar_vectors_nt3[:, i, 0],
            solar_vectors_nt3[:, i, 1],
            solar_vectors_nt3[:, i, 2],
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
    ax.set_title("Solar Vectors in Car Body Frame (time_sweep)")
    ax.view_init(elev=10, azim=0)
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/solar_vectors_X.png")
    ax.view_init(elev=00, azim=90)
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/solar_vectors_Y.png")
    ax.view_init(elev=90, azim=0)
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/solar_vectors_Z.png")
    plt.close()

    route_race = RouteV3(mode="race")
    points = route_race.get_route_points()
    hours = np.array([t.hour + t.minute / 60 for t in route_race.times])

    plt.figure(figsize=(4.5, 6))
    plt.scatter(points[:, 1], points[:, 0], c=hours, cmap="viridis")
    plt.colorbar(label="Time of day (hours)")
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Route Points Colored by Time of Day (race)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/route_points_time.png")

    solar_vectors_race = route_race.solar_vectors().numpy()  # (N, 3)
    N = solar_vectors_race.shape[0]
    norm = plt.Normalize(0, N - 1)
    cmap = plt.colormaps["viridis"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(N):
        ax.quiver(
            0,
            0,
            0,
            solar_vectors_race[i, 0],
            solar_vectors_race[i, 1],
            solar_vectors_race[i, 2],
            color=cmap(norm(i)),
            alpha=0.4,
            length=1.0,
        )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Route progression", shrink=0.6)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-0.3, 1.0)
    ax.set_xlabel("Forward/X")
    ax.set_ylabel("Left/Y")
    ax.set_zlabel("Up/Z")
    ax.set_title("Solar Vectors Colored by Race Progression (race)")
    ax.view_init(elev=10, azim=0)
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/time_solar_vector_route_points_X.png")
    ax.view_init(elev=0, azim=90)
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/time_solar_vector_route_points_Y.png")
    ax.view_init(elev=90, azim=0)
    plt.savefig("subprojects/wp3_solar_estimator/australia/data/time_solar_vector_route_points_Z.png")
    plt.close()

    print("\n--- RouteV3 (mode='race') ---")
    route_race.summary()
    print("  subsample(6) shape:", route_race.solar_vectors(subsample=6).shape)
    print("  subsample(6) vectors:\n", route_race.solar_vectors(subsample=6).numpy())

    print("\n--- RouteV3 (mode='time_sweep') ---")
    route_sweep.summary()

    return route_race, route_sweep


if __name__ == "__main__":
    route_race, route_sweep = test_route_loader()
    print("\nSaved verification plots")

    for label, route in [("race", route_race), ("time_sweep", route_sweep)]:
        print(f"\n=== {label} ===")
        route.summary()
        print("First 5 route points and corresponding solar vectors:")
        for j in range(min(5, route.num_points)):
            print(f"  Point {j + 1}: {route.route_points[j].tolist()}  solar: {route.solar_vectors()[j].tolist()}")
        print("  Full solar_vectors() shape:", route.solar_vectors().shape)
