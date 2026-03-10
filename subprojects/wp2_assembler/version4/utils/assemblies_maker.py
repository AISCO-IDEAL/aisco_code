import time
from pathlib import Path
from typing import Any

import yaml

from subprojects.wp2_assembler.version4.components.inner.seat import Seat
from subprojects.wp2_assembler.version4.components.inner.wheels import Wheels
from subprojects.wp2_assembler.version4.components.outer.outer_nbz import aabb_inside_outer_nbz


def check_validity(wheel_margin, seat_margin, assembly_nr: int = 0, nr_of_w: int=3):
    """
    Docstring for check_validity

    :param assembly_nr: Description
    :type assembly_nr: int

    checks validity of a given assembly. checks for collisions, and seat in wheels.
    """

    nr_of_wheels = nr_of_w

    map = generate_assembly_map(assembly_nr, nr_of_w)

    positions = [(map["seat"]["x"], map["seat"]["y"], map["seat"]["z"])]

    for i in range(nr_of_wheels):  # adds wheels
        positions.append((map["wheel_" + str(i)]["x"], map["wheel_" + str(i)]["y"], map["wheel_" + str(i)]["z"]))  # noqa: PERF401

    if all(
        p[0] == positions[1][0] for p in positions[1:]
    ):  # this checks if all wheels are in the same x. Would give problems with bounding box, nothing would be inside 60% of 0!
        return False, map

    bounding_boxes = [seat_margin] + [wheel_margin] * nr_of_wheels

    valid = check_no_collisions(positions, bounding_boxes)
    if not valid:
        return False, map

    positions_2d = [(i[0], i[1]) for i in positions]

    valid = point_in_polygon(positions_2d[0], positions_2d[1:])

    return valid, map

def bbox_list_to_margin_dict(bbox: list[float]) -> dict[str, float]:
    # this function is needed as an adaptor, before we were working with dicts,
    # but now our wheel and seat bounding boxes functions return a list.
    # this function converts the list into a dict.
    # yes, i know there are better ways to do this.

    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    return {
        "x_fwd": float(xmax),
        "x_back": float(-xmin),
        "y_up": float(ymax),
        "y_down": float(-ymin),
        "z_up": float(zmax),
        "z_down": float(-zmin),
    }

def compute_aabb(pos, bb):
    """
    pos = (x, y, z)
    bb = dict with x_fwd, x_back, y_up, y_down, z_up, z_down

    helper function for check no collisions. computes bounding box given xmin xmax etc.
    """
    x, y, z = pos

    return {
        "xmin": x - bb["x_back"],
        "xmax": x + bb["x_fwd"],
        "ymin": y - bb["y_down"],
        "ymax": y + bb["y_up"],
        "zmin": z - bb["z_down"],
        "zmax": z + bb["z_up"],
    }


def aabb_overlap(a, b):
    """
    Docstring for aabb_overlap

    helper function for check no collisions
    """
    return (
        a["xmin"] <= b["xmax"]
        and a["xmax"] >= b["xmin"]
        and a["ymin"] <= b["ymax"]
        and a["ymax"] >= b["ymin"]
        and a["zmin"] <= b["zmax"]
        and a["zmax"] >= b["zmin"]
    )


def check_no_collisions(positions, bounding_boxes):
    """
    positions: list of (x, y, z)
    bounding_boxes: list of bb dicts (same length)

    checks if there is a collision between bounding boxes. used to check if wheels or seat would collide in final assembly
    """
    aabbs = [compute_aabb(pos, bb) for pos, bb in zip(positions, bounding_boxes, strict=False)]

    n = len(aabbs)
    for i in range(n):
        for j in range(i + 1, n):
            if aabb_overlap(aabbs[i], aabbs[j]):
                return False  # collision found

    return True


def point_in_polygon(point, polygon, eps=1e-9):
    """
    point: (x, y)
    polygon: list of (x, y) vertices
    eps: numerical tolerance

    checks if a point is inside a polygon. used to determine if seat is inside the wheels.
    """
    x, y = point
    inside = False
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        # --- check if point is exactly on the edge ---
        cross = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
        if abs(cross) < eps:
            dot = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
            if 0 - eps <= dot <= (x2 - x1) ** 2 + (y2 - y1) ** 2 + eps:
                return True

        # --- ray casting ---
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_int > x:
                inside = not inside

    return inside


def generate_assembly_map(assembly_nr: int = 0, nr_of_w: int = 3):
    """
    Docstring for generate_assembly_map

    :param assembly_nr: nr of assembly
    :type assembly_nr: int

    returns a dict with the parameters of a given assembly
    """
    nr_of_wheels = nr_of_w

    ordered = get_assembly_parameters("seat.combinations")
    unordered = get_assembly_parameters("wheel.combinations")

    if not isinstance(unordered, dict):
        raise TypeError("Expected 'wheels' assembly parameters to be a dict")

    unordered['count'] = nr_of_w    # patch fix. count is now a list, so we have to make it back to a single int.

    map: dict[str, Any] = {  # start of map
        "real id": assembly_nr,
        "seat": {"x": 0, "y": 0, "z": 0},
    }

    for i in range(nr_of_wheels):  # adds wheels
        map["wheel_" + str(i)] = {"x": 0, "y": 0, "z": 0}

    combination = get_combination(assembly_nr, ordered, unordered)

    map["seat"]["x"] = combination["ordered"][0]
    map["seat"]["y"] = combination["ordered"][1]
    map["seat"]["z"] = combination["ordered"][2]

    for i in range(nr_of_wheels):  # adds wheels
        map["wheel_" + str(i)] = {
            "x": combination["unordered"][i][0],
            "y": combination["unordered"][i][1],
            "z": combination["unordered"][i][2],
        }

    # print(map)  # noqa: ERA001
    # save_assembly_parameters(map) # noqa: ERA001
    return map


def save_assembly_parameters(parameters: dict) -> str:
    """
    Save assembly parameters to assembly_map.yaml with the next available ID.

    Args:
        parameters: dict with keys like 'seat: x: '

    Returns:
        str: The ID number assigned (e.g., '042')
    """
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # Read existing data
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}  # Handle empty file

    # Get next ID number
    if assembly_map:
        # Keys are integers like 1, 2, 3 (YAML interprets 001 as 1)
        id_numbers = [int(key) for key in assembly_map]
        next_id = max(id_numbers) + 1
    else:
        next_id = 0

    # Add new entry with integer key (YAML will format it appropriately)
    assembly_map[next_id] = parameters

    # Write back to file
    with config_path.open("w") as file:
        yaml.dump(assembly_map, file, default_flow_style=False, sort_keys=False)

    return str(next_id)


def save_assembly_parameters_batch(batch: dict) -> None:
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    # Read existing data
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    # Get next ID
    next_id = max(int(k) for k in assembly_map) + 1 if assembly_map else 0

    # Insert batch
    for params in batch.values():
        assembly_map[next_id] = params
        next_id += 1

    # Write once
    with config_path.open("w") as file:
        yaml.dump(assembly_map, file, default_flow_style=False, sort_keys=False)


def get_combination(n, ordered, unordered):
    """
    Docstring for get_combination

    :param n: Description
    :param ordered: dict with start, end and how many
    :param unordered: dict with start, end and how many of wheels (order of them doesnt matter, 1,2 = 2,1)

    returns the nth combination of the given paramters
    """

    # ---------- helpers ----------
    def nck(n, k):
        if k < 0 or k > n:
            return 0
        k = min(k, n - k)
        r = 1
        for i in range(1, k + 1):
            r = r * (n - k + i) // i
        return r

    def unrank_comb_with_repetition(rank, m, k):
        res = []
        x = 0
        for i in range(k):
            while True:
                c = nck(m - x + k - i - 2, k - i - 1)
                if rank < c:
                    res.append(x)
                    break
                rank -= c
                x += 1
        return res

    # ---------- ordered count ----------
    ordered_total = 1
    for h in ordered["how_many"]:
        ordered_total *= h

    # ---------- unordered count ----------
    # total unordered states per element (cartesian inside element)
    m = 1
    for h in unordered["how_many"]:
        m *= h

    k = unordered["count"]

    unordered_total = nck(m + k - 1, k)

    total = ordered_total * unordered_total
    if n < 0 or n >= total:
        raise ValueError(f"n={n} out of range (0 <= n < {total})")

    # ---------- split index ----------
    unordered_rank = n // ordered_total
    ordered_rank = n % ordered_total

    # ---------- decode ordered ----------
    ordered_vals = []
    for s, e, h in zip(ordered["start"], ordered["end"], ordered["how_many"], strict=False):
        idx = ordered_rank % h
        ordered_rank //= h
        if h == 1:
            ordered_vals.append(s)
        else:
            step = (e - s) / (h - 1)
            ordered_vals.append(s + idx * step)

    # ---------- decode unordered ----------
    wheel_state_indices = unrank_comb_with_repetition(unordered_rank, m, k)

    # precompute per-wheel cartesian decode
    wheel_vals = []
    for state in wheel_state_indices:
        coords = []
        tmp = state
        for s, e, h in zip(unordered["start"], unordered["end"], unordered["how_many"], strict=False):
            idx = tmp % h
            tmp //= h
            if h == 1:
                coords.append(s)
            else:
                step = (e - s) / (h - 1)
                coords.append(s + idx * step)
        wheel_vals.append(coords)

    return {"ordered": ordered_vals, "unordered": wheel_vals, "total": total}


def get_assembly_parameters(section: str) -> int | float | dict[str, Any]:
    """
    Fetch regulation data from a YAML configuration file.
    Args:
        section (str): The section of the regulations to retrieve, e.g. dimensions.max_length
    Returns:
        int | float | dict: The value of the specified regulation section, or -1 if not found.
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
            print("Error in get_regulation: Error 13 with: " + section)
            # raise Exception("Error in get_regulation: Error 1")  # noqa: ERA001
            return -1  # Key not found or current is not a dict

    return current


def filter_and_reindex_by_outer_nbz(
    assembly_map_path: Path,
    seat_margin: dict[str, float],
    tol: float = 1e-6,
) -> tuple[int, int]:


    start_total = time.time()

    def compute_aabb(pos: tuple[float, float, float], bb: dict[str, float]) -> dict[str, float]:
        x, y, z = pos
        return {
            "xmin": x - bb["x_back"],
            "xmax": x + bb["x_fwd"],
            "ymin": y - bb["y_down"],
            "ymax": y + bb["y_up"],
            "zmin": z - bb["z_down"],
            "zmax": z + bb["z_up"],
        }

    print("Loading assembly map...")

    with assembly_map_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    items = sorted(data.items(), key=lambda kv: int(kv[0]))
    total = len(items)

    print(f"Loaded {total} assemblies")

    kept_entries: list[dict] = []

    last_print = time.time()

    for i, (_yaml_id, entry) in enumerate(items):

        seat = entry["seat"]
        seat_pos = (float(seat["x"]), float(seat["y"]), float(seat["z"]))

        aabb = compute_aabb(seat_pos, seat_margin)

        if aabb_inside_outer_nbz(entry, aabb, tol):
            kept_entries.append(entry)

        # print progress every ~2 seconds
        if time.time() - last_print > 2:
            progress = 100 * i / total
            elapsed = time.time() - start_total

            print(
                f"Progress: {i}/{total} ({progress:.1f}%) | "
                f"kept={len(kept_entries)} | "
                f"time={elapsed:.1f}s"
            )

            last_print = time.time()

    print("Filtering finished")

    print("Reindexing entries...")

    new_data = {i: kept_entries[i] for i in range(len(kept_entries))}

    print("Writing YAML...")

    with assembly_map_path.open("w") as f:
        yaml.dump(new_data, f, default_flow_style=False, sort_keys=False)

    kept = len(kept_entries)
    removed = len(items) - kept

    print(
        f"Done | kept={kept} removed={removed} | "
        f"total_time={time.time() - start_total:.2f}s"
    )

    return kept, removed

def make_assemblys(nr_of_w):

    wheel_margin = bbox_list_to_margin_dict(Wheels().bounding_box_list())
    seat_margin = bbox_list_to_margin_dict(Seat().bounding_box_list())

    ordered = get_assembly_parameters("seat.combinations")
    unordered = get_assembly_parameters("wheel.combinations")

    if not isinstance(unordered, dict):
        raise TypeError("Expected 'wheels' assembly parameters to be a dict")

    unordered["count"] = nr_of_w    # patch fix. count is now a list, so we have to make it nack to a single int.

    total = get_combination(0, ordered, unordered)["total"]
    print(total)

    start = time.time()
    last = start

    batch = {}
    batch_count = 0

    for i in range(total):
        if i % 500 == 0:
            print(i, time.time() - last)
            last = time.time()

            if batch:
                save_assembly_parameters_batch(batch)
                batch.clear()

        valid, map = check_validity(wheel_margin, seat_margin, i, nr_of_w)
        if valid:
            batch[batch_count] = map
            batch_count += 1

    # Final flush
    if batch:
        save_assembly_parameters_batch(batch)

    # NEW: filter + reindex based on outer NBZ containment

    assembly_map_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"

    kept, removed = filter_and_reindex_by_outer_nbz(assembly_map_path, seat_margin)
    print(f"outer_nbz filter done: kept={kept}, removed={removed}")

    print(time.time() - start)

def make_assemblys_for_n_wheels():
    nr_of_w = get_assembly_parameters("wheel.combinations.count")
    for i in nr_of_w:
        make_assemblys(i)


if __name__ == "__main__":
    """
    All of the [ # noqa: ERA001 ] are to ignore ruff warnings for unused code
    the code is indeed unused, but i use it sometimes manually to test changes.
    """
    start = time.time()

    make_assemblys_for_n_wheels()

    end = time.time()
    print(end - start)

    # ordered = get_assembly_parameters("seat")         # noqa: ERA001
    # unordered = get_assembly_parameters("wheels")     # noqa: ERA001

    # # for i in range(20):
    # #     cfg = get_combination(i, ordered, unordered)       # noqa: ERA001
    # #     print(cfg)      # noqa: ERA001

    # total = get_combination(0, ordered, unordered)["total"]           # noqa: ERA001
    # legal_assemblies = []     # noqa: ERA001
    # start = time.time()       # noqa: ERA001
    # for i in range(total):
    #     if check_validity(i):
    #         legal_assemblies.append(i)        # noqa: ERA001
    # end = time.time()     # noqa: ERA001
    # print(end-start)      # noqa: ERA001
    # print(total, len(legal_assemblies))       # noqa: ERA001
    # # for i in legal_assemblies:
    # #     save_assembly_parameters(generate_assembly_map(i))      # noqa: ERA001
    # # generate_assembly_map()     # noqa: ERA001
    # # print(check_validity())     # noqa: ERA001
