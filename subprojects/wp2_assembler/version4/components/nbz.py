from pathlib import Path
import yaml
import time
from subprojects.wp2_assembler.version3.components.inner.seat import Seat
from subprojects.wp2_assembler.version3.components.inner.wheels import Wheels
import cadquery as cq

def check_validity(assembly_nr: int = 0):
    """
    Docstring for check_validity
    
    :param assembly_nr: Description
    :type assembly_nr: int

    checks validity of a given assembly. checks for collisions, and seat in wheels. 
    """
    wheel_margin = get_assembly_parameters("wheel_margin")
    seat_margin = get_assembly_parameters("seat_margin")
    nr_of_wheels = get_assembly_parameters("wheels.count")

    map = generate_assembly_map(assembly_nr)

    positions = [(map["seat"]["x"],map["seat"]["y"],map["seat"]["z"])]

    for i in range(nr_of_wheels):   #adds wheels 
        positions.append((map["wheel_"+str(i)]["x"],map["wheel_"+str(i)]["y"],map["wheel_"+str(i)]["z"]))

    if all(p[0] == positions[1][0] for p in positions[1:]): # this checks if all wheels are in the same x. Would give problems with bounding box, nothing would be inside 60% of 0!
        return False

    bounding_boxes = [seat_margin]+[wheel_margin]*nr_of_wheels

    valid = check_no_collisions(positions, bounding_boxes)
    if valid == False:
        return False
    
    positions_2d = [(i[0], i[1]) for i in positions]
    
    valid = point_in_polygon(positions_2d[0], positions_2d[1:])
    
    if valid == False:
        return False

    return True

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
        a["xmin"] <= b["xmax"] and a["xmax"] >= b["xmin"] and
        a["ymin"] <= b["ymax"] and a["ymax"] >= b["ymin"] and
        a["zmin"] <= b["zmax"] and a["zmax"] >= b["zmin"]
    )

def check_no_collisions(positions, bounding_boxes):
    """
    positions: list of (x, y, z)
    bounding_boxes: list of bb dicts (same length)

    checks if there is a collision between bounding boxes. used to check if wheels or seat would collide in final assembly
    """
    aabbs = [
        compute_aabb(pos, bb)
        for pos, bb in zip(positions, bounding_boxes)
    ]

    n = len(aabbs)
    for i in range(n):
        for j in range(i + 1, n):
            if aabb_overlap(aabbs[i], aabbs[j]):
                return False # collision found

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
            if 0 - eps <= dot <= (x2 - x1)**2 + (y2 - y1)**2 + eps:
                return True

        # --- ray casting ---
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_int > x:
                inside = not inside

    return inside

def generate_assembly_map(assembly_nr: int = 0):
    """
    Docstring for generate_assembly_map
    
    :param assembly_nr: nr of assembly
    :type assembly_nr: int
    
    returns a dict with the parameters of a given assembly
    """
    nr_of_wheels = get_assembly_parameters("wheels.count")

    ordered = get_assembly_parameters("seat")
    unordered = get_assembly_parameters("wheels")
    
    map = {             #start of map
        "real id": assembly_nr,
        "seat": {
            "x": 0,
            "y": 0,
            "z": 0
        },
    }

    for i in range(nr_of_wheels):   #adds wheels 
        map["wheel_"+str(i)] = {
            "x": 0,
            "y": 0,
            "z": 0
        }
    
    combination = get_combination(assembly_nr, ordered, unordered)
    
    map["seat"]["x"] = combination["ordered"][0]
    map["seat"]["y"] = combination["ordered"][1]
    map["seat"]["z"] = combination["ordered"][2]

    for i in range(nr_of_wheels):   #adds wheels 
        map["wheel_"+str(i)] = {
            "x": combination["unordered"][i][0],
            "y": combination["unordered"][i][1],
            "z": combination["unordered"][i][2]
        }

    # print(map)
    # save_assembly_parameters(map)
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
    if assembly_map:
        next_id = max(int(k) for k in assembly_map) + 1
    else:
        next_id = 0

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
    def nCk(n, k):
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
                c = nCk(m - x + k - i - 2, k - i - 1)
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
    unordered_total = nCk(m + k - 1, k)

    total = ordered_total * unordered_total
    if n < 0 or n >= total:
        raise ValueError(f"n={n} out of range (0 <= n < {total})")

    # ---------- split index ----------
    unordered_rank = n // ordered_total
    ordered_rank = n % ordered_total

    # ---------- decode ordered ----------
    ordered_vals = []
    for s, e, h in zip(ordered["start"], ordered["end"], ordered["how_many"]):
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
        for s, e, h in zip(unordered["start"], unordered["end"], unordered["how_many"]):
            idx = tmp % h
            tmp //= h
            if h == 1:
                coords.append(s)
            else:
                step = (e - s) / (h - 1)
                coords.append(s + idx * step)
        wheel_vals.append(coords)

    return {
        "ordered": ordered_vals,
        "unordered": wheel_vals,
        "total": total
    }

def get_assembly_parameters(section: str) -> int | float:
    """
    Fetch regulation data from a YAML configuration file.
    Args:
        section (str): The section of the regulations to retrieve, e.g. dimensions.max_length
    Returns:
        int | float: The value of the specified regulation section, or -1 if not found.
    """
    keys = section.split(".")
    config_path = Path(__file__).parent.parent / "config" / "Assembly_parameters.yaml"

    with config_path.open() as file:
        data = yaml.safe_load(file)

    # Traverse the nested dictionary structure
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            print("Error in get_regulation: Error 1 with: "+ section)
            #raise Exception("Error in get_regulation: Error 1")
            return -1  # Key not found or current is not a dict

    return current

def inner_nbz(assembly_nr: int = 0):
    config_path = Path(__file__).parent.parent / "config" / "assembly_map.yaml"
    
    #print(config_path)
    with config_path.open("r") as file:
        assembly_map = yaml.safe_load(file) or {}

    params = assembly_map[assembly_nr]

    wheel = Wheels().create_wheel()

    seat = Seat().create_seat()

    assembly = cq.Assembly(name="inner nbz " + str(assembly_nr))

    assembly.add(
        seat,
        loc=cq.Location(
            cq.Vector(params["seat"]["x"], params["seat"]["y"], params["seat"]["z"])
        ),
        name="Seat",
    )

    for key in params:
        if key.startswith("wheel_"):
            assembly.add(
            wheel,
            loc=cq.Location(
                cq.Vector(params[key]["x"], params[key]["y"], params[key]["z"])
            ),
            name=key,
            )

    return assembly

def make_assemblys():
    ordered = get_assembly_parameters("seat")
    unordered = get_assembly_parameters("wheels")

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

        if check_validity(i):
            batch[batch_count] = generate_assembly_map(i)
            batch_count += 1

    # Final flush
    if batch:
        save_assembly_parameters_batch(batch)

    print(time.time() - start)


#TODO: maybe inner nbz as list??? 
# [[type,x,y,z]] * elements

if __name__ == "__main__":
   #make_assemblys()
    import random
    from ocp_vscode import show
    # for i in range(527):
    #     show(inner_nbz(i))
    print("showing")
    start = time.time()
    #show([inner_nbz(random.randint(0, 2337)) for i in range(1)])
    show(inner_nbz(0))
    end = time.time()
    print(end-start)
    # ordered = get_assembly_parameters("seat")
    # unordered = get_assembly_parameters("wheels")

    # # for i in range(20):
    # #     cfg = get_combination(i, ordered, unordered)
    # #     print(cfg)

    # total = get_combination(0, ordered, unordered)["total"]
    # legal_assemblies = []
    # start = time.time()
    # for i in range(total):
    #     if check_validity(i):
    #         legal_assemblies.append(i)
    # end = time.time()
    # print(end-start)
    # print(total, len(legal_assemblies))
    # # for i in legal_assemblies:
    # #     save_assembly_parameters(generate_assembly_map(i))
    # # generate_assembly_map()
    # # print(check_validity())

