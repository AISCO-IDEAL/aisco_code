

def generate_assemblys():

    d = {
        "seat": {
            "x": 0,
            "y": 0,
            "z": 0
        },
        "canopy": {
            "x": 0,
            "y": 0,
            "z": 0
        }
    }

def save_assembly_parameters(parameters: dict) -> str:
    """
    Save assembly parameters to assembly_map.yaml with the next available ID.

    Args:
        parameters: dict with keys like 'wheel_base_width', 'wheel_base_length', etc.

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
        next_id = 1

    # Add new entry with integer key (YAML will format it appropriately)
    assembly_map[next_id] = parameters

    # Write back to file
    with config_path.open("w") as file:
        yaml.dump(assembly_map, file, default_flow_style=False, sort_keys=False)

    return str(next_id)

def get_combination(n, start, end, how_many):
    """
    n: integer index
    start, end, how_many: lists, one per dimension
    """
    total = 1
    for h in how_many:
        total *= h

    if n < 0 or n >= total:
        raise ValueError(f"n={n} out of range (0 <= n < {total})")

    values = []

    for i in range(len(start)):
        base = how_many[i]
        idx = n % base
        n //= base

        if how_many[i] == 1:
            val = start[i]
        else:
            step = (end[i] - start[i]) / (how_many[i] - 1)
            val = start[i] + idx * step

        values.append(val)

    return values

if __name__ == "__main__":
    start = [0, 10]
    end = [1, 20]
    how_many = [1, 3]

    for n in range(9):
        print(n, get_combination(n, start, end, how_many))