from pathlib import Path
import yaml


def get_regulation(section: str) -> int:
    """
    Fetch regulation data from a YAML configuration file.
    Args:
        section (str): The section of the regulations to retrieve, e.g. dimensions.max_length
    Returns:
        int: The value of the specified regulation section, or -1 if not found.
    """
    keys = section.split(".")
    # Get the path relative to this file's location
    config_path = Path(__file__).parent.parent / "config" / "example_regulations.yaml"
    with config_path.open() as file:
        regulations = yaml.safe_load(file)
    for key in keys:
        print(key)
        regulations = regulations.get(key, -1)
    return regulations
