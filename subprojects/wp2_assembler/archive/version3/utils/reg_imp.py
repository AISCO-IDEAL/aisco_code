import warnings
from pathlib import Path

import yaml


def get_regulation(section: str) -> int | float:
    """
    Fetch regulation data from a YAML configuration file.
    Args:
        section (str): The section of the regulations to retrieve, e.g. dimensions.max_length
    Returns:
        int | float: The value of the specified regulation section, or -1 if not found.
    """
    keys = section.split(".")
    config_path = Path(__file__).parent.parent / "config" / "example_regulations.yaml"

    with config_path.open() as file:
        data = yaml.safe_load(file)

    # Traverse the nested dictionary structure
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            warnings.warn(
                "\n\n\nError in get_regulation: Error 1\nError in get_regulation: Error 1\nError in get_regulation: Error 1\n\n\n",
                stacklevel=2,
            )
            return -1  # Key not found or current is not a dict

    # Check if we ended up at a numeric value
    if isinstance(current, (int, float)):
        return current

    warnings.warn(
        "\n\n\nError in get_regulation: Error 2\nError in get_regulation: Error 2\nError in get_regulation: Error 2\n\n\n",
        stacklevel=2,
    )
    return -1  # Final value is not a number

if __name__ == "__main__":
    # Example usage - get values from nested YAML structure in test - print values 3 2 1
    print(get_regulation("test.text.secondIndent.thirdIndent.drink"))
    print(get_regulation("test.text.secondIndent.food"))
    print(get_regulation("test.firstIndent.launch"))
