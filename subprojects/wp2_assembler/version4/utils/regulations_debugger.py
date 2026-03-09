"""

READ ME!

This file search for all uses in version 4 of the parameters defined in example rules.
it says where there are used and what line.
Usefull to see where they lead to, or if they are unusued.

"""

from pathlib import Path

import yaml

REGULATIONS_FILE = Path("subprojects/wp2_assembler/version4/config/example_regulations.yaml")
SEARCH_ROOT = Path("subprojects/wp2_assembler/version4")
FILE_EXTENSIONS = {".py", ".yaml", ".yml", ".txt", ".md"}

def flatten_dict(data: object, prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}

    if isinstance(data, dict):
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_dict(value, new_prefix))
    elif isinstance(data, list):
        for i, value in enumerate(data):
            new_prefix = f"{prefix}[{i}]"
            flat.update(flatten_dict(value, new_prefix))
    else:
        flat[prefix] = data

    return flat


def load_regulation_keys(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    return flatten_dict(data)


def search_key_usage(search_root: Path, key: str) -> list[str]:
    matches: list[str] = []

    for path in search_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in FILE_EXTENSIONS:
            continue
        if path == REGULATIONS_FILE:
            continue

        try:
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    if key in line:
                        matches.append(f"{path}:{line_number}: {line.rstrip()}")
        except (UnicodeDecodeError, OSError):
            continue

    return matches


def main() -> None:
    if not REGULATIONS_FILE.exists():
        raise FileNotFoundError(f"Could not find: {REGULATIONS_FILE}")

    regulations = load_regulation_keys(REGULATIONS_FILE)

    if not regulations:
        print("No regulations found.")
        return

    print(f"Scanning {len(regulations)} regulation entries from:")
    print(f"  {REGULATIONS_FILE}")
    print("Only checking inside:")
    print(f"  {SEARCH_ROOT}\n")

    for key, value in regulations.items():
        print("=" * 100)
        print(f"REGULATION: {key}")
        print(f"VALUE: {value}")

        matches = search_key_usage(SEARCH_ROOT, key)

        if matches:
            print("USED IN:")
            for match in matches:
                print(f"  {match}")
        else:
            print("USED IN:")
            print("  NOT FOUND")

        print()


if __name__ == "__main__":

    """

    READ ME!

    This file search for all uses in version 4 of the parameters defined in example rules.
    it says where there are used and what line.
    Usefull to see where they lead to, or if they are unusued.

    """

    main()
