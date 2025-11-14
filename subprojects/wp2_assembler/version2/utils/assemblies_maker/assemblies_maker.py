import sys
from pathlib import Path
from typing import List

import cadquery as cq

# Add version2 directory to sys.path so we can import sibling modules.
VERSION2_DIR = Path(__file__).resolve().parents[2]
if str(VERSION2_DIR) not in sys.path:
    sys.path.insert(0, str(VERSION2_DIR))

from components.nbz import inner_nbz2_sym

# Directory where the generated STEP files will be written.
ASSEMBLIES_DIR = Path(__file__).resolve().parent / "assemblies"

BASE_FILENAME = "inner_sym_nbz"


def ensure_output_dir() -> Path:
    """Create the assemblies directory if needed and return its path."""
    ASSEMBLIES_DIR.mkdir(parents=True, exist_ok=True)
    return ASSEMBLIES_DIR


def export_step_files(count: int) -> list[Path]:
    """
    Build `count` symmetric inner NBZ assemblies and export each as a STEP file.

    Returns:
        List of Paths for the exported STEP files.
    """
    if count < 1:
        raise ValueError("Count must be greater than zero.")

    output_dir = ensure_output_dir()
    exported_files: list[Path] = []

    start_index = _next_file_index(output_dir, BASE_FILENAME)
    end_index = start_index + count

    for index in range(start_index, end_index):
        assembly = inner_nbz2_sym()
        shape = _assembly_to_shape(assembly)
        filename = f"{BASE_FILENAME}_{index:03d}.step"
        file_path = output_dir / filename
        # Export as STEP regardless of extension casing to avoid relying on defaults.
        cq.exporters.export(shape, str(file_path), cq.exporters.ExportTypes.STEP)
        exported_files.append(file_path)
        print(f"Created assembly {filename}")

    return exported_files


def _next_file_index(output_dir: Path, base_filename: str) -> int:
    """
    Determine the next available sequence number for the given assembly type,
    ensuring we don't overwrite existing files (e.g., inner_sym_nbz_001.step).
    """
    prefix = f"{base_filename}_"
    max_index = 0
    for file in output_dir.glob(f"{base_filename}_*.step"):
        stem = file.stem
        if not stem.startswith(prefix):
            continue
        suffix = stem[len(prefix) :]
        if suffix.isdigit():
            max_index = max(max_index, int(suffix))

    return max_index + 1


def _assembly_to_shape(obj):
    """
    Convert CadQuery Assemblies to a Shape that exporters can handle.
    """
    if isinstance(obj, cq.Assembly):
        return obj.toCompound()
    return obj


def _prompt_count() -> int:
    """Ask the user how many assemblies to build and validate the answer."""
    while True:
        raw = input("How many assemblies do you want to make? ").strip()
        try:
            count = int(raw)
        except ValueError:
            print("Please enter a whole number (e.g., 5).")
            continue
        if count < 1:
            print("Count must be at least 1.")
            continue
        return count


if __name__ == "__main__":
    print("NBZ Assemblies Maker")
    count = _prompt_count()
    files = export_step_files(count)
    print(f"Exported {len(files)} STEP file(s) to {ASSEMBLIES_DIR}:")
    for file in files:
        print(f" - {file.name}")
