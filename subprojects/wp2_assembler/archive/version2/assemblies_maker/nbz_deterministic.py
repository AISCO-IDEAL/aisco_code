from pathlib import Path

import cadquery as cq

ASSEMBLIES_DIR = Path(__file__).resolve().parent / "assemblies"
BASE_FILENAME = "inner_sym_nbz"


def inner_nbz_deterministic(file_number: int) -> cq.Shape:
    """
    Load a deterministic NBZ STEP file and return it as the original CadQuery Shape.

    Args:
        file_number: The numeric suffix of the STEP file to load (e.g. 1 -> inner_sym_nbz_001.step).

    Returns:
        cq.Shape representing the NBZ geometry stored in the STEP file.
    """
    if file_number < 1:
        raise ValueError("file_number must be at least 1.")

    filename = f"{BASE_FILENAME}_{file_number:03d}.step"
    file_path = ASSEMBLIES_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"STEP file not found: {file_path}")

    result = cq.importers.importStep(str(file_path))
    if hasattr(result, "val"):
        return result.val()
    return result
