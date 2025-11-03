import os
from pathlib import Path
import numpy as np


def test(x: int) -> str:
    # kdiehl
    """Beispiel-Funktion: gibt eine stringbasierte Bestätigung zurück.

    Args:
        x: eine Ganzzahl (wird hier nur zu Demonstrationszwecken angezeigt)

    Returns:
        Eine Bestätigungszeichenkette mit dem übergebenen Wert.
    """

    # Determine data output directory portably:
    # 1) Allow user override via environment variable AISCO_DATA_DIR
    env = os.getenv("AISCO_DATA_DIR")
    if env:
        outdir = Path(env)
    else:
        # 2) Try to locate repository root (ancestor that contains 'subprojects')
        here = Path(__file__).resolve()
        repo_root = None
        for parent in here.parents:
            if (parent / "subprojects").is_dir():
                repo_root = parent
                break
        # Fallback: use a fixed ancestor level if repo_root not found
        if repo_root is None:
            repo_parent_level = 3
            repo_root = (
                here.parents[repo_parent_level]
                if len(here.parents) > repo_parent_level
                else here.parents[-1]
            )

        outdir = repo_root / "subprojects" / "wp5" / "data"

    outdir.mkdir(parents=True, exist_ok=True)

    # Speichern als .npy
    np.save(outdir / "x.npy", x)

    # Laden
    loaded = np.load(outdir / "x.npy")
    print(loaded)

    return f"hat funktioniert: {x}"
