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

    outdir = Path(
        "/cluster/home/eschaerer/AISCO_code/subprojects/wp5/data"
    )  # setze hier deinen Zielpfad
    outdir.mkdir(
        parents=True, exist_ok=True
    )  # Verzeichnis anlegen, falls noch nicht vorhanden

    # Speichern als .npy
    np.save(outdir / "x.npy", x)

    # Laden
    loaded = np.load(outdir / "x.npy")
    print(loaded)

    return f"hat funktioniert: {x}"
