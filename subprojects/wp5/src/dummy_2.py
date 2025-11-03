from pathlib import Path

import numpy as np

from subprojects.wp5.src.dummy_3 import llinsi


def test(x: int) -> str:
    # kdiehl
    """Beispiel-Funktion: gibt eine stringbasierte Bestätigung zurück.

    Args:
        x: eine Ganzzahl (wird hier nur zu Demonstrationszwecken angezeigt)

    Returns:
        Eine Bestätigungszeichenkette mit dem übergebenen Wert.
    """
    print(llinsi())
    outdir = Path(
        "/home/kdiehl/Dokumente/Studium__Stipendium__Lebenslauf/ETH/AISCO/AISCO_code/AISCO/subprojects/wp5/data/"
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
