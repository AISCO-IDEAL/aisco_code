import shutil
from pathlib import Path

# ================= CONFIGURATION =================
# Ersetze diese Pfade durch die tatsächlichen Pfade auf dem Euler-Cluster
PATH_A = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_blended_wing_cp_full_filtered")    # Blended Wing Dataset
PATH_B = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_cp_downsampled")           # Drivaer Dataset
PATH_C = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full")                    # Euler Dataset

# Zielordner
OUTPUT_PATH = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/united_dataset")

# Name der Index-Datei
INDEX_FILENAME = "order_.txt"
# =================================================

def consolidate_datasets():
    # Zielordner erstellen
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    datasets = [
        (PATH_A, 1, "A"),
        (PATH_B, 2, "B"),
        (PATH_C, 3, "C")
    ]

    # Liste, um alle neuen Dateinamen zu speichern
    all_sample_names = []

    print(f"Starte Konsolidierung nach: {OUTPUT_PATH}\n" + "-"*50)

    for source_path, prefix, label in datasets:
        if not source_path.exists():
            print(f"⚠️ Warnung: Pfad für Datenset {label} nicht gefunden: {source_path}")
            continue

        print(f"Verarbeite Datenset {label}...")

        # Such-Logik
        if label == "A":
            files = sorted(source_path.glob("case_*.pt"))
        else:
            files = sorted(source_path.rglob("data.pt"))

        # Kopieren, Umbenennen und Protokollieren
        count = 0
        for index, file_path in enumerate(files, start=1):
            # Auf 4 Stellen erhöht für mehr Kapazität (z.B. sample_10001)
            base_name = f"sample_{prefix}{index:04d}"
            new_filename = f"{base_name}.pt"

            target_file = OUTPUT_PATH / new_filename

            try:
                shutil.copy2(file_path, target_file)

                # Nur den base_name (ohne Endung) in die Liste eintragen
                all_sample_names.append(base_name)
                count += 1

            except Exception as e:
                print(f"❌ Fehler bei {file_path}: {e}")

        print(f"✅ {count} Dateien aus Datenset {label} kopiert.")

    # --- Generierung der order_.txt ---
    if all_sample_names:
        index_file_path = OUTPUT_PATH / INDEX_FILENAME
        # Wir sortieren die Liste am Ende nochmal streng aufsteigend
        all_sample_names.sort()

        with open(index_file_path, "w", encoding="utf-8") as f:
            for name in all_sample_names:
                f.write(f"{name}\n")

        print("-"*50)
        print(f"Index-Datei erstellt: {index_file_path}")
        print(f"Anzahl Einträge: {len(all_sample_names)}")
    else:
        print("⚠️ Keine Dateien gefunden. Index-Datei wurde nicht erstellt.")

    print(f"\nGesamtvorgang abgeschlossen.")

if __name__ == "__main__":
    consolidate_datasets()
