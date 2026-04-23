from pathlib import Path

import torch

# RUN COMMAND: python /cluster/home/USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/unite_datasets.py

# ================= CONFIGURATION =================
# Pfade auf dem Euler-Cluster
PATH_A = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_blended_wing_cp_full_filtered")
PATH_B = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_drivaer_cp_downsampled")
PATH_C = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/processed_euler_cp_full")

# Zielordner
OUTPUT_PATH = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/united_dataset")

# Name der Index-Datei
INDEX_FILENAME = "order_.txt"
# =================================================


def consolidate_datasets():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    datasets = [(PATH_A, 1, "A"), (PATH_B, 2, "B"), (PATH_C, 3, "C")]

    all_sample_names = []

    print(f"Starte Konsolidierung nach: {OUTPUT_PATH}\n" + "-" * 50)

    for source_path, prefix, label in datasets:
        if not source_path.exists():
            print(f"⚠️ Warnung: Pfad für Datenset {label} nicht gefunden: {source_path}")
            continue

        print(f"Verarbeite Datenset {label}...")

        files = sorted(source_path.glob("case_*.pt")) if label == "A" else sorted(source_path.rglob("data.pt"))

        count = 0
        for index, file_path in enumerate(files, start=1):
            base_name = f"sample_{prefix}{index:05d}"
            new_filename = f"{base_name}.pt"

            target_file = OUTPUT_PATH / new_filename

            try:
                # 1. Datei in den Arbeitsspeicher laden
                data = torch.load(file_path)

                # 2. Den internen Dateinamen überschreiben
                # Wir prüfen zur Sicherheit, ob es ein Dictionary oder ein Objekt (z.B. PyG Data) ist
                if isinstance(data, dict):
                    data["filename"] = base_name
                else:
                    data.filename = base_name

                # 3. Datei am neuen Ort speichern
                torch.save(data, target_file)

                # 4. Namen für die order_.txt protokollieren
                all_sample_names.append(base_name)
                count += 1

            except Exception as e:
                print(f"❌ Fehler bei {file_path}: {e}")

        print(f"✅ {count} Dateien aus Datenset {label} verarbeitet und umgeschrieben.")

    # --- Generierung der order_.txt ---
    if all_sample_names:
        index_file_path = OUTPUT_PATH / INDEX_FILENAME
        all_sample_names.sort()

        with open(index_file_path, "w", encoding="utf-8") as f:
            f.writelines(f"{name}\n" for name in all_sample_names)

        print("-" * 50)
        print(f"Index-Datei erstellt: {index_file_path}")
        print(f"Anzahl Einträge: {len(all_sample_names)}")
    else:
        print("⚠️ Keine Dateien gefunden. Index-Datei wurde nicht erstellt.")

    print("\nGesamtvorgang abgeschlossen.")


if __name__ == "__main__":
    consolidate_datasets()
