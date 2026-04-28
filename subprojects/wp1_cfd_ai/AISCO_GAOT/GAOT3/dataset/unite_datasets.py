import json
import re
from pathlib import Path

import torch

# RUN COMMAND: python /cluster/home/USER/AISCO_code/subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/unite_datasets.py

# CHANGE USER TO YOUR USERNAME


# ================= CONFIGURATION =================
# Root-Ordner mit allen preprocessierten Datensaetzen
# (enthaelt Unterordner wie euler/, drivaer/, ahmed/, blended_wing/, ...)
PREPROCESSED_ROOT = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/all_processed")

# Zielordner
OUTPUT_PATH = Path("/cluster/work/fuge/AISCO/wp1_cfd_ai/united_all")

# Name der Index-Datei
INDEX_FILENAME = "order_.txt"

# Name der Datensatz-Index-Datei innerhalb jedes Datensatz-Ordners
DATASET_INDEX_FILENAME = "order_.txt"

# Persistentes Mapping fuer stabile Source->Unified Zuordnung.
NAME_MAPPING_FILENAME = "name_mapping.json"

# Feste Prefix-Zuordnung fuer bestimmte Datensaetze.
# Beide Blended-Wing-Datensaetze teilen sich denselben Prefix.
DATASET_PREFIX_OVERRIDES = {
    "blended_wing": 1,
    "blended_wing2": 1,
}
# =================================================


BASE_NAME_REGEX = re.compile(r"^sample_(\d+)$")


def _discover_datasets(root: Path) -> list[Path]:
    """Findet alle Datensatz-Unterordner, die eine order_.txt enthalten."""
    if not root.exists():
        return []

    dataset_paths = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if (candidate / DATASET_INDEX_FILENAME).exists():
            dataset_paths.append(candidate)
    return dataset_paths


def _build_dataset_specs(dataset_paths: list[Path]) -> list[tuple[Path, int, str]]:
    """Erzeugt Datensatz-Spezifikation (Pfad, Prefix, Label) mit Prefix-Overrides."""
    used_prefixes = set(DATASET_PREFIX_OVERRIDES.values())
    next_auto_prefix = 1
    specs: list[tuple[Path, int, str]] = []

    for path in dataset_paths:
        label = path.name
        if label in DATASET_PREFIX_OVERRIDES:
            prefix = DATASET_PREFIX_OVERRIDES[label]
        else:
            while next_auto_prefix in used_prefixes:
                next_auto_prefix += 1
            prefix = next_auto_prefix
            used_prefixes.add(prefix)

        specs.append((path, prefix, label))

    return specs


def _read_dataset_order(source_path: Path) -> list[str]:
    """Liest die Sample-Namen aus order_.txt eines Datensatzes."""
    index_path = source_path / DATASET_INDEX_FILENAME
    with open(index_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _resolve_sample_file(source_path: Path, sample_name: str) -> Path:
    """Resolves a sample file from modern or legacy layouts.

    Modern layout: <dataset>/<sample_name>/data.pt
    Legacy layout: <dataset>/<sample_name>.pt
    """
    subfolder_path = source_path / sample_name / "data.pt"
    if subfolder_path.exists():
        return subfolder_path

    flat_path = source_path / f"{sample_name}.pt"
    if flat_path.exists():
        return flat_path

    raise FileNotFoundError(
        f"Sample-Datei nicht gefunden fuer '{sample_name}' in {source_path}. "
        f"Erwartet: {subfolder_path} oder {flat_path}"
    )


def _collect_united_sample_names(output_path: Path) -> list[str]:
    """Liest alle vereinheitlichten Sample-Namen aus dem Zielordner."""
    sample_names = {
        pt_file.stem
        for pt_file in output_path.glob("sample*.pt")
        if pt_file.is_file()
    }
    return sorted(sample_names)


def _source_sample_key(dataset_label: str, sample_name: str) -> str:
    """Baut einen stabilen Schluessel pro Source-Sample."""
    return f"{dataset_label}::{sample_name}"


def _load_name_mapping(mapping_path: Path) -> dict[str, str]:
    """Liest das persistente Source->Unified Mapping."""
    if not mapping_path.exists():
        return {}

    try:
        with open(mapping_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠️ Warnung: Konnte {mapping_path} nicht lesen ({e}). Starte ohne Mapping.")
        return {}

    assignments = payload.get("assignments") if isinstance(payload, dict) else None
    if not isinstance(assignments, dict):
        print(f"⚠️ Warnung: Ungueltiges Mapping-Format in {mapping_path}. Starte ohne Mapping.")
        return {}

    clean_mapping: dict[str, str] = {
        key: value
        for key, value in assignments.items()
        if isinstance(key, str) and isinstance(value, str) and value.startswith("sample_")
    }
    return clean_mapping


def _save_name_mapping(mapping_path: Path, mapping: dict[str, str]) -> None:
    """Speichert das persistente Source->Unified Mapping."""
    payload = {
        "version": 1,
        "assignments": dict(sorted(mapping.items())),
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _parse_base_name_for_prefixes(base_name: str, prefixes: set[int]) -> tuple[int, int] | None:
    """Parst sample_<prefix><index> anhand bekannter Prefixe (index ist 5-stellig)."""
    match = BASE_NAME_REGEX.fullmatch(base_name)
    if match is None:
        return None

    suffix = match.group(1)
    for prefix in sorted(prefixes, key=lambda value: len(str(value)), reverse=True):
        prefix_str = str(prefix)
        if not suffix.startswith(prefix_str):
            continue

        index_str = suffix[len(prefix_str) :]
        if len(index_str) != 5 or not index_str.isdigit():
            continue

        return prefix, int(index_str)

    return None


def _initialize_prefix_counters(prefixes: set[int], used_base_names: set[str]) -> dict[int, int]:
    """Initialisiert Prefix-Zaehler aus bereits existierenden/zugewiesenen Unified-Namen."""
    counters = dict.fromkeys(prefixes, 0)
    for base_name in used_base_names:
        parsed = _parse_base_name_for_prefixes(base_name, prefixes)
        if parsed is None:
            continue
        prefix, index = parsed
        counters[prefix] = max(counters[prefix], index)
    return counters


def _next_available_base_name(prefix: int, start_index: int, used_base_names: set[str]) -> tuple[str, int]:
    """Ermittelt den naechsten freien Unified-Namen fuer ein Prefix."""
    current_index = start_index
    while True:
        current_index += 1
        base_name = f"sample_{prefix}{current_index:05d}"
        if base_name not in used_base_names:
            return base_name, current_index


def _process_dataset_samples(
    source_path: Path,
    prefix: int,
    label: str,
    prefix_counters: dict[int, int],
    used_base_names: set[str],
    name_mapping: dict[str, str],
) -> tuple[int, int]:
    """Verarbeitet alle Samples eines Datensatzes und liefert (neu_geschrieben, uebersprungen)."""
    try:
        sample_names = _read_dataset_order(source_path)
    except FileNotFoundError:
        print(f"⚠️ Warnung: Keine {DATASET_INDEX_FILENAME} in {source_path}, ueberspringe {label}.")
        return 0, 0

    count_written = 0
    count_skipped = 0
    for sample_name in sample_names:
        source_key = _source_sample_key(label, sample_name)
        mapped_base_name = name_mapping.get(source_key)

        assigned_index: int | None = None
        if mapped_base_name is None:
            base_name, assigned_index = _next_available_base_name(
                prefix,
                prefix_counters.get(prefix, 0),
                used_base_names,
            )
        else:
            base_name = mapped_base_name

        target_file = OUTPUT_PATH / f"{base_name}.pt"
        if target_file.exists():
            count_skipped += 1
            continue

        try:
            file_path = _resolve_sample_file(source_path, sample_name)
        except FileNotFoundError as e:
            print(f"⚠️ Warnung: {e}")
            continue

        try:
            # 1. Datei in den Arbeitsspeicher laden
            data = torch.load(file_path, map_location="cpu", weights_only=False)

            # 2. Den internen Dateinamen überschreiben
            # Wir prüfen zur Sicherheit, ob es ein Dictionary oder ein Objekt (z.B. PyG Data) ist
            if isinstance(data, dict):
                data["filename"] = base_name
            else:
                data.filename = base_name

            # 3. Datei am neuen Ort speichern
            torch.save(data, target_file)
            count_written += 1

            # Neue Zuordnungen erst nach erfolgreichem Schreiben persistieren.
            if assigned_index is not None:
                name_mapping[source_key] = base_name
                used_base_names.add(base_name)
                prefix_counters[prefix] = assigned_index

        except Exception as e:
            print(f"❌ Fehler bei {file_path}: {e}")

    return count_written, count_skipped


def consolidate_datasets():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    dataset_paths = _discover_datasets(PREPROCESSED_ROOT)
    datasets = _build_dataset_specs(dataset_paths)

    mapping_path = OUTPUT_PATH / NAME_MAPPING_FILENAME
    name_mapping = _load_name_mapping(mapping_path)

    existing_base_names = set(_collect_united_sample_names(OUTPUT_PATH))
    used_base_names = existing_base_names | set(name_mapping.values())
    prefixes = {prefix for _, prefix, _ in datasets}

    # Zaehlt fortlaufend je Prefix, damit gemeinsame Prefix-Gruppen
    # (z.B. blended_wing + blended_wing2) keine Dateinamen ueberschreiben.
    prefix_counters = _initialize_prefix_counters(prefixes, used_base_names)

    print(f"Starte Konsolidierung nach: {OUTPUT_PATH}\n" + "-" * 50)

    if not datasets:
        print(f"⚠️ Keine Datensaetze mit {DATASET_INDEX_FILENAME} gefunden in: {PREPROCESSED_ROOT}")
        return

    for source_path, prefix, label in datasets:
        if not source_path.exists():
            print(f"⚠️ Warnung: Pfad für Datenset {label} nicht gefunden: {source_path}")
            continue

        print(f"Verarbeite Datenset {label}...")
        count_written, count_skipped = _process_dataset_samples(
            source_path=source_path,
            prefix=prefix,
            label=label,
            prefix_counters=prefix_counters,
            used_base_names=used_base_names,
            name_mapping=name_mapping,
        )

        print(
            f"✅ {count_written} Dateien aus Datenset {label} neu geschrieben, "
            f"{count_skipped} uebersprungen."
        )

    _save_name_mapping(mapping_path, name_mapping)
    print(f"Persistentes Mapping aktualisiert: {mapping_path}")

    # --- Generierung der order_.txt aus dem realen Zielordnerinhalt ---
    all_sample_names = _collect_united_sample_names(OUTPUT_PATH)
    if all_sample_names:
        index_file_path = OUTPUT_PATH / INDEX_FILENAME

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
