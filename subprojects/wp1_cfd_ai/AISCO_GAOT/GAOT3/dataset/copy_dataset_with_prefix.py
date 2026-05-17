"""Copy a processed dataset folder into another folder with a new sample prefix.

Example:
    python subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT3/dataset/copy_dataset_with_prefix.py \
        /path/to/source_dataset \
        /path/to/target_dataset \
        7

The script reads ``order_.txt`` from the source folder, copies sample ``.pt``
files in that order, renames them to ``sample_<target_prefix><5-digit-id>.pt``,
and appends the new names to the target folder's ``order_.txt``. Source samples
may be stored either as ``sample_100001.pt`` or ``sample_100001/data.pt``; the
target layout is always flat.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import torch

ORDER_FILENAME = "order_.txt"
SAMPLE_NAME_RE = re.compile(r"^sample_(\d+)$")


@dataclass(frozen=True)
class CopyPlanItem:
    source_path: Path
    target_path: Path
    new_name: str


def _read_order_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Order file not found: {path}")
    with path.open(encoding="utf-8") as f:
        return [line.strip().removesuffix(".pt") for line in f if line.strip()]


def _write_order_file(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(f"{name}\n" for name in names)


def _resolve_sample_path(source_folder: Path, sample_name: str) -> Path:
    nested_path = source_folder / sample_name
    nested_data_path = nested_path / "data.pt"
    if nested_data_path.is_file():
        return nested_data_path

    flat_path = source_folder / f"{sample_name}.pt"
    if flat_path.is_file():
        return flat_path

    raise FileNotFoundError(f"Could not find sample '{sample_name}'. Expected {flat_path} or {nested_path / 'data.pt'}")


def _collect_existing_sample_names(target_folder: Path, target_order_names: list[str]) -> set[str]:
    names = set(target_order_names)
    names.update(path.stem for path in target_folder.glob("sample_*.pt") if path.is_file())
    names.update(path.name for path in target_folder.glob("sample_*") if path.is_dir())
    return names


def _next_sample_name(prefix: str, used_names: set[str], current_index: int) -> tuple[str, int]:
    while True:
        current_index += 1
        candidate = f"sample_{prefix}{current_index:05d}"
        if candidate not in used_names:
            return candidate, current_index


def _highest_existing_index(prefix: str, names: set[str]) -> int:
    highest = 0
    for name in names:
        match = SAMPLE_NAME_RE.fullmatch(name)
        if match is None:
            continue
        suffix = match.group(1)
        if not suffix.startswith(prefix):
            continue
        index_part = suffix[len(prefix) :]
        if len(index_part) == 5 and index_part.isdigit():
            highest = max(highest, int(index_part))
    return highest


def _build_copy_plan(source_folder: Path, target_folder: Path, prefix: str) -> tuple[list[CopyPlanItem], list[str]]:
    source_order_names = _read_order_file(source_folder / ORDER_FILENAME)
    target_order_path = target_folder / ORDER_FILENAME
    target_order_names = _read_order_file(target_order_path) if target_order_path.exists() else []

    used_names = _collect_existing_sample_names(target_folder, target_order_names)
    current_index = _highest_existing_index(prefix, used_names)

    plan = []
    new_names = []
    for sample_name in source_order_names:
        source_path = _resolve_sample_path(source_folder, sample_name)
        new_name, current_index = _next_sample_name(prefix, used_names, current_index)
        target_path = target_folder / f"{new_name}.pt"
        if target_path.exists():
            raise FileExistsError(f"Target path already exists: {target_path}")
        plan.append(CopyPlanItem(source_path=source_path, target_path=target_path, new_name=new_name))
        new_names.append(new_name)
        used_names.add(new_name)

    return plan, target_order_names + new_names


def copy_dataset_with_prefix(source_folder: Path, target_folder: Path, target_prefix: str, *, dry_run: bool = False) -> None:
    if not target_prefix.isdigit():
        raise ValueError(f"target_prefix must contain only digits, got: {target_prefix!r}")

    source_folder = source_folder.resolve()
    target_folder = target_folder.resolve()
    if source_folder == target_folder:
        raise ValueError("source_folder and target_folder must be different")

    plan, updated_order_names = _build_copy_plan(source_folder, target_folder, target_prefix)
    print(f"Source: {source_folder}")
    print(f"Target: {target_folder}")
    print(f"Target prefix: {target_prefix}")
    print(f"Samples to copy: {len(plan)}")

    for item in plan:
        print(f"{item.source_path.name} -> {item.new_name}")

    if dry_run:
        print("Dry run only; no files copied and order_.txt was not updated.")
        return

    target_folder.mkdir(parents=True, exist_ok=True)
    for item in plan:
        data = torch.load(item.source_path, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            data["filename"] = item.new_name
        else:
            data.filename = item.new_name
        torch.save(data, item.target_path)

    _write_order_file(target_folder / ORDER_FILENAME, updated_order_names)
    print(f"Updated order file: {target_folder / ORDER_FILENAME}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy samples from a processed dataset folder into another folder with a new sample prefix."
    )
    parser.add_argument("source_folder", type=Path, help="Folder containing source samples and order_.txt")
    parser.add_argument("target_folder", type=Path, help="Folder to receive renamed samples and updated order_.txt")
    parser.add_argument("target_prefix", help="Prefix number for new names, e.g. 7 -> sample_700001")
    parser.add_argument("--dry-run", action="store_true", help="Print the copy plan without writing files")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    copy_dataset_with_prefix(
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        target_prefix=str(args.target_prefix),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
