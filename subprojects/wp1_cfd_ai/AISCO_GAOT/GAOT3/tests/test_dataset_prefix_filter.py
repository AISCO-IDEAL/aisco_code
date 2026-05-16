from __future__ import annotations

import sys
from pathlib import Path

GAOT3_ROOT = Path(__file__).resolve().parents[1]
if str(GAOT3_ROOT) not in sys.path:
    sys.path.insert(0, str(GAOT3_ROOT))

from src.dataloader.pyg_datasets import VTKMeshDataset


def _write_order(root: Path, sample_names: list[str]) -> Path:
    order_file = root / "order_.txt"
    order_file.write_text("\n".join(sample_names) + "\n", encoding="utf-8")
    for sample_name in sample_names:
        (root / f"{sample_name}.pt").touch()
    return order_file


def _dataset(tmp_path: Path, sample_names: list[str], overrides: dict | None = None, split: str = "train"):
    order_file = _write_order(tmp_path, sample_names)
    config = {
        "processed_folder": "",
        "train_size": 10,
        "val_size": 0,
        "test_size": 0,
        "rand_dataset": False,
        "handpicked_val": False,
        "val_files_path": None,
    }
    if overrides:
        config.update(overrides)
    return VTKMeshDataset(str(tmp_path), str(order_file), config, split=split)


def test_missing_prefix_filter_keeps_all_samples(tmp_path: Path):
    sample_names = ["sample_100001", "sample_200001", "sample_300001"]

    dataset = _dataset(tmp_path, sample_names)

    assert dataset.split_filenames == [f"{name}.pt" for name in sample_names]


def test_null_prefix_filter_keeps_all_samples(tmp_path: Path):
    sample_names = ["sample_100001", "sample_200001", "sample_300001"]

    dataset = _dataset(tmp_path, sample_names, {"allowed_dataset_prefixes": None})

    assert dataset.split_filenames == [f"{name}.pt" for name in sample_names]


def test_single_prefix_filter_keeps_only_matching_samples(tmp_path: Path):
    sample_names = ["sample_100001", "sample_200001", "sample_100002"]

    dataset = _dataset(tmp_path, sample_names, {"allowed_dataset_prefixes": [1]})

    assert dataset.split_filenames == ["sample_100001.pt", "sample_100002.pt"]


def test_multi_prefix_filter_keeps_all_matching_samples(tmp_path: Path):
    sample_names = ["sample_100001", "sample_200001", "sample_300001", "sample_100002"]

    dataset = _dataset(tmp_path, sample_names, {"allowed_dataset_prefixes": [1, 3]})

    assert dataset.split_filenames == ["sample_100001.pt", "sample_300001.pt", "sample_100002.pt"]


def test_handpicked_validation_ignores_disallowed_prefixes(tmp_path: Path):
    sample_names = ["sample_100001", "sample_200001", "sample_100002"]
    val_file = tmp_path / "val_files.yaml"
    val_file.write_text("- sample_200001\n- sample_100002\n", encoding="utf-8")

    dataset = _dataset(
        tmp_path,
        sample_names,
        {
            "allowed_dataset_prefixes": [1],
            "handpicked_val": True,
            "val_files_path": str(val_file),
            "train_size": 1,
            "val_size": 1,
            "test_size": 0,
        },
        split="val",
    )

    assert dataset.split_filenames == ["sample_100002.pt"]
