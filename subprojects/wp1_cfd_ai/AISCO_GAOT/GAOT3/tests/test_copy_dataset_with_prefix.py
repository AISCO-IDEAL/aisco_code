from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

GAOT3_ROOT = Path(__file__).resolve().parents[1]
if str(GAOT3_ROOT) not in sys.path:
    sys.path.insert(0, str(GAOT3_ROOT))

from dataset.copy_dataset_with_prefix import copy_dataset_with_prefix


def test_copy_dataset_with_prefix_updates_internal_filename_and_preserves_provenance(tmp_path: Path):
    source_folder = tmp_path / "source"
    target_folder = tmp_path / "target"
    source_folder.mkdir()
    (source_folder / "order_.txt").write_text("sample_100001\n", encoding="utf-8")

    data = Data()
    data.filename = "sample_100001"
    data.source_dataset = "original_dataset"
    data.source_case_id = "case_001"
    torch.save(data, source_folder / "sample_100001.pt")

    copy_dataset_with_prefix(source_folder, target_folder, "7")

    copied = torch.load(target_folder / "sample_700001.pt", weights_only=False)
    assert copied.filename == "sample_700001"
    assert copied.source_dataset == "original_dataset"
    assert copied.source_case_id == "case_001"
    assert (target_folder / "order_.txt").read_text(encoding="utf-8") == "sample_700001\n"
