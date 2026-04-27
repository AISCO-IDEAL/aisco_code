"""Contract/manifest utility tests for unified preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset.common_preprocessing import (
    PREPROCESS_CONTRACT_VERSION,
    apply_canonical_contract,
    validate_canonical_contract,
    write_dataset_manifest,
)


def test_apply_canonical_contract_sets_required_keys_and_syncs_x_cp() -> None:
    """Missing optional tensors should be created and x/cp should be identical."""

    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    data = Data(pos=pos)
    data.c = torch.ones((3, 3), dtype=torch.float32)
    data.face = torch.tensor([[0], [1], [2]], dtype=torch.long)
    data.cp = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32)

    out = apply_canonical_contract(
        data,
        source_dataset="blended_wing",
        source_case_id="case_0001",
        filename="case_0001",
        geom_name="geom_000",
        flow_conditions={"M_inf": 0.3},
    )

    assert torch.equal(out.x, out.cp)
    assert out.pressure.shape == (3, 1)
    assert torch.isnan(out.pressure).all()
    assert out.wall_shear.shape == (3, 3)
    assert torch.isnan(out.wall_shear).all()
    assert out.source_dataset == "blended_wing"
    assert out.source_case_id == "case_0001"
    assert out.preprocess_version == PREPROCESS_CONTRACT_VERSION


def test_validate_canonical_contract_raises_with_actionable_message() -> None:
    """Validator should raise clear errors listing missing keys."""

    data = Data()

    with pytest.raises(ValueError, match="missing required key"):
        validate_canonical_contract(data)


def test_write_dataset_manifest_writes_expected_json(tmp_path: Path) -> None:
    """Manifest writer should persist contract metadata for each dataset output."""

    write_dataset_manifest(
        output_dir=str(tmp_path),
        dataset_name="blended_wing",
        processed_count=10,
        total_candidates=12,
        failed_count=2,
        contract_version=PREPROCESS_CONTRACT_VERSION,
        source_root="/tmp/source",
    )

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["dataset"] == "blended_wing"
    assert manifest["processed_count"] == 10
    assert manifest["failed_count"] == 2
    assert manifest["contract_version"] == PREPROCESS_CONTRACT_VERSION
