"""Tests for unified preprocess orchestrator summary behavior."""

from __future__ import annotations

from pathlib import Path

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset import preprocess_all_datasets as orchestrator


def test_run_datasets_returns_per_dataset_counts(monkeypatch, tmp_path: Path) -> None:
    """Runner should return per-dataset processed counts for overall summary."""

    monkeypatch.setattr(orchestrator, "process_euler_runs", lambda **_kwargs: ["e1", "e2"])
    monkeypatch.setattr(orchestrator, "process_vtp_runs", lambda dataset_name, **_kwargs: [f"{dataset_name}_1"])
    monkeypatch.setattr(orchestrator, "process_blended_wing", lambda **_kwargs: ["case_0001", "case_0002", "case_0003"])

    summary = orchestrator.run_datasets(
        datasets=["euler", "drivaer", "ahmed", "blended_wing"],
        output_root=str(tmp_path),
        euler_workers=1,
        drivaer_workers=1,
        ahmed_workers=1,
        blended_wing_workers=1,
        target_reduction=0.0,
    )

    assert summary == {
        "euler": 2,
        "drivaer": 1,
        "ahmed": 1,
        "blended_wing": 3,
    }
