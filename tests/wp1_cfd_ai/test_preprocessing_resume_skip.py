"""Tests for incremental resume behavior in preprocessing scripts."""

from __future__ import annotations

from pathlib import Path

import torch

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset import preprocessing_ahmed_drivaer as vtp_module
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset import preprocessing_euler as euler_module


def test_euler_process_all_runs_skips_existing_data_pt(monkeypatch, tmp_path: Path) -> None:
    """Euler reruns should skip runs that already have output data.pt."""

    results_dir = tmp_path / "results"
    output_dir = tmp_path / "output"
    results_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    (results_dir / "RUN_001_results.cas.h5").write_text("dummy")
    (results_dir / "RUN_002_results.cas.h5").write_text("dummy")

    existing_dir = output_dir / "euler_run_1"
    existing_dir.mkdir(parents=True)
    torch.save({"ok": True}, existing_dir / "data.pt")

    processed_ids: list[int] = []

    def _fake_process_single_run(run_id: int, _results_dir: str, _force_dir: str, out_dir: str) -> str:
        processed_ids.append(run_id)
        save_name = f"euler_run_{run_id}"
        save_dir = Path(out_dir) / save_name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"ok": True}, save_dir / "data.pt")
        return save_name

    monkeypatch.setattr(euler_module, "process_single_run", _fake_process_single_run)
    monkeypatch.setattr(euler_module, "write_dataset_manifest", lambda **_kwargs: None)

    out = euler_module.process_all_runs(
        results_dir=str(results_dir),
        force_dir=str(tmp_path / "force"),
        output_dir=str(output_dir),
        run_ids=None,
        max_runs=None,
        num_workers=1,
    )

    assert processed_ids == [2]
    assert out == ["euler_run_1", "euler_run_2"]
    assert (output_dir / "order_.txt").read_text().splitlines() == ["euler_run_1", "euler_run_2"]


def test_vtp_process_all_skips_existing_data_pt(monkeypatch, tmp_path: Path) -> None:
    """DrivAer/Ahmed reruns should skip runs that already have output data.pt."""

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)

    existing_dir = output_dir / "drivaer_run_1"
    existing_dir.mkdir(parents=True)
    torch.save({"ok": True}, existing_dir / "data.pt")

    processed_ids: list[int] = []

    def _fake_process_single_vtp(run_id: int, dataset_cfg: dict, target_reduction: float = 0.0) -> str:
        del target_reduction
        processed_ids.append(run_id)
        save_name = f"{dataset_cfg['prefix']}_{run_id}"
        save_dir = Path(dataset_cfg["output_dir"]) / save_name
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"ok": True}, save_dir / "data.pt")
        return save_name

    monkeypatch.setattr(vtp_module, "_detect_run_ids", lambda _input_dir: [1, 2])
    monkeypatch.setattr(vtp_module, "process_single_vtp", _fake_process_single_vtp)
    monkeypatch.setattr(vtp_module, "write_dataset_manifest", lambda **_kwargs: None)

    out = vtp_module.process_all(
        dataset_name="drivaer",
        output_dir=str(output_dir),
        input_dir=str(tmp_path / "input"),
        num_workers=1,
        target_reduction=0.0,
    )

    assert processed_ids == [2]
    assert out == ["drivaer_run_1", "drivaer_run_2"]
    assert (output_dir / "order_.txt").read_text().splitlines() == ["drivaer_run_1", "drivaer_run_2"]
