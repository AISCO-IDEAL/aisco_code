"""Tests for the centralized preprocessing settings in the orchestrator."""

from __future__ import annotations

import os

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.dataset import preprocess_all_datasets as runner


def test_preprocessing_config_contains_all_datasets() -> None:
    """One editable section in orchestrator should define all dataset paths."""

    datasets = runner.PREPROCESSING_CONFIG["datasets"]
    assert set(datasets) == {"euler", "drivaer", "ahmed", "blended_wing"}
    assert isinstance(runner.PREPROCESSING_CONFIG["defaults"]["output_path"], str)


def test_run_datasets_uses_configured_dataset_paths(monkeypatch) -> None:
    """run_datasets should pass configured input/output paths to dataset preprocessors."""

    calls: dict[str, dict] = {}

    def _fake_euler(**kwargs):
        calls["euler"] = kwargs
        return ["e1"]

    def _fake_vtp(dataset_name, **kwargs):
        calls[dataset_name] = kwargs
        return [f"{dataset_name}_1"]

    def _fake_blended(**kwargs):
        calls["blended_wing"] = kwargs
        return ["case_0001"]

    monkeypatch.setattr(runner, "process_euler_runs", _fake_euler)
    monkeypatch.setattr(runner, "process_vtp_runs", _fake_vtp)
    monkeypatch.setattr(runner, "process_blended_wing", _fake_blended)

    summary = runner.run_datasets(
        datasets=["euler", "drivaer", "ahmed", "blended_wing"],
        output_root=None,
        euler_workers=1,
        drivaer_workers=1,
        ahmed_workers=1,
        blended_wing_workers=1,
        target_reduction=0.2,
    )

    config = runner.PREPROCESSING_CONFIG
    output_path = config["defaults"]["output_path"]
    datasets = config["datasets"]

    assert calls["euler"]["results_dir"] == datasets["euler"]["results_dir"]
    assert calls["euler"]["force_dir"] == datasets["euler"]["force_dir"]
    assert calls["euler"]["output_dir"] == os.path.join(output_path, "euler")

    assert calls["drivaer"]["input_dir"] == datasets["drivaer"]["input_dir"]
    assert calls["drivaer"]["output_dir"] == os.path.join(output_path, "drivaer")
    assert calls["drivaer"]["target_reduction"] == 0.2

    assert calls["ahmed"]["input_dir"] == datasets["ahmed"]["input_dir"]
    assert calls["ahmed"]["output_dir"] == os.path.join(output_path, "ahmed")
    assert calls["ahmed"]["target_reduction"] == 0.2

    assert calls["blended_wing"]["vtk_dir"] == datasets["blended_wing"]["vtk_dir"]
    assert calls["blended_wing"]["case_data_path"] == datasets["blended_wing"]["case_data_path"]
    assert calls["blended_wing"]["output_dir"] == os.path.join(output_path, "blended_wing")

    assert summary == {
        "euler": 1,
        "drivaer": 1,
        "ahmed": 1,
        "blended_wing": 1,
    }
