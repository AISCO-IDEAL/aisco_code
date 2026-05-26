from pathlib import Path
from types import SimpleNamespace

from subprojects.wp7_gui.v3.modes.launcher import routes


def test_sd_candidates_include_wsl_drive_root_for_windows_volume_label(monkeypatch) -> None:
    """Windows-hosted WSL exposes labelled drives as /mnt/<letter>, not E:/."""
    monkeypatch.setenv("AISCO_BCKUP_DIR", "")
    monkeypatch.delenv("AISCO_SD_DIR", raising=False)
    monkeypatch.delenv("AISCO_BCKUP_DRIVE", raising=False)
    monkeypatch.setattr(routes, "_is_wsl", lambda: True)
    monkeypatch.setattr(routes, "_mount_roots_with_label", lambda label: [])

    def fake_run(cmd, capture_output, text, timeout, check):
        if cmd[0] == "powershell.exe":
            return SimpleNamespace(returncode=0, stdout="E\n", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(routes.subprocess, "run", fake_run)

    assert Path("/mnt/e") in routes._candidate_sd_roots()


def test_sd_drive_override_maps_to_wsl_mount(monkeypatch) -> None:
    monkeypatch.setenv("AISCO_BCKUP_DRIVE", "F:")
    monkeypatch.delenv("AISCO_BCKUP_DIR", raising=False)
    monkeypatch.delenv("AISCO_SD_DIR", raising=False)
    monkeypatch.setattr(routes, "_is_wsl", lambda: True)
    monkeypatch.setattr(routes, "_windows_volume_roots_by_label", lambda label: [])
    monkeypatch.setattr(routes, "_mount_roots_with_label", lambda label: [])

    assert routes._candidate_sd_roots()[0] == Path("/mnt/f")


def test_copy_printables_to_sd_writes_files(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "storage" / "run_001"
    out_dir = run_dir / "printables_3d"
    out_dir.mkdir(parents=True)
    stl = out_dir / "mesh.stl"
    three_mf = out_dir / "mesh.3mf"
    report = out_dir / "print_report.json"
    stl.write_text("solid mesh", encoding="utf-8")
    three_mf.write_text("3mf", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    sd_root = tmp_path / "AISCO_BCKUP"
    sd_root.mkdir()
    monkeypatch.setattr(routes, "_find_sd_backup_root", lambda: sd_root)

    result = routes._copy_printables_to_sd(
        run_dir=run_dir,
        stl_path=str(stl),
        three_mf_path=str(three_mf),
        report_path=report,
    )

    dest = sd_root / "AISCO_print_exports" / "run_001"
    assert result["ok"] is True
    assert (dest / "mesh.stl").read_text(encoding="utf-8") == "solid mesh"
    assert (dest / "mesh.3mf").read_text(encoding="utf-8") == "3mf"
    assert (dest / "print_report.json").read_text(encoding="utf-8") == "{}"


def test_copy_printables_to_sd_uses_powershell_fallback_when_wsl_drive_is_not_mounted(monkeypatch, tmp_path) -> None:
    run_dir = tmp_path / "run_001"
    out_dir = run_dir / "printables_3d"
    out_dir.mkdir(parents=True)
    stl = out_dir / "mesh.stl"
    report = out_dir / "print_report.json"
    stl.write_text("solid mesh", encoding="utf-8")
    report.write_text("{}", encoding="utf-8")
    commands = []

    monkeypatch.setattr(routes, "_is_wsl", lambda: True)
    monkeypatch.setattr(routes, "_find_sd_backup_root", lambda: None)
    monkeypatch.setattr(routes, "_windows_sd_drive_root", lambda: "D:\\")
    monkeypatch.setattr(routes, "_wsl_windows_path_candidates", lambda path: [f"\\\\wsl$\\Ubuntu-24.04\\{path.name}"])

    def fake_run(cmd, capture_output, text, timeout, check):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(routes.subprocess, "run", fake_run)

    result = routes._copy_printables_to_sd(
        run_dir=run_dir,
        stl_path=str(stl),
        three_mf_path=None,
        report_path=report,
    )

    assert result["ok"] is True
    assert result["method"] == "powershell"
    assert len(commands) == 2
    assert all(cmd[:3] == ["powershell.exe", "-NoProfile", "-Command"] for cmd in commands)
    assert "D:\\AISCO_print_exports\\run_001" in commands[0][3]


def test_print_mesh_exports_local_then_sd_then_wandb(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo" / "subprojects"
    export_script = repo_root / "wp2_assembler" / "version4" / "3D_printing_helpers" / "export_printables.py"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("# export helper", encoding="utf-8")
    run_dir = tmp_path / "run_001"
    out_dir = run_dir / "printables_3d"
    out_dir.mkdir(parents=True)
    stl = out_dir / "mesh.stl"
    three_mf = out_dir / "mesh.3mf"
    report = out_dir / "print_report.json"
    stl.write_text("solid mesh", encoding="utf-8")
    three_mf.write_text("3mf", encoding="utf-8")
    report.write_text(
        (
            '{"outputs": {'
            f'"rollout_unmodified_mesh_flat_bottom": "{stl}", '
            f'"rollout_unmodified_flat_bottom_3mf": "{three_mf}"'
            "}}"
        ),
        encoding="utf-8",
    )
    order = []

    monkeypatch.setattr(routes, "SETTINGS", SimpleNamespace(repo_root=repo_root, cache_dir=tmp_path / "cache"))
    monkeypatch.setitem(routes._runtime_state, "active_cfg_folder", "")
    monkeypatch.setattr(routes, "get_storage_root_for_session", lambda session_dir: tmp_path)
    monkeypatch.setattr(routes, "_find_latest_printable_run_dir", lambda storage_root, active_cfg_session: run_dir)
    def fake_run(*args, **kwargs):
        order.append("local")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_sd_copy(**kwargs):
        order.append("sd")
        return {"enabled": True, "ok": True}

    def fake_wandb_upload(**kwargs):
        order.append("wandb")
        return {"enabled": True, "ok": True}

    monkeypatch.setattr(routes.subprocess, "run", fake_run)
    monkeypatch.setattr(routes, "_copy_printables_to_sd", fake_sd_copy)
    monkeypatch.setattr(routes, "_upload_printables_to_wandb", fake_wandb_upload)

    response = routes.run_print_mesh()

    assert order == ["local", "sd", "wandb"]
    assert b'"sd_export"' in response.body
    assert b'"wandb_upload"' in response.body
    assert b'"status":"ok"' in response.body


def test_print_mesh_runs_wandb_even_when_sd_export_fails(monkeypatch, tmp_path) -> None:
    repo_root = tmp_path / "repo" / "subprojects"
    export_script = repo_root / "wp2_assembler" / "version4" / "3D_printing_helpers" / "export_printables.py"
    export_script.parent.mkdir(parents=True)
    export_script.write_text("# export helper", encoding="utf-8")
    run_dir = tmp_path / "run_001"
    out_dir = run_dir / "printables_3d"
    out_dir.mkdir(parents=True)
    stl = out_dir / "mesh.stl"
    three_mf = out_dir / "mesh.3mf"
    report = out_dir / "print_report.json"
    stl.write_text("solid mesh", encoding="utf-8")
    three_mf.write_text("3mf", encoding="utf-8")
    report.write_text(
        (
            '{"outputs": {'
            f'"rollout_unmodified_mesh_flat_bottom": "{stl}", '
            f'"rollout_unmodified_flat_bottom_3mf": "{three_mf}"'
            "}}"
        ),
        encoding="utf-8",
    )
    order = []

    monkeypatch.setattr(routes, "SETTINGS", SimpleNamespace(repo_root=repo_root, cache_dir=tmp_path / "cache"))
    monkeypatch.setitem(routes._runtime_state, "active_cfg_folder", "")
    monkeypatch.setattr(routes, "get_storage_root_for_session", lambda session_dir: tmp_path)
    monkeypatch.setattr(routes, "_find_latest_printable_run_dir", lambda storage_root, active_cfg_session: run_dir)
    monkeypatch.setattr(
        routes.subprocess,
        "run",
        lambda *args, **kwargs: order.append("local") or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    def fake_sd_copy(**kwargs):
        order.append("sd")
        return {"enabled": True, "ok": False, "message": "sd card not found"}

    def fake_wandb_upload(**kwargs):
        order.append("wandb")
        return {"enabled": True, "ok": True}

    monkeypatch.setattr(routes, "_copy_printables_to_sd", fake_sd_copy)
    monkeypatch.setattr(routes, "_upload_printables_to_wandb", fake_wandb_upload)

    response = routes.run_print_mesh()

    assert order == ["local", "sd", "wandb"]
    assert b'"status":"ok"' in response.body
    assert b"sd card not found" in response.body
