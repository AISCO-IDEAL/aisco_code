from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class PatchSpec:
    file: Path
    updates: dict[str, Any]


@dataclass
class RunSpec:
    name: str
    main_file: Path
    working_directory: Path
    timeout_sec: int
    args: list[str]
    base_env: dict[str, str]
    base_patches: list[PatchSpec]


@dataclass
class CaseSpec:
    name: str
    description: str
    patches: list[PatchSpec]


@dataclass
class GpuSpec:
    name: str
    cuda_visible_devices: str
    gpu_query_index: int
    label: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Scenario root must be a dict, got {type(data).__name__}")
    return data


def _parse_patch_specs(items: list[dict[str, Any]], repo_root: Path) -> list[PatchSpec]:
    parsed: list[PatchSpec] = []
    for item in items:
        file_rel = item["file"]
        updates = item["updates"]
        if not isinstance(updates, dict):
            raise TypeError(f"updates for {file_rel} must be a dict")
        parsed.append(PatchSpec(file=repo_root / file_rel, updates=updates))
    return parsed


def _set_nested_value(payload: Any, dotted_path: str, value: Any) -> None:
    keys = dotted_path.split(".")
    cur = payload
    for key in keys[:-1]:
        if isinstance(cur, list):
            idx = int(key)
            cur = cur[idx]
        else:
            if key not in cur:
                cur[key] = {}
            cur = cur[key]
    last = keys[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def _apply_yaml_updates(file_path: Path, updates: dict[str, Any]) -> None:
    with file_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    for dotted_path, value in updates.items():
        _set_nested_value(payload, dotted_path, value)
    with file_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _query_gpu_total(gpu_query_index: int) -> tuple[float | None, float | None, str | None]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return None, None, None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx = int(parts[0])
        if idx == gpu_query_index:
            return float(parts[2]), float(parts[3]), parts[1]
    return None, None, None


def _query_gpu_process_mem(pid: int) -> float | None:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return None
    values: list[float] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            row_mem = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            values.append(row_mem)
    if not values:
        return None
    return sum(values)


def _monitor_gpu_memory(
    pid: int,
    gpu_query_index: int,
    interval_sec: float,
    stop_event: threading.Event,
    sink: list[dict[str, Any]],
) -> None:
    while not stop_event.is_set():
        ts = time.time()
        proc_mem = _query_gpu_process_mem(pid)
        gpu_used, gpu_total, gpu_name = _query_gpu_total(gpu_query_index)
        sink.append(
            {
                "timestamp": ts,
                "proc_used_mb": proc_mem,
                "gpu_used_mb": gpu_used,
                "gpu_total_mb": gpu_total,
                "gpu_name": gpu_name,
            }
        )
        stop_event.wait(interval_sec)


def _run_one_case(
    run: RunSpec,
    case: CaseSpec,
    gpu: GpuSpec,
    python_executable: str,
    sample_interval_sec: float,
    output_dir: Path,
    force_cuda_visible_devices: str | None,
    force_gpu_query_index: int | None,
) -> dict[str, Any]:
    backup: dict[Path, str] = {}
    all_patches = run.base_patches + case.patches
    for patch in all_patches:
        if patch.file not in backup:
            backup[patch.file] = patch.file.read_text(encoding="utf-8")
    try:
        for patch in all_patches:
            _apply_yaml_updates(patch.file, patch.updates)
        env = os.environ.copy()
        env.update(run.base_env)
        effective_cuda_visible_devices = force_cuda_visible_devices or gpu.cuda_visible_devices
        effective_gpu_query_index = force_gpu_query_index if force_gpu_query_index is not None else gpu.gpu_query_index
        env["CUDA_VISIBLE_DEVICES"] = effective_cuda_visible_devices
        env["PYTHONPATH"] = str(run.working_directory)
        log_file = output_dir / f"{run.name}__{case.name}__{gpu.name}.log"
        cmd = [python_executable, str(run.main_file), *run.args]
        start = time.time()
        with log_file.open("w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(run.working_directory),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            samples: list[dict[str, Any]] = []
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=_monitor_gpu_memory,
                args=(proc.pid, effective_gpu_query_index, sample_interval_sec, stop_event, samples),
                daemon=True,
            )
            monitor_thread.start()
            timed_out = False
            try:
                return_code = proc.wait(timeout=run.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                return_code = -9
            stop_event.set()
            monitor_thread.join(timeout=5)
        elapsed = time.time() - start
        peak_proc = max((s["proc_used_mb"] for s in samples if s["proc_used_mb"] is not None), default=None)
        peak_gpu = max((s["gpu_used_mb"] for s in samples if s["gpu_used_mb"] is not None), default=None)
        gpu_total = None
        gpu_name = gpu.label
        if samples:
            latest = samples[-1]
            gpu_total = latest.get("gpu_total_mb")
            gpu_name = latest.get("gpu_name") or gpu.label
        sample_file = output_dir / f"{run.name}__{case.name}__{gpu.name}.samples.json"
        sample_file.write_text(json.dumps(samples, indent=2), encoding="utf-8")
        return {
            "run_name": run.name,
            "case_name": case.name,
            "gpu_name": gpu.name,
            "gpu_label": gpu_name,
            "cuda_visible_devices": effective_cuda_visible_devices,
            "gpu_query_index": effective_gpu_query_index,
            "main_file": str(run.main_file),
            "return_code": return_code,
            "timed_out": timed_out,
            "success": return_code == 0 and not timed_out,
            "elapsed_sec": elapsed,
            "peak_process_mem_mb": peak_proc,
            "peak_gpu_used_mb": peak_gpu,
            "gpu_total_mb": gpu_total,
            "log_file": str(log_file),
            "samples_file": str(sample_file),
        }
    finally:
        for file_path, text in backup.items():
            file_path.write_text(text, encoding="utf-8")


def _parse_run_specs(root: dict[str, Any], repo_root: Path) -> list[RunSpec]:
    return [
        RunSpec(
            name=item["name"],
            main_file=repo_root / item["main_file"],
            working_directory=repo_root / item.get("working_directory", "."),
            timeout_sec=int(item.get("timeout_sec", 3600)),
            args=[str(x) for x in item.get("args", [])],
            base_env={str(k): str(v) for k, v in item.get("base_env", {}).items()},
            base_patches=_parse_patch_specs(item.get("base_patches", []), repo_root),
        )
        for item in root.get("runs", [])
    ]


def _parse_case_specs(root: dict[str, Any], repo_root: Path) -> list[CaseSpec]:
    return [
        CaseSpec(
            name=item["name"],
            description=item.get("description", ""),
            patches=_parse_patch_specs(item.get("patches", []), repo_root),
        )
        for item in root.get("cases", [])
    ]


def _parse_gpu_specs(root: dict[str, Any]) -> list[GpuSpec]:
    gpus: list[GpuSpec] = []
    profiles = root.get("gpu_profiles", {})
    for name, cfg in profiles.items():
        gpus.append(
            GpuSpec(
                name=name,
                cuda_visible_devices=str(cfg["cuda_visible_devices"]),
                gpu_query_index=int(cfg.get("gpu_query_index", 0)),
                label=str(cfg.get("label", name)),
            )
        )
    return gpus


def _write_results(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fieldnames = [
        "run_name",
        "case_name",
        "gpu_name",
        "gpu_label",
        "cuda_visible_devices",
        "gpu_query_index",
        "main_file",
        "return_code",
        "timed_out",
        "success",
        "elapsed_sec",
        "peak_process_mem_mb",
        "peak_gpu_used_mb",
        "gpu_total_mb",
        "log_file",
        "samples_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


TItem = TypeVar("TItem")


def _filter_by_name(
    items: list[TItem],
    include_names: set[str] | None,
    key_fn: Callable[[TItem], str],
) -> list[TItem]:
    if include_names is None:
        return items
    return [item for item in items if key_fn(item) in include_names]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark mesh extraction memory usage across main files and GPU profiles.")
    parser.add_argument("--scenario", required=True, help="Path to scenario yaml file")
    parser.add_argument("--python", dest="python_executable", default=sys.executable, help="Python executable to run mains")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="GPU sample interval in seconds")
    parser.add_argument("--only-runs", nargs="*", default=None, help="Optional run names filter")
    parser.add_argument("--only-cases", nargs="*", default=None, help="Optional case names filter")
    parser.add_argument("--only-gpus", nargs="*", default=None, help="Optional gpu profile names filter")
    parser.add_argument(
        "--repo-root-override",
        default=None,
        help="Optional override for repo root used to resolve run/config paths (useful for worktree launches).",
    )
    parser.add_argument(
        "--force-cuda-visible-devices",
        default=None,
        help="Optional override for CUDA_VISIBLE_DEVICES applied to all selected profiles (useful for SLURM remapping).",
    )
    parser.add_argument(
        "--force-gpu-query-index",
        type=int,
        default=None,
        help="Optional override for nvidia-smi GPU query index used by the monitor (default uses profile value).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only validate scenario and print matrix")
    args = parser.parse_args()
    scenario_path = Path(args.scenario).resolve()
    scenario = _read_yaml(scenario_path)
    repo_root_value = args.repo_root_override or scenario.get("repo_root", ".")
    repo_root = Path(repo_root_value).resolve()
    output_base = (repo_root / scenario.get("output_dir", "tests/wp5/benchmark_results")).resolve()
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / run_stamp
    runs = _parse_run_specs(scenario, repo_root)
    cases = _parse_case_specs(scenario, repo_root)
    gpus = _parse_gpu_specs(scenario)
    runs = _filter_by_name(runs, set(args.only_runs) if args.only_runs else None, lambda x: x.name)
    cases = _filter_by_name(cases, set(args.only_cases) if args.only_cases else None, lambda x: x.name)
    gpus = _filter_by_name(gpus, set(args.only_gpus) if args.only_gpus else None, lambda x: x.name)
    if not runs or not cases or not gpus:
        raise ValueError("Filtered benchmark matrix is empty. Check scenario and filter arguments.")
    matrix = [(r, c, g) for r in runs for c in cases for g in gpus]
    print(f"Benchmark matrix size: {len(matrix)}")
    for r, c, g in matrix:
        print(f"- run={r.name} case={c.name} gpu={g.name}")
    if args.dry_run:
        return 0
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run, case, gpu in matrix:
        print(f"\n[START] run={run.name} case={case.name} gpu={gpu.name}")
        row = _run_one_case(
            run=run,
            case=case,
            gpu=gpu,
            python_executable=args.python_executable,
            sample_interval_sec=args.sample_interval,
            output_dir=output_dir,
            force_cuda_visible_devices=args.force_cuda_visible_devices,
            force_gpu_query_index=args.force_gpu_query_index,
        )
        rows.append(row)
        status = "OK" if row["success"] else "FAIL"
        print(
            f"[DONE] {status} elapsed={row['elapsed_sec']:.2f}s peak_proc={row['peak_process_mem_mb']}MB peak_gpu={row['peak_gpu_used_mb']}MB"
        )
    _write_results(rows, output_dir)
    print(f"\nResults written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
