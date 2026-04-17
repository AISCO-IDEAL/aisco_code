#!/usr/bin/env python
"""Deterministic GAOT2-vs-GAOT3 equivalence check.

This is intentionally small and synthetic: it catches behavioral drift while
staying independent of cluster datasets, W&B, and long training jobs.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Batch, Data


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT2.src.model.gaot_3d import GAOT3D as GAOT2Model  # noqa: E402
from subprojects.wp1_cfd_ai.AISCO_GAOT.GAOT3.src.model.gaot_3d import GAOT3D as GAOT3Model  # noqa: E402


def _tiny_model_config() -> dict:
    return {
        "input_size": 1,
        "output_size": 1,
        "latent_tokens": [2, 2, 2],
        "field_output_size": 0,
        "norm_domain": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "magno": {
            "use_gno": True,
            "gno_coord_dim": 3,
            "neighbor_strategy": "radius",
            "projection_channels": 8,
            "in_gno_channel_mlp_hidden_layers": [8],
            "out_gno_channel_mlp_hidden_layers": [8],
            "lifting_channels": 8,
            "gno_radius": 10.0,
            "attention_type": "cosine",
            "use_geoembed": [False, False],
            "embedding_method": "statistical",
            "encoder_feature_attr": "x",
            "mlp_type": "linear",
            "precompute_edges": True,
            "scales": [1.0],
            "use_scale_weights": False,
            "k_neighbors": 1,
            "in_gno_transform_type": "linear",
            "out_gno_transform_type": "linear",
            "sampling_strategy": None,
            "max_neighbors": None,
            "sample_ratio": None,
        },
        "transformer": {
            "patch_size": 1,
            "hidden_size": 8,
            "use_attn_norm": True,
            "use_ffn_norm": True,
            "norm_eps": 1e-6,
            "num_layers": 2,
            "positional_embedding": "absolute",
            "use_long_range_skip": True,
            "attn_config": {
                "hidden_size": 8,
                "num_heads": 2,
                "num_kv_heads": 2,
                "use_conditional_norm": False,
                "cond_norm_hidden_size": 4,
                "atten_dropout": 0.0,
                "positional_embedding": "absolute",
            },
            "ffn_config": {
                "hidden_size": 16,
                "use_conditional_norm": False,
                "cond_norm_hidden_size": 4,
            },
        },
        "decoder": {
            "hidden_features": 16,
            "num_shared_layers": 2,
            "output_heads": {"force": 1, "cd": 1},
        },
        "scalar_encoder": {"enabled": False},
        "late_scalar_encoder": {"enabled": False},
    }


def _local_encoder_edges(num_nodes: int) -> torch.Tensor:
    """Build source->latent edges with varying neighbor counts per query."""
    edges = []
    for query_idx in range(num_nodes):
        for source_idx in range(query_idx + 1):
            edges.append((source_idx, query_idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _local_decoder_edges(num_nodes: int) -> torch.Tensor:
    """Build latent->physical edges for the real trainer's precomputed-edge path."""
    edges = []
    for physical_idx in range(num_nodes):
        for latent_idx in range(physical_idx + 1):
            edges.append((latent_idx, physical_idx))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _synthetic_batch(device: torch.device) -> Batch:
    base_pos = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    edge_index = _local_encoder_edges(num_nodes=base_pos.shape[0])
    graphs = []
    for graph_idx in range(2):
        offset = torch.tensor([0.05 * graph_idx, -0.03 * graph_idx, 0.02 * graph_idx])
        x = torch.linspace(-0.5, 0.5, base_pos.shape[0], dtype=torch.float32).unsqueeze(-1)
        graphs.append(
            Data(
                pos=base_pos + offset,
                x=x + graph_idx,
                encoder_edge_index_s0=edge_index,
            )
        )
    return Batch.from_data_list(graphs).to(device)


def _synthetic_training_sample(sample_idx: int, num_latent_nodes: int) -> Data:
    base_pos = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    offset = torch.tensor([0.03 * sample_idx, -0.02 * sample_idx, 0.01 * sample_idx], dtype=torch.float32)
    pos = base_pos + offset
    x = (torch.linspace(-0.3, 0.3, pos.shape[0], dtype=torch.float32) + 0.05 * sample_idx).unsqueeze(-1)
    c = torch.nn.functional.normalize(pos + 2.0, dim=-1)
    face = torch.tensor(
        [
            [0, 0, 0, 0, 4, 4, 2, 2, 1, 1, 0, 0],
            [1, 2, 4, 3, 5, 6, 3, 6, 5, 3, 4, 2],
            [2, 4, 1, 7, 6, 7, 7, 3, 3, 7, 6, 6],
        ],
        dtype=torch.long,
    )
    return Data(
        pos=pos,
        x=x,
        c=c,
        face=face,
        force=torch.tensor([0.15 + 0.01 * sample_idx], dtype=torch.float32),
        cd=torch.tensor([0.03 + 0.005 * sample_idx], dtype=torch.float32),
        encoder_edge_index_s0=_local_encoder_edges(pos.shape[0]),
        decoder_edge_index_s0=_local_decoder_edges(pos.shape[0]),
        num_latent_nodes=num_latent_nodes,
        filename=f"sample_{sample_idx}",
    )


@dataclass
class DifferenceReport:
    sections: dict[str, list[str]] = field(default_factory=dict)

    def add(self, section: str, message: str) -> None:
        self.sections.setdefault(section, []).append(message)

    @property
    def has_differences(self) -> bool:
        return any(self.sections.values())

    def print(self) -> None:
        if not self.has_differences:
            print("No differences found.")
            return

        print("Differences found:")
        for section, messages in self.sections.items():
            print(f"\n[{section}]")
            for message in messages:
                print(f"- {message}")

    def fail_if_needed(self) -> None:
        if self.has_differences:
            self.print()
            raise AssertionError("GAOT2/GAOT3 equivalence check failed. See full difference report above.")


def _make_model(model_cls: type[torch.nn.Module], device: torch.device, training: bool = False) -> torch.nn.Module:
    torch.manual_seed(12345)
    model = model_cls(config=copy.deepcopy(_tiny_model_config())).to(device)
    model.train(training)
    return model


def _first_different_index(lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float) -> tuple[int, ...] | None:
    if lhs.is_floating_point() or lhs.is_complex():
        close = torch.isclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True)
    else:
        close = lhs == rhs
    different = ~close
    if not different.any():
        return None
    flat_idx = int(different.reshape(-1).nonzero()[0].item())
    return tuple(int(i) for i in torch.unravel_index(torch.tensor(flat_idx), lhs.shape))


def _format_value(value: torch.Tensor) -> str:
    if value.numel() != 1:
        return str(value.detach().cpu().tolist())
    return f"{value.detach().cpu().item():.9g}"


def _tensor_diff_message(name: str, lhs: torch.Tensor, rhs: torch.Tensor, atol: float, rtol: float) -> str | None:
    if lhs.shape != rhs.shape:
        return f"{name}: shape differs, GAOT2={tuple(lhs.shape)} GAOT3={tuple(rhs.shape)}"
    if lhs.dtype != rhs.dtype:
        return f"{name}: dtype differs, GAOT2={lhs.dtype} GAOT3={rhs.dtype}"
    if lhs.device != rhs.device:
        rhs = rhs.to(lhs.device)
    if lhs.is_floating_point() or lhs.is_complex():
        if torch.allclose(lhs, rhs, atol=atol, rtol=rtol, equal_nan=True):
            return None
        abs_diff = (lhs - rhs).abs()
    else:
        if torch.equal(lhs, rhs):
            return None
        abs_diff = (lhs - rhs).abs().float()

    first_idx = _first_different_index(lhs, rhs, atol=atol, rtol=rtol)
    first_detail = ""
    if first_idx is not None:
        first_detail = (
            f", first_diff_index={first_idx}, "
            f"GAOT2={_format_value(lhs[first_idx])}, GAOT3={_format_value(rhs[first_idx])}"
        )

    return (
        f"{name}: shape={tuple(lhs.shape)}, max_abs_diff={abs_diff.max().item():.6e}, "
        f"mean_abs_diff={abs_diff.mean().item():.6e}{first_detail}"
    )


def _compare_tensor_maps(
    section: str,
    label_v2: str,
    label_v3: str,
    values_v2: dict[str, torch.Tensor],
    values_v3: dict[str, torch.Tensor],
    report: DifferenceReport,
    atol: float,
    rtol: float,
) -> None:
    keys_v2 = set(values_v2)
    keys_v3 = set(values_v3)
    for key in sorted(keys_v2 - keys_v3):
        report.add(section, f"{key}: present in {label_v2}, missing in {label_v3}")
    for key in sorted(keys_v3 - keys_v2):
        report.add(section, f"{key}: present in {label_v3}, missing in {label_v2}")

    for key in sorted(keys_v2 & keys_v3):
        message = _tensor_diff_message(key, values_v2[key], values_v3[key], atol=atol, rtol=rtol)
        if message is not None:
            report.add(section, message)


def _assert_state_dicts_match(
    model_v2: torch.nn.Module,
    model_v3: torch.nn.Module,
    report: DifferenceReport,
    section: str,
    atol: float,
    rtol: float,
) -> None:
    state_v2 = model_v2.state_dict()
    state_v3 = model_v3.state_dict()
    _compare_tensor_maps(section, "GAOT2", "GAOT3", state_v2, state_v3, report, atol=atol, rtol=rtol)


def _targets(device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "force": torch.tensor([[0.25], [-0.125]], dtype=torch.float32, device=device),
        "cd": torch.tensor([[0.03125], [0.0625]], dtype=torch.float32, device=device),
    }


def _normal_training_model_config() -> dict:
    config = copy.deepcopy(_tiny_model_config())
    config["input_size"] = 1
    config["output_size"] = 2
    config["field_output_size"] = 0
    config["magno"]["encoder_feature_attr"] = "x"
    config["magno"]["precompute_edges"] = True
    config["magno"]["asynchronous_graph_building"] = False
    config["decoder"]["output_heads"] = {"force": 1, "cd": 1}
    return config


def _write_normal_training_dataset(data_root: Path, processed_folder: str) -> None:
    processed_dir = data_root / processed_folder
    processed_dir.mkdir(parents=True, exist_ok=True)
    names = [f"sample_{idx}" for idx in range(4)]
    (data_root / f"order_{processed_folder}.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    num_latent_nodes = 2 * 2 * 2
    for idx, name in enumerate(names):
        torch.save(_synthetic_training_sample(idx, num_latent_nodes=num_latent_nodes), processed_dir / f"{name}.pt")


def _load_main_config(version: str) -> dict:
    config_path = REPO_ROOT / "subprojects" / "wp1_cfd_ai" / "AISCO_GAOT" / version / "config" / "main_config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"Expected mapping in {config_path}, got {type(config).__name__}")
    return config


def _normal_training_config(version: str, root: Path, train_epochs: int, device_name: str) -> dict:
    data_root = root / "data"
    results_root = root / "results"
    processed_folder = "processed"
    _write_normal_training_dataset(data_root, processed_folder)

    config = _load_main_config(version)

    config.setdefault("setup", {})
    config["setup"].update(
        {
            "name": f"{version.lower()}_normal_equivalence",
            "id": "normal_equivalence",
            "seed": 12345,
            "device": device_name,
            "dtype": "torch.float32",
            "trainer_name": "static3d",
            "train": True,
            "test": False,
            "ckpt": False,
            "use_variance_test": False,
            "distributed": False,
            "wandb": False,
            "wandb_run_name": f"{version.lower()}_normal_equivalence",
            "wandb_project": "disabled",
            "wandb_entity": "disabled",
            "wandb_group": "equivalence",
            "wandb_mode": "disabled",
            "wandb_tags": ["equivalence"],
            "wandb_watch_model": False,
            "wandb_notes": "Synthetic GAOT2/GAOT3 full-pipeline equivalence check",
            "test_during_training": False,
        }
    )

    config.setdefault("model", {})
    config["model"].update({"name": "gaot_3d", "use_conditional_norm": False, "args": _normal_training_model_config()})

    config.setdefault("dataset", {})
    config["dataset"].update(
        {
            "name": f"{version.lower()}_synthetic_equivalence",
            "metaname": "incompressible_fluids/euler_pressure",
            "base_path": str(data_root),
            "processed_folder": processed_folder,
            "force_recompute_stats": True,
            "rand_dataset": False,
            "train_size": 2,
            "val_size": 1,
            "test_size": 1,
            "batch_size": 2,
            "num_workers": 0,
            "shuffle": False,
            "metric_suite": "general",
            "update_pt_files_with_edges": False,
            "training_strategy": "full_grid",
            "uncertainty_weighting": False,
            "field_loss_weight": 0.0,
            "use_rescale_new": False,
            "active_variables": None,
            "order_file_name": None,
            "active_heads": ["force", "cd"],
            "drop_pressure": False,
            "subsample_nodes": None,
            "subsample_strategy": "random",
            "v_inf": 25.0,
            "rho": 1.225,
            "num_viz_samples": 0,
        }
    )

    config.setdefault("optimizer", {})
    config["optimizer"]["name"] = "adamw"
    config["optimizer"].setdefault("args", {})
    config["optimizer"]["args"].update(
        {
            "lr": 1e-3,
            "weight_decay": 0.0,
            "epoch": train_epochs,
            "loss_scale": 1.0,
            "eval_every_eps": 1,
            "viz_every_eps": 0,
            "viz_warmup_epochs": 0,
            "scheduler": None,
            "save_every_epochs": 0,
            "save_every_steps": 0,
            "early_save_metric": "train",
            "loss": "mse",
        }
    )

    config["path"] = {
        "ckpt_path": str(results_root / "model.pt"),
        "loss_path": str(results_root / "loss.png"),
        "result_path": str(results_root / "results.png"),
        "config_save_path": str(results_root / "config.yaml"),
        "database_path": str(results_root / "metrics.csv"),
    }

    return config


def _run_normal_training(version: str, root: Path, train_epochs: int, device_name: str) -> dict:
    version_dir = REPO_ROOT / "subprojects" / "wp1_cfd_ai" / "AISCO_GAOT" / version
    run_root = root / version
    run_root.mkdir(parents=True, exist_ok=True)
    config = _normal_training_config(version, run_root, train_epochs=train_epochs, device_name=device_name)
    config_path = run_root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), str(version_dir), env.get("PYTHONPATH", "")])
    env["WANDB_DISABLED"] = "true"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    result = subprocess.run(
        [sys.executable, "main.py", "-c", str(config_path)],
        cwd=version_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
        check=False,
    )

    return {
        "returncode": result.returncode,
        "config_path": config_path,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ckpt_path": Path(config["path"]["ckpt_path"]),
        "model_config_path": Path(config["path"]["ckpt_path"]).with_name("model_config.yaml"),
        "full_config_path": Path(config["path"]["config_save_path"]),
        "metrics_path": Path(config["path"]["database_path"]),
        "epoch_metrics_path": Path(config["path"]["result_path"]).with_name("results_epoch_metrics.csv"),
    }


def _load_checkpoint_model_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Expected checkpoint was not created: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain a 'model' entry: {path}")
    return checkpoint["model"]


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Expected config was not created: {path}")
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded or {}


def _flatten_yaml_scalars(prefix: str, value) -> dict[str, torch.Tensor]:
    if isinstance(value, dict):
        flattened = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_yaml_scalars(child_prefix, child))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            flattened.update(_flatten_yaml_scalars(child_prefix, child))
        return flattened
    if value is None:
        return {prefix: torch.tensor([0])}
    if isinstance(value, bool):
        return {prefix: torch.tensor([int(value)])}
    if isinstance(value, (int, float)):
        return {prefix: torch.tensor([float(value)], dtype=torch.float64)}
    return {prefix: torch.tensor([hash(str(value))], dtype=torch.int64)}


def _compare_normal_training_outputs(
    root: Path,
    report: DifferenceReport,
    train_epochs: int,
    device_name: str,
    atol: float,
    rtol: float,
) -> None:
    print(f"Running real GAOT main.py training for GAOT2 and GAOT3 in {root}")
    runs = {}
    for version in ["GAOT2", "GAOT3"]:
        try:
            runs[version] = _run_normal_training(version, root, train_epochs=train_epochs, device_name=device_name)
        except Exception as exc:
            report.add("normal main.py training execution", f"{version}: failed to launch normal training: {exc}")

    for version, run in runs.items():
        if run["returncode"] != 0:
            report.add(
                "normal main.py training execution",
                f"{version}: failed with exit code {run['returncode']}.\n"
                f"Config: {run['config_path']}\n"
                f"STDOUT:\n{run['stdout'][-4000:]}\n"
                f"STDERR:\n{run['stderr'][-4000:]}",
            )

    if set(runs) != {"GAOT2", "GAOT3"} or any(run["returncode"] != 0 for run in runs.values()):
        for version, paths in runs.items():
            print(f"{version} normal training attempted with config: {paths['config_path']}")
        return

    state_v2 = _load_checkpoint_model_state(runs["GAOT2"]["ckpt_path"])
    state_v3 = _load_checkpoint_model_state(runs["GAOT3"]["ckpt_path"])
    _compare_tensor_maps(
        f"normal main.py checkpoint after {train_epochs} epochs",
        "GAOT2",
        "GAOT3",
        state_v2,
        state_v3,
        report,
        atol=atol,
        rtol=rtol,
    )

    model_config_v2 = _flatten_yaml_scalars("", _load_yaml(runs["GAOT2"]["model_config_path"]))
    model_config_v3 = _flatten_yaml_scalars("", _load_yaml(runs["GAOT3"]["model_config_path"]))
    _compare_tensor_maps(
        "normal main.py saved model_config.yaml",
        "GAOT2",
        "GAOT3",
        model_config_v2,
        model_config_v3,
        report,
        atol=0.0,
        rtol=0.0,
    )

    for version, paths in runs.items():
        print(f"{version} normal training artifacts:")
        for name in ["ckpt_path", "model_config_path", "full_config_path", "metrics_path", "epoch_metrics_path"]:
            print(f"  {name}: {paths[name]}")


def _training_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    losses = []
    for key, target in targets.items():
        losses.append(torch.nn.functional.mse_loss(outputs[key], target))
    return torch.stack(losses).mean()


def _train_for_equivalence(
    model: torch.nn.Module,
    batch: Batch,
    targets: dict[str, torch.Tensor],
    epochs: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    loss_trace = []

    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss = _training_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_trace.append(loss.detach())

    model.eval()
    with torch.no_grad():
        final_outputs = {key: value.detach().clone() for key, value in model(batch).items()}
    return torch.stack(loss_trace).detach(), final_outputs


def run_equivalence_check(
    device_name: str,
    atol: float,
    rtol: float,
    train_epochs: int,
    lr: float,
    skip_normal_training: bool,
    keep_normal_training_artifacts: bool,
) -> None:
    device = torch.device(device_name)
    torch.manual_seed(12345)

    model_v2 = _make_model(GAOT2Model, device)
    model_v3 = _make_model(GAOT3Model, device)
    report = DifferenceReport()
    _assert_state_dicts_match(model_v2, model_v3, report, "initial state_dict", atol=atol, rtol=rtol)

    batch = _synthetic_batch(device)
    with torch.no_grad():
        output_v2 = {key: value.detach().clone() for key, value in model_v2(batch).items()}
        output_v3 = {key: value.detach().clone() for key, value in model_v3(batch).items()}
    _compare_tensor_maps("initial forward outputs", "GAOT2", "GAOT3", output_v2, output_v3, report, atol=atol, rtol=rtol)

    train_v2 = _make_model(GAOT2Model, device, training=True)
    train_v3 = _make_model(GAOT3Model, device, training=True)
    targets = _targets(device)
    loss_trace_v2, trained_output_v2 = _train_for_equivalence(train_v2, batch, targets, epochs=train_epochs, lr=lr)
    loss_trace_v3, trained_output_v3 = _train_for_equivalence(train_v3, batch, targets, epochs=train_epochs, lr=lr)

    _compare_tensor_maps(
        "training loss trace",
        "GAOT2",
        "GAOT3",
        {"loss": loss_trace_v2},
        {"loss": loss_trace_v3},
        report,
        atol=atol,
        rtol=rtol,
    )
    _assert_state_dicts_match(train_v2, train_v3, report, f"state_dict after {train_epochs} training epochs", atol=atol, rtol=rtol)
    _compare_tensor_maps(
        f"forward outputs after {train_epochs} training epochs",
        "GAOT2",
        "GAOT3",
        trained_output_v2,
        trained_output_v3,
        report,
        atol=atol,
        rtol=rtol,
    )

    normal_training_root = Path(tempfile.mkdtemp(prefix="gaot_equivalence_normal_"))
    try:
        if not skip_normal_training:
            _compare_normal_training_outputs(
                normal_training_root,
                report,
                train_epochs=train_epochs,
                device_name=device_name,
                atol=atol,
                rtol=rtol,
            )
    except Exception as exc:
        report.add("normal main.py training execution", str(exc))
    finally:
        if skip_normal_training:
            shutil.rmtree(normal_training_root, ignore_errors=True)
        elif keep_normal_training_artifacts or report.has_differences:
            print(f"Kept normal-training artifacts at: {normal_training_root}")
        else:
            shutil.rmtree(normal_training_root, ignore_errors=True)

    report.fail_if_needed()

    for key in sorted(output_v2):
        max_abs_diff = (output_v2[key] - output_v3[key]).abs().max().item()
        print(f"initial {key}: shape={tuple(output_v2[key].shape)} max_abs_diff={max_abs_diff:.3e}")
    for key in sorted(trained_output_v2):
        max_abs_diff = (trained_output_v2[key] - trained_output_v3[key]).abs().max().item()
        print(f"trained {key}: shape={tuple(trained_output_v2[key].shape)} max_abs_diff={max_abs_diff:.3e}")
    print(f"training loss trace: {loss_trace_v2.detach().cpu().tolist()}")

    if not skip_normal_training:
        print(f"GAOT2 and GAOT3 normal main.py training artifacts match after {train_epochs} epochs.")
    print(f"GAOT2 and GAOT3 are equivalent after initial forward and {train_epochs} training epochs.")


def test_gaot2_gaot3_forward_equivalence() -> None:
    run_equivalence_check(
        device_name="cpu",
        atol=1e-6,
        rtol=1e-6,
        train_epochs=5,
        lr=1e-3,
        skip_normal_training=False,
        keep_normal_training_artifacts=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify deterministic GAOT2/GAOT3 model equivalence.")
    parser.add_argument("--device", default="cpu", help="Torch device to run on, e.g. cpu or cuda:0.")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--train-epochs", type=int, default=5, help="Number of tiny synthetic training epochs to compare.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the tiny synthetic training check.")
    parser.add_argument(
        "--skip-normal-training",
        action="store_true",
        help="Skip the full main.py training-path comparison and run only the fast in-memory check.",
    )
    parser.add_argument(
        "--keep-normal-training-artifacts",
        action="store_true",
        help="Keep temporary datasets/configs/checkpoints from the full main.py training-path comparison.",
    )
    args = parser.parse_args()
    run_equivalence_check(
        device_name=args.device,
        atol=args.atol,
        rtol=args.rtol,
        train_epochs=args.train_epochs,
        lr=args.lr,
        skip_normal_training=args.skip_normal_training,
        keep_normal_training_artifacts=args.keep_normal_training_artifacts,
    )


if __name__ == "__main__":
    main()
