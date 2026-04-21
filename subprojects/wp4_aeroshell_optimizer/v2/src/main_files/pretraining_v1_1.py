import importlib
import os

import numpy as np
import torch
import trimesh

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.ground_truth_masks.simple_geoms import (
    cube_gt,
    ply_gt,
    sphere_gt,
)  # Added cube_gt
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.debug_manager import (
    DebugManager,
    collect_cuda_memory,
    collect_optimizer_state,
    collect_param_stats,
)


def dynamic_model_loading(config_name: str = "pretraining_v0_config.yaml") -> SupportsModelContract:
    """
    Dynamically loads the model architecture based on the configuration file.
    I.e. imports only from the relevant file, not all the models
    """
    config = load_config(cfg_name=config_name)  # still not great
    model_name = config["model"]  # dynamically import the model module
    model_file_name = config["model_filename"]  # file where the model is located
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v2.src.models." + model_file_name)
    ModelClass = getattr(module, model_name)

    # adjust to include config parameters here, stil WIP
    model = ModelClass()
    return model


def _apply_override_config(config: dict, override_config: dict) -> dict:
    for k, v in override_config.items():
        if k in config:
            if isinstance(config[k], dict) and isinstance(v, dict):
                config[k].update(v)
            else:
                config[k] = v
        elif "." in k:
            parts = k.split(".")
            if parts[0] in config and isinstance(config[parts[0]], dict):
                config[parts[0]][parts[1]] = v
    return config


def _load_ply_mesh(config: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    ply_path = config.get("pretraining_ply_path")
    if not ply_path:
        raise ValueError("pretraining_shape='ply' requires 'pretraining_ply_path' in pretraining config")
    if not os.path.isfile(ply_path):
        fallback = config.get("beast_pretraining_ply_path", "")
        if fallback and os.path.isfile(fallback):
            print(f"[pretraining] '{ply_path}' not found, falling back to beast_pretraining_ply_path: '{fallback}'")
            ply_path = fallback
        else:
            raise ValueError(
                f"PLY file not found: '{ply_path}'. Set 'beast_pretraining_ply_path' in pretraining config for local runs."
            )
    ply_scale = float(config.get("pretraining_ply_scale", 1000.0))
    raw_mesh = trimesh.load(ply_path)
    assert isinstance(raw_mesh, trimesh.Trimesh), f"Expected Trimesh, got {type(raw_mesh)}"
    ply_verts_t = torch.tensor(np.array(raw_mesh.vertices, dtype="float32") * ply_scale, device=device)
    ply_faces_t = torch.tensor(np.array(raw_mesh.faces, dtype="int64"), device=device)
    print(
        f"[pretraining] Loaded PLY '{ply_path}' | scale={ply_scale} | vertices={ply_verts_t.shape[0]} faces={ply_faces_t.shape[0]}"
    )
    return ply_verts_t, ply_faces_t


def _compute_gt_target(
    pretraining_shape: str,
    input_data: torch.Tensor,
    ply_verts_t: torch.Tensor | None,
    ply_faces_t: torch.Tensor | None,
) -> torch.Tensor:
    if pretraining_shape == "cube":
        return cube_gt(input_data, side_length=2000.0)
    if pretraining_shape == "ply":
        assert ply_verts_t is not None and ply_faces_t is not None
        return ply_gt(input_data, ply_verts_t, ply_faces_t)
    return sphere_gt(input_data)


def _compute_eikonal(model, input_data: torch.Tensor, device: str) -> torch.Tensor:
    input_m = (input_data / 1000.0).detach().requires_grad_(True)
    input_mm_for_model = input_m * 1000.0
    sdf_m = (model(input_mm_for_model).squeeze()) / 1000.0
    grad = torch.autograd.grad(
        outputs=sdf_m,
        inputs=input_m,
        grad_outputs=torch.ones_like(sdf_m),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is not None:
        return ((grad.norm(dim=-1) - 1.0) ** 2).mean()
    print("WARNING: Gradient is still None. Check model connectivity.")
    return torch.tensor(0.0, device=device)


# 2026 03 05, eschaerer:
# look i'm not sure if the changes i made also work on the HPC. Please revert them if they don't
def pretraining(
    model: SupportsModelContract | None = None,
    pretraining_config_name: str = "pretraining_v0_config.yaml",
    override_config: dict | None = None,
    debug_manager: DebugManager | None = None,
) -> SupportsModelContract:
    # Does pretraining, for now only on a sphere

    config = load_config(v="v2", cfg_name=pretraining_config_name)
    if override_config:
        config = _apply_override_config(config, override_config)

    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        print("Careful: No model provided, dynamically loading model based on config...")
        model = dynamic_model_loading(config_name=pretraining_config_name)
    model = model.to(device)

    pretraining_shape = config.get("pretraining_shape", "sphere")

    # Pre-load PLY mesh once so mesh_sdf can be called efficiently each batch
    ply_verts_t = ply_faces_t = None
    if pretraining_shape == "ply":
        ply_verts_t, ply_faces_t = _load_ply_mesh(config, device)

    # Initialise a no-op DebugManager if none was provided
    dbg = debug_manager if debug_manager is not None else DebugManager(config={"GLOBAL_DEBUG": False})

    # Training loop
    learning_rate = config["training_loop"]["learning_rate"]
    num_epochs = config["training_loop"]["epochs"]
    batch_size = config["training_loop"]["random_batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── Debug: one-time run config dump ──────────────────────
    dbg.log_run_config(
        model=model,
        device=device,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        optimizer_type="Adam",
        extra={
            "pretraining_shape": pretraining_shape,
            "batch_size": batch_size,
            "print_interval": config["training_loop"]["print_interval"],
            "eikonal_weight": 0.1,
        },
    )

    for i in range(num_epochs):
        # ── Debug: start-of-iteration stats tracking ─────────────
        dbg.begin_iteration(i)

        optimizer.zero_grad()

        # Constrained uniform sampling in a tighter box around the sphere (±2.5 m)
        input_data = (torch.rand(config["training_loop"]["random_batch_size"], 3, device=device) - 0.5) * 5000.0

        output = model(input_data)
        target = _compute_gt_target(pretraining_shape, input_data, ply_verts_t, ply_faces_t)
        eikonal = _compute_eikonal(model, input_data, device)

        loss = torch.nn.MSELoss()((target / 1000).squeeze(), (output / 1000).squeeze()) + 0.1 * eikonal

        loss.backward()

        # ── Debug: compute aggregate gradient norm ───────────────
        param_grad_norm = sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None) ** 0.5

        loss_val = loss.item()
        mse_val = (loss - 0.1 * eikonal).item()
        eikonal_val = eikonal.item()
        lr = optimizer.param_groups[0]["lr"]

        # ── Debug: print raw values ──────────────────────────────
        dbg.log_values(
            i,
            inputs=input_data,
            outputs=output,
            loss=loss_val,
            loss_components={"mse": mse_val, "eikonal": eikonal_val, "eikonal_weighted": 0.1 * eikonal_val},
            grad_norm=param_grad_norm,
            lr=lr,
            param_stats=collect_param_stats(model) if dbg.print_raw_values and dbg.log_param_stats else None,
            optimizer_state=collect_optimizer_state(optimizer) if dbg.print_raw_values and dbg.log_optimizer_state else None,
            cuda_memory=collect_cuda_memory() if dbg.print_raw_values and dbg.log_cuda_memory else None,
            loss_weights={"mse": 1.0, "eikonal": 0.1},
            extra={
                "target_stats": f"min={target.min().item():.2f} max={target.max().item():.2f} mean={target.mean().item():.2f}",
                "output_stats": f"min={output.min().item():.2f} max={output.max().item():.2f} mean={output.mean().item():.2f}",
                "batch_size": input_data.shape[0],
                "device": device,
            },
        )

        # ── Debug: check warnings ────────────────────────────────
        dbg.check_warnings(
            i,
            loss=loss_val,
            grad_norm=param_grad_norm,
            loss_components={"mse": mse_val, "eikonal": eikonal_val},
        )

        # ── Debug: generate insights ─────────────────────────────
        dbg.check_insights(
            i,
            loss=loss_val,
            lr=lr,
            grad_norm=param_grad_norm,
        )

        optimizer.step()

        # ── Debug: end-of-iteration stats ────────────────────────
        dbg.end_iteration(
            i,
            loss=loss_val,
            grad_norm=param_grad_norm,
            lr=lr,
        )

        if i % config["training_loop"]["print_interval"] == 0:
            print(f"Epoch {i}, Loss: {loss_val:.4f}, Eikonal: {eikonal_val:.4f}")

    # ── Debug: cleanup ───────────────────────────────────────────
    dbg.stop()
    return model
