import importlib

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.ground_truth_masks.simple_geoms import cube_gt, sphere_gt  # Added cube_gt
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.debug_manager import (
    DebugManager,
    collect_cuda_memory,
    collect_optimizer_state,
    collect_param_stats,
)


def dynamic_model_loading(config_name: str = "pretraining_v0_config.yaml"):
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


# 2026 03 05, eschaerer:
# look i'm not sure if the changes i made also work on the HPC. Please revert them if they don't
def pretraining(
    model: torch.nn.Module | None = None,
    pretraining_config_name: str = "pretraining_v0_config.yaml",
    debug_manager: DebugManager | None = None,
):
    # Does pretraining, for now only on a sphere

    config = load_config(v="v2", cfg_name=pretraining_config_name)

    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        print("Careful: No model provided, dynamically loading model based on config...")
        model = dynamic_model_loading(config_name=pretraining_config_name)
    model = model.to(device)

    pretraining_shape = config.get("pretraining_shape", "sphere")

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

        # Dummy target data, replace with actual target generation
        target = cube_gt(input_data, side_length=2000.0) if pretraining_shape == "cube" else sphere_gt(input_data)

        # Eikonal regularization: ||∇f(x)|| = 1 in meters

        # 1. Start the 'chain' here
        input_m = (input_data / 1000.0).detach().requires_grad_(True)

        # 2. Re-scale for the model (assuming model wants mm)
        # By multiplying input_m, we keep the grad history alive
        input_mm_for_model = input_m * 1000.0

        # 3. PASS THE NEW TENSOR TO THE MODEL
        sdf_m = (model(input_mm_for_model).squeeze()) / 1000.0

        # 4. Now 'grad' will actually contain values
        grad = torch.autograd.grad(
            outputs=sdf_m,
            inputs=input_m,
            grad_outputs=torch.ones_like(sdf_m),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        # Add a safety check for the norm
        if grad is not None:
            eikonal = ((grad.norm(dim=-1) - 1.0) ** 2).mean()
        else:
            print("WARNING: Gradient is still None. Check model connectivity.")
            eikonal = torch.tensor(0.0, device=input_data.device)
        # eikonal = torch.tensor(0.0, device=device)  # Placeholder if Eikonal is disabled # noqa: ERA001
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
