import importlib

import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.ground_truth_masks.simple_geoms import cube_gt, sphere_gt  # Added cube_gt
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config


def dynamic_model_loading(config_name: str = "pretraining_v0_config.yaml"):
    """
    Dynamically loads the model architecture based on the configuration file.
    I.e. imports only from the relevant file, not all the models
    """
    config = load_config(cfg_name=config_name)  # still not great
    model_name = config["model"]  # dynamically import the model module
    model_file_name = config["model_filename"]  # file where the model is located
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v1.src.models." + model_file_name)
    ModelClass = getattr(module, model_name)

    # adjust to include config parameters here, stil WIP
    model = ModelClass()
    return model


# 2026 03 05, eschaerer:
# look i'm not sure if the changes i made also work on the HPC. Please revert them if they don't
def pretraining(
    model: BaseModel | None = None,
    pretraining_config_name: str = "pretraining_v0_config.yaml",
    override_config: dict | None = None,
):
    # Does pretraining, for now only on a sphere

    config = load_config(v="v1", cfg_name=pretraining_config_name)
    if override_config:
        # Merge override_config into config
        # Simple recursive merge or specific key override
        for k, v in override_config.items():
            if k in config:
                if isinstance(config[k], dict) and isinstance(v, dict):
                    config[k].update(v)
                else:
                    config[k] = v
            # Also check if it's a flat wandb param that should go into nested dict
            # e.g. "training_loop.learning_rate" -> config["training_loop"]["learning_rate"]
            elif "." in k:
                parts = k.split(".")
                if parts[0] in config and isinstance(config[parts[0]], dict):
                    config[parts[0]][parts[1]] = v

    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = dynamic_model_loading(config_name=pretraining_config_name)
    model = model.to(device)

    pretraining_shape = config.get("pretraining_shape", "sphere")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training_loop"]["learning_rate"])
    for i in range(config["training_loop"]["epochs"]):
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
        optimizer.step()

        if i % config["training_loop"]["print_interval"] == 0:
            print(f"Epoch {i}, Loss: {loss.item():.4f}, Eikonal: {eikonal.item():.4f}")

    return model
