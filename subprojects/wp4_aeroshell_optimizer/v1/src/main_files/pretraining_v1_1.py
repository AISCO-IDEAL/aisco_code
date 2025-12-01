import importlib

import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.ground_truth_masks.simple_geoms import sphere_gt
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_general_config


def dynamic_model_loading(config_name: str = "pretraining_v0_config.yaml"):
    """
    Dynamically loads the model architecture based on the configuration file.
    I.e. imports only from the relevant file, not all the models
    """
    config = load_general_config(cfg_name=config_name)  # still not great
    model_name = config["model"]  # dynamically import the model module
    model_file_name = config["model_filename"]  # file where the model is located
    module = importlib.import_module("subprojects.wp4_aeroshell_optimizer.v1.src.models." + model_file_name)
    ModelClass = getattr(module, model_name)  # noqa: N806

    # adjust to include config parameters here, stil WIP
    model = ModelClass()
    return model


def pretraining(model: BaseModel = None, pretraining_config_name: str = "pretraining_v0_config.yaml"):
    # Does pretraining, for now only on a sphere

    config = load_general_config(v="v1", cfg_name=pretraining_config_name)
    # Try to use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training_loop"]["learning_rate"])
    for i in range(config["training_loop"]["epochs"]):
        optimizer.zero_grad()

        # Constrained uniform sampling in a tighter box around the sphere (±2.5 m)
        input_data = (torch.rand(config["training_loop"]["random_batch_size"], 3, device=device) - 0.5) * 5000.0

        output = model(input_data)

        # Dummy target data, replace with actual target generation
        target = sphere_gt(input_data)

        # Eikonal regularization: ||∇f(x)|| = 1 in meters

        input_m = (input_data / 1000.0).detach().requires_grad_(True)
        sdf_m = (model(input_m * 1000.0).squeeze()) / 1000.0
        grad = torch.autograd.grad(
            outputs=sdf_m,
            inputs=input_m,
            grad_outputs=torch.ones_like(sdf_m),
            create_graph=True,
            retain_graph=True,
        )[0]
        eikonal = ((grad.norm(dim=-1) - 1.0) ** 2).mean()

        # eikonal = torch.tensor(0.0, device=device)  # Placeholder if Eikonal is disabled # noqa: ERA001
        loss = torch.nn.MSELoss()((target / 1000).squeeze(), (output / 1000).squeeze()) + 0.1 * eikonal

        loss.backward()
        optimizer.step()

        if i % config["training_loop"]["print_interval"] == 0:
            print(f"Epoch {i}, Loss: {loss.item():.4f}, Eikonal: {eikonal.item():.4f}")

    return model
