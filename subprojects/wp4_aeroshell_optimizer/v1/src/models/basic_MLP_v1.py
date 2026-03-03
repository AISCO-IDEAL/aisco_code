# Basic MLP Model

import torch
import wandb

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager, Wrapper


class BasicMLP(BaseModel):
    def __init__(
        self, config: dict | None = None, overwrite_base_init: bool = False, in_dim: int = 3, hidden: int = 128, layers: int = 6
    ):
        if config is None:
            config = {}
        super().__init__(config, overwrite_base_init, in_dim, hidden, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

    def optimise(
        self,
        renderer_manager: VisualisationManager | None = None,
        renderer_wrapper: Wrapper | None = None,  # noqa: ARG002
        wandb_name: str | None = None,  # noqa: ARG002
        general_config: dict | None = None,  # noqa: ARG002
        training_config: dict | None = None,
    ) -> None:
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()  # Set model to training mode, i.e. things like dropout layers are active, turn off by using self.eval() or model.eval()

        # Check for WandB sweep overrides for learning rate
        if wandb.run and "learning_rate " in wandb.config:
            learning_rate = wandb.config.learning_rate
            print(f"Sweep: Updated learning_rate to {learning_rate}")
        else:
            learning_rate = training_config.get("learning_rate", 1e-3)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        num_epochs = training_config["training_params"].get("num_epochs", 1000)

        # IMPORTANT: the device the model is on at time of function call defines the device passed to the loss_class
        # Get device from model parameters since nn.Module doesn't have a .device attribute
        device = str(next(self.parameters()).device)
        loss_function = TotalLoss(device, renderer_manager=renderer_manager)  # Initialise TotalLoss with the correct device

        for epoch in range(num_epochs):
            # Clear gradients
            optimizer.zero_grad()

            # Calculate loss, Tracking should also be done in TotalLoss
            loss = loss_function.get_total_loss(model=self, epoch_name=epoch)

            # Optimize w.r.t. total loss
            # Do this by calculating the gradients of the weights w.r.t. the loss
            loss.backward()
            # The optimizer takes the weights of self.parameters(), i.e. the weights and updates them based on the gradients calculated in loss.backward()
            optimizer.step()

    def get_model_info_dict(self):
        # Return model info as a dictionary
        model_info = {
            "Model_name": "BasicMLP",
            "in_dim": self.in_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        return model_info

    def debug_info(self):
        # Print model architecture and number of parameters, or anthing you might want to debug
        print("===================================")
        print("Model Debug Info:")

        print(self)

        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
        print("Model Layers:")
        for i, layer in enumerate(self.model_layers):
            print(f" Layer {i}: {layer}")

        print("Device/Gradient Info:")
        device = next(self.parameters()).device
        print(f" Model is on device: {device}")
        print(f"Do Model Parameters require Gradients? (Y/N): {any(p.requires_grad for p in self.parameters())}")

        print("End of Model Debug Info")
        print("===================================")
