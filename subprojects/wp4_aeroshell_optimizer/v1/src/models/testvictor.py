import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import Wrapper


class VictorModel(nn.Module):
    # For now just a copy of BasicMLP, to be modified later
    def __init__(self, config: dict = {}, in_dim: int = 3, hidden: int = 64, layers: int = 3, out_dim: int = 1):
        super().__init__()

        self.config = config
        self.in_dim = in_dim
        self.hidden = hidden
        self.layers = layers
        self.out_dim = out_dim

        self.model_layers = nn.ModuleList()
        self.model_layers.append(nn.Linear(in_dim, hidden))
        for i in range(layers - 1):
            self.model_layers.append(nn.Linear(hidden, hidden))
            # Alternate activations
            if i % 2:
                self.model_layers.append(nn.Tanh())
            else:
                self.model_layers.append(nn.ReLU())

        self.model_layers.append(nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 1000.0  # Scale input to meters
        for layer in self.model_layers:
            x = layer(x)
        output = x.squeeze(-1)
        return output * 1000.0

    def optimise(self, training_config: dict = None, renderer_wrapper: Wrapper | None = None) -> None:
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()  # Set model to training mode, i.e. things like dropout layers are active, turn off by using self.eval() or model.eval()
        optimizer = torch.optim.Adam(self.parameters(), lr=training_config.get("learning_rate", 1e-3))
        num_epochs = training_config["training_params"].get("num_epochs", 1000)

        # IMPORTANT: the device the model is on at time of function call defines the device passed to the loss_class
        # Get device from model parameters since nn.Module doesn't have a .device attribute
        device = next(self.parameters()).device
        loss_function = TotalLoss(device, renderer_wrapper=renderer_wrapper)  # Initialise TotalLoss with the correct device

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
