# WIRE model

import torch
from torch import nn

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import allround_save_data
from subprojects.wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import VisualisationManager, Wrapper

#
# WIRE architecture from https://arxiv.org/pdf/2301.05187

# This code is from the GINN PAPER repo: https://github.com/ml-jku/GINNs-Geometry-informed-Neural-Networks/blob/main/models/wire.py
# Modified by Alex to inherit from BaseModel class instead of nn.Module (BaseModel already inherits from nn.Module)

# Note that this implementation uses complex numbers!
#


class ComplexGaborLayer(nn.Module):
    """
    Implicit representation with complex Gabor nonlinearity

    Inputs;
        in_features: Input features
        out_features; Output features
        bias: if True, enable bias for the linear operation
        is_first: Legacy SIREN parameter
        omega_0: Legacy SIREN parameter
        omega0: Frequency of Gabor sinusoid term
        sigma0: Scaling of Gabor Gaussian term
        trainable: If True, omega and sigma are trainable parameters
    """

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega0=10.0, sigma0=40.0, trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first

        self.in_features = in_features

        dtype = torch.float if self.is_first else torch.cfloat

        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)

        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    def forward(self, input):
        lin = self.linear(input)
        omega = self.omega_0 * lin
        scale = self.scale_0 * lin

        return torch.exp(1j * omega - scale.abs().square())


class WIRE(BaseModel):
    def __init__(
        self,
        config: dict | None = None,
        in_features=3,
        hidden_features=256,
        hidden_layers=16,
        out_features=1,
        first_omega_0=30,
        hidden_omega_0=1.0,
        scale=1.0,
        trainable_params=True,
    ):
        if config is None:
            config = {}
        super().__init__(config=config)  # Initialize config to satisfy BaseModel inheritance (does nothing for now)

        # variables needed for model info stored here
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.trainable_params = trainable_params
        self.first_omega_0 = first_omega_0
        self.hidden_omega_0 = hidden_omega_0
        self.scale = scale
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of
        # hidden parameters by 2
        hidden_features = int(hidden_features / torch.sqrt(torch.tensor(2.0)))
        dtype = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"

        # Legacy parameter
        self.pos_encode = False

        net_layers = [
            self.nonlin(
                in_features, hidden_features, omega0=first_omega_0, sigma0=scale, is_first=True, trainable=self.trainable_params
            )
        ]

        net_layers.extend(
            [
                self.nonlin(
                    hidden_features, hidden_features, omega0=hidden_omega_0, sigma0=scale, trainable=self.trainable_params
                )
                for i in range(hidden_layers)
            ]
        )

        final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)
        net_layers.append(final_linear)

        self.net = nn.Sequential(*net_layers)

        # Small-weight init for stability: limit Linear weights, zero biases
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords / 1000  # Scale mm to meters, nicer for activations
        output = self.net(coords)

        if self.wavelet == "gabor":
            return 1000 * (output.real)  # Scale back to mm
        else:
            return output * 1000  # Scale back to mm

    def get_model_info_dict(self):
        model_info = {
            "in_features": self.in_features,
            "hidden_features": self.hidden_features,
            "hidden_layers": self.hidden_layers,
            "out_features": self.out_features,
            "trainable_params": self.trainable_params,
            "first_omega_0": self.first_omega_0,
            "hidden_omega_0": self.hidden_omega_0,
            "scale": self.scale,
            "total_parameters": sum(p.numel() for p in self.parameters()),
        }
        return model_info

    def optimise(
        self,
        renderer_manager: VisualisationManager | None = None,  # noqa: ARG002
        renderer_wrapper: Wrapper | None = None,
        wandb_name: str | None = None,
        general_config: dict | None = None,
        training_config: dict | None = None,
    ) -> None:
        """
        Args:
            training_config: Dictionary containing training configuration parameters.
            renderer_wrapper: Not used in this model, included for compatibility.
        Return:
            Nothing. Optimises the model in place, calculating the total loss and updating weights of the Neural Network until num_epochs is reached.
        """
        if training_config is None:
            raise ValueError("Training config must be provided for optimisation.")

        self.train()  # Set model to training mode, i.e. things like dropout layers are active, turn off by using self.eval() or model.eval()
        optimizer = torch.optim.Adam(self.parameters(), lr=training_config.get("learning_rate", 1e-3))
        num_epochs = training_config.get("epochs", 1000)

        # IMPORTANT: the device the model is on defines the device passed to the loss_class
        # Get device from model parameters since nn.Module doesn't have a .device attribute

        device = next(self.parameters()).device
        loss_function = TotalLoss(device, renderer_wrapper=renderer_wrapper)

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

            # save mesh every 5 epochs
            if general_config is not None and epoch % training_config.get("mesh_save_interval", 5) == 0:
                allround_save_data(
                    model=self,
                    storage_path=general_config["general_info"].get(
                        "storage_path", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/bulk_storage"
                    ),
                    save_model=False,
                    save_mesh=True,
                    verbose=False,
                    pretraining=False,
                    wandb_run=wandb_name,
                    epoch=epoch,
                )
