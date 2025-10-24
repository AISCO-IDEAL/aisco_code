from sdf import SDFNetwork
import torch
import torch.nn as nn
import typing as T
from losses import SDFLosses

basic_config = {
    "learning_rate": 1e-3,
    "num_epochs": 1000,
    "batch_size": 1024,
    # Add other training parameters as needed, don't forget to update training loop accordingly
}

def train_sdf_model(model: SDFNetwork, config: T.Dict) -> SDFNetwork:
    """
    Train the given SDFNetwork model based on the provided configuration.

    Pre-conditions:
        - model: An instance of SDFNetwork, already initialized and ready for training.
        - config: A dictionary containing at least the following keys:
            - learning_rate: Positive float (default: 1e-3).
            - num_epochs: Positive integer (default: 1000).
            - batch_size: Positive integer (default: 1024).
            - (Optional) Any additional parameters required by SDFLosses or the training process.

    Post-conditions:
        - The model's parameters are updated via gradient descent for num_epochs iterations.
        - The loss is printed every 100 epochs.
        - The trained model is returned.
        - No side effects on the input config or model other than parameter updates.
    """
    # Pre-condition check: Ensure required config keys exist and are valid
    learning_rate = config.get("learning_rate", 1e-3)
    num_epochs = config.get("num_epochs", 1000)
    batch_size = config.get("batch_size", 1024)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SDFLosses(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Adjust weights as needed

    for epoch in range(num_epochs):
        # Pre-condition for each epoch:
        # - model.parameters() must be in a state ready for gradient computation
        # - optimizer must be initialized and linked to model.parameters()
        # - loss_fn must be ready to compute loss given the model

        optimizer.zero_grad()

        # Pre-condition for loss computation:
        # - model must be in eval() or train() mode as required by SDFLosses
        # - SDFLosses must be able to handle the current model and any required data
        loss = loss_fn.get_total_loss(model)

        # Pre-condition for backward pass:
        # - loss must be a scalar tensor with requires_grad=True
        loss.backward()

        # Pre-condition for step:
        # - gradients must be computed and ready for update
        optimizer.step()

        # Post-condition for each epoch:
        # - model.parameters() are updated once
        # - gradients are zeroed for the next iteration
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Post-condition for the function:
    # - model is trained for num_epochs iterations
    # - model is returned in the same type as input, but with updated parameters
    return model
