#Put the SDF training loop here


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
    
    Args:
        model: An instance of SDFNetwork to be trained.
        config: A dictionary containing training parameters such as learning rate, 
                number of epochs, batch size, and training data.
    """

    learning_rate = config.get("learning_rate", 1e-3)
    num_epochs = config.get("num_epochs", 1000)
    batch_size = config.get("batch_size", 1024)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = SDFLosses()

    for epoch in range(num_epochs):
        

        optimizer.zero_grad()
        


        loss = loss_fn.get_losses(model)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model
