import torch

from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel


def loss_from_deformed_mesh(deformed_vertices: torch.Tensor, sdf_model: BaseModel):
    """
    Given deformed vertices, calculate a loss that will update the SDF model's parameters to better fit these vertices.
    This is a placeholder function and assumes the existence of an SDF model with a suitable interface.

    deformed_vertices: tensor [V, 3]
    sdf_model: an instance of an SDF model with a method to compute loss given target points

    Returns: loss value after update
    """
    # Compute loss between SDF model and deformed vertices
    output = sdf_model(deformed_vertices)
    sdf_values = output[0] if isinstance(output, tuple) else output

    loss = sdf_values.abs().mean()
    # Backpropagate and update SDF model parameters

    return loss
