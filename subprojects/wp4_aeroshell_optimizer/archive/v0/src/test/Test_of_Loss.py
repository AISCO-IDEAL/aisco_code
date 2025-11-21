# This code only thests the bounding box loss from losses.py
import cadquery as cq
from ..utils.CadQuery_to_SDF import SDFfromCadQuery
from ..utils.losses import SDFLosses
import torch

def test_bounding_box_loss():
    loss_function = SDFLosses(weights=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dev="cpu")
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
    )
    # create a simple training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(10):
        optimizer.zero_grad()
        loss = loss_function.bounding_box_loss_2(model)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Bounding Box Loss: {loss.item()}")

if __name__ == "__main__":
    test_bounding_box_loss()