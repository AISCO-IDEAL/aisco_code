import torch
from torch import nn, optim

from subprojects.wp4_aeroshell_optimizer.v1.src.losses.total_loss import TotalLoss

if __name__ == "__main__":
    # Initialize SDF Network
    model = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1)).to("cuda")

    # Initialize Losses with only bounding box loss weight set
    lossm = TotalLoss()

    # Define optimizerpip
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = lossm.get_total_loss(model)
        print("loss.requires_grad:", loss.requires_grad)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Bounding Box Loss: {loss.item()}")
        print(f"Epoch {epoch}, Bounding Box Loss: {loss.item()}")
        print(f"Epoch {epoch}, Bounding Box Loss: {loss.item()}")
        print(f"Epoch {epoch}, Bounding Box Loss: {loss.item()}")
