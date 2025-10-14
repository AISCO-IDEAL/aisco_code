#training loop for SDF network
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from sdf import SDFNet
from geometries import sdf_circle, sdf_box, sdf_union_GT, get_points

def train(sdf_func, epochs=1000, batch_size=512, lr=1e-3, device='cpu'):

    model = SDFNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # Sample random points in [-1,1]x[-1,1]
    points= get_points()
    #Here we have GT Values
    with torch.no_grad():
        sdf_gt = sdf_func(points).to(device)
        #Training loop
    for epoch in range(epochs):
        

       
        

        # Predict SDF values
        sdf_pred = model(points)

        # Compute loss
        loss = loss_fn(sdf_pred, sdf_gt)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

    return model

