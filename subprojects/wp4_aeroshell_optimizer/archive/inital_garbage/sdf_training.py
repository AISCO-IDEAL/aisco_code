import numpy as np
import torch
from sdf import SDFNetwork


def random_points(size):
    points = np.zeros((size, 2))
    for i in range(size):
        points[i][0] = np.random.normal(0, 1.5)
        points[i][1] = np.random.normal(0, 1.5)

    points = torch.tensor(points, dtype=torch.float32)
    return points


def grid_points(size):
    x = np.linspace(-1.5, 1.5, int(np.sqrt(size)))
    y = np.linspace(-1.5, 1.5, int(np.sqrt(size)))
    noise_scale = 0.05  # adjust this value to control noise amount
    x = x + np.random.normal(0, noise_scale, x.shape)
    y = y + np.random.normal(0, noise_scale, y.shape)
    xx, yy = np.meshgrid(x, y)
    points = np.stack([xx.flatten(), yy.flatten()], axis=1)
    return torch.tensor(points, dtype=torch.float32)


def circle_loss(radius, input, points, random=True, size=60):
    # biases the gradient<=1 condition, for testing
    if random:
        points_pow2 = torch.sum(points**2, dim=1)
        target = torch.sqrt(points_pow2) - radius
        loss = torch.mean(torch.abs(target - input))
        return loss
    else:
        phis = np.linspace(0, 2 * np.pi, size)
        points = np.zeros((size, 2))
        for i in range(size):
            points[i][0] = radius * np.cos(phis[i])
            points[i][1] = radius * np.sin(phis[i])
        points = torch.tensor(points, dtype=torch.float32)

        loss = torch.tensor(np.mean(np.abs(input - radius)), dtype=torch.float32)
        return loss


def eikonel_loss(model, points):
    points.requires_grad_(True)
    sdf_values = model(points)
    gradients = torch.autograd.grad(sdf_values.sum(), points, create_graph=True)[0]
    gradient_norms = torch.norm(gradients, dim=1)
    loss = torch.mean((gradient_norms - 1.0) ** 2)
    return loss


def main_loop():
    epochs = 2500

    learning_rate = 2e-3
    radius = 1.0  # Example radius

    # Initialize model and optimizer
    model = SDFNetwork()
    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=learning_rate, lr_decay=1e-3, weight_decay=1e-4
    )

    for epoch in range(epochs):
        # Dummy input data (replace with real data)
        # Clear gradients before forward pass
        optimizer.zero_grad()

        # Generate and convert points to tensor
        points = random_points(50)

        # Forward pass - process all points at once
        outputs = model(points)

        # Compute combined loss (circle shape + eikonal term)
        circle_term = circle_loss(radius, outputs, points)
        eikonal_term = eikonel_loss(model, points)
        loss = circle_term + 0.05 * eikonal_term  # weight factor for eikonal term

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Circle Loss: {circle_term}, Eikonal Loss: {eikonal_term}, Total Loss: {loss.item():.4f}"
            )
            model.validate(radius)
    model.visualize_sdf(grid_size=100)
    model.visualize_gradients(grid_size=100)


main_loop()
