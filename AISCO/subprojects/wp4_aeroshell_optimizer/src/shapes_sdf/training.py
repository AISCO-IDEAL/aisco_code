#training loop for SDF network
import torch
from geometries import get_points
from sdf import SDFNet
from torch import optim
from utils import plot_loss_curve


def curvature_loss(model, points):
    """Compute curvature loss to encourage smooth SDFs by penalizing high second derivatives."""
    #Still buggy, doesn't get minimized, stays = 0
    points.requires_grad_(True)
    sdf_pred = model(points)
    # First compute gradients (first derivatives)
    gradients = torch.autograd.grad(
        outputs=sdf_pred,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Now compute second derivatives for each dimension
    curvature_sum = 0
    for i in range(points.shape[1]):  # For each dimension (x, y)
        second_deriv = torch.autograd.grad(
            outputs=gradients[:, i],
            inputs=points,
            grad_outputs=torch.ones_like(gradients[:, i]),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        # Add the second derivative magnitude to our curvature measure
        curvature_sum += torch.norm(second_deriv, dim=1)**2
    # Return mean curvature across all points
    return curvature_sum.mean()


def eikonel_loss(model, points):
    # Eikonal loss to enforce |∇f(x)| = 1
    points.requires_grad_(True)
    sdf_pred = model(points)
    gradients = torch.autograd.grad(
        outputs=sdf_pred,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_norm = torch.linalg.norm(gradients, dim=-1)
    eikonal_loss = ((grad_norm - 1)**2).mean()
    return eikonal_loss


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
    losses = []*2
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Predict SDF values
        sdf_pred = model(points)

        # Compute loss terms
        mse_term = loss_fn(sdf_pred, sdf_gt)
        eikonel_term = eikonel_loss(model, points)
        #curvature_term = curvature_loss(model, points)
        # Loss weights
        eikonel_factor = 0.2
        curvature_factor = 1.05  # Start with a small weight for curvature
        # Combined loss
        loss = mse_term + eikonel_factor * eikonel_term #+ curvature_factor * curvature_term

        # Backpropagation
        loss.backward()
        optimizer.step()
        # Make sure points don't accumulate gradients
        points.grad = None
        losses.append([loss.item(), mse_term.item(), eikonel_factor*eikonel_term.item()]) #, curvature_factor*curvature_term.item()])
        if (epoch+1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, '
                  f'MSE: {mse_term:.6f}, '
                  f'Eikonal: {eikonel_factor*eikonel_term:.6f}, ')
                  #f'Curvature: {curvature_factor*curvature_term:.6f}')
    plot_loss_curve(losses, title="Training Loss Curve")
    return model
