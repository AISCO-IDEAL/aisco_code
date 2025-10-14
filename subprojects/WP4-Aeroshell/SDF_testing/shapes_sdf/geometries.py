#Signed Distance Functions (SDFs) for basic geometries and operations
import torch
import matplotlib.pyplot as plt
def get_points(batch_size=512):
    device="cpu"
    points = torch.rand(batch_size, 2) * 2 - 1
    points = points.to(device)
    return points

def sdf_circle(xy, center, radius):
    # xy: [N,2]
    return torch.linalg.norm(xy - center, dim=1) - radius

def sdf_box(xy, center, half_extents):
    # classic box SDF (axis-aligned)
    q = torch.abs(xy - center) - half_extents
    outside_dist = torch.linalg.norm(torch.clamp(q, min=0.0), dim=1)
    inside_dist  = torch.clamp(q.max(dim=1).values, max=0.0)
    return outside_dist + inside_dist

def sdf_union(d1, d2):
    return torch.minimum(d1, d2)

def sdf_intersection(d1, d2):
    return torch.maximum(d1, d2)

def sdf_subtraction(d1, d2):
    return torch.maximum(d1, -d2)


def sdf_union_GT(xy):
    d_circle = sdf_circle(xy, torch.tensor([0.1, 0.0]), 0.4)
    d_box    = sdf_box(xy,    torch.tensor([-0.3, 0.0]), torch.tensor([0.25,0.25]))
    return sdf_union(d_circle, d_box)   # union of circle and box
def sdf_intersection_GT(xy):
    d_circle = sdf_circle(xy, torch.tensor([-0.1, 0.0]), 0.4)
    d_box    = sdf_box(xy,    torch.tensor([-0.3, 0.0]), torch.tensor([0.25,0.25]))
    return sdf_intersection(d_circle, d_box)   # intersection of circle and box

def visualization(sdf_values, text="Network"):
    x = torch.linspace(-1.0, 1.0, 400)
    y = torch.linspace(-1.0, 1.0, 400)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    plt.figure(figsize=(8, 8))
    contour = plt.contourf(X.numpy(), Y.numpy(), sdf_values.detach().numpy(), levels=50, cmap='RdBu_r')
    plt.colorbar(contour)
    plt.contour(X.numpy(), Y.numpy(), sdf_values.detach().numpy(), levels=[0], colors='black')  # Zero level set
    plt.title(f'SDF of {text}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.show()

def visualize_gradients(model):

    x = torch.linspace(-1.0, 1.0, 400)
    y = torch.linspace(-1.0, 1.0, 400)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    points.requires_grad_(True)
    sdf_values = model(points)
    gradients = torch.autograd.grad(
        outputs=sdf_values,
        inputs=points,
        grad_outputs=torch.ones_like(sdf_values),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # Calculate gradient magnitudes and reshape to grid
    gradient_magnitudes = torch.norm(gradients, dim=1).reshape(X.shape)
    gradient_magnitudes = gradient_magnitudes.detach().numpy()
    
    plt.figure(figsize=(8, 8))
    contour = plt.contourf(X.numpy(), Y.numpy(), gradient_magnitudes, 
                          levels=50, cmap='viridis', vmin=0, vmax=2)
    plt.colorbar(contour, label='Gradient Magnitude')
    plt.title('Gradient Magnitude Field of the SDF')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.axis('equal')
    plt.show(block=False)