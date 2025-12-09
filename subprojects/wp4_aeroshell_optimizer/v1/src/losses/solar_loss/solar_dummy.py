# Use aalthaus' solar dummy loss that is differentiable wrt the mesh vertices

import kaolin
import torch


def solar_mesh_deformation(raw_vertices: torch.Tensor, faces: torch.Tensor, sun_dir: torch.Tensor, dev: str="cuda",
                           slope_param: float = 7.0, offset_param: float = 3.0):


    """
    Adapted the above function to test backwards transfer to SDF

    Returns: new vertices after one gradient descent step to maximize solar score

    Is WIP...

    """

    if faces.numel() == 0:
        # skip or return a safe zero-loss
        print("Warning: Empty mesh received in dummy_solar_loss!")
        return torch.tensor(2.0, device=raw_vertices.device, requires_grad=True)


    vertices = raw_vertices.detach().clone()  # break graph
    vertices.requires_grad_(True)

    verts_b = vertices.unsqueeze(0)     # (1, V, 3)
    faces_b = faces.unsqueeze(0)

    # Face vertices
    v0 = verts_b[0][faces_b[:, 0]]
    v1 = verts_b[0][faces_b[:, 1]]
    v2 = verts_b[0][faces_b[:, 2]]

    # Compute face normals
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-12)

    # Normalize sun direction
    s = sun_dir / (torch.norm(sun_dir) + 1e-12)

    # Compute power ratio (dot product of normals and sun direction)
    score = (normals * s).sum(dim=-1)


    #Compute loss with custom activation function that flows well with gradients
    loss = - (torch.sigmoid(slope_param * torch.log(score + 1.5) - offset_param)).mean()

    loss.backward()
    learning_rate = 1e-2
    # Direct gradient descent step
    vertices_new = vertices - learning_rate * vertices.grad

    return vertices_new
