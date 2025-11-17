import torch

def soft_knapsack_selection(scores_per_area, areas, area_budget, tau=0.1, penalty=10.0):
    """
    scores_per_area: tensor [N], already score/area, i.e. powerratio --> normal vector *dot* sun vector
    areas: tensor [N]
    area_budget: float
    tau: softmax temperature (lower -> closer to discrete selection)
    penalty: strength of soft constraint

    Returns: (selected_score, selected_area, weights)
    """
    # Softmax over scores_per_area gives a differentiable "soft sort"
    w = torch.softmax(scores_per_area / tau, dim=0)
    weights = w / (w.sum() + 1e-8)
    

    # Expected area usage and score
    expected_area = (weights * areas).sum()
    expected_score = (weights * scores_per_area * areas).sum()
    # Note: score_per_area * area = actual score contribution

    # Soft area budget penalty
    constraint_violation = torch.relu(expected_area - area_budget)
    objective = expected_score - penalty * (constraint_violation ** 2)

    # We return -objective as the "loss" if the caller wants to optimize it.
    loss = -objective

    return loss, expected_score, expected_area, weights


# ---- Example usage ----
"""

N = 8
scores_per_area = torch.tensor([5., 3., 8., 1., 7., 4., 6., 2.], requires_grad=True)
areas = torch.tensor([4., 2., 6., 3., 5., 4., 7., 3.])
area_budget = 10.0

loss, score, used_area, w = soft_knapsack_selection(scores_per_area, areas, area_budget)

loss.backward()  # gradients flow through scores_per_area or upstream network

print("Expected score:", score.item())
print("Expected area:", used_area.item())
print("Selection weights:", w)

"""

def dummy_solar_loss(verts: torch.Tensor, faces: torch.Tensor, sun_dir: torch.Tensor, dev: str="cuda",
                     slope_param: float = 8.0, offset_param: float = 3.0):
    """
    A dummy loss function, written by Alex in order to have at least something working properly 
    for system integration at 18.11/25



    Calculates the normal vectors, takes the scalar product of normals and sun_dir and applies some activations
    For visual of the custom activation function see https://drive.google.com/file/d/1UEoSXcBtUWlcOHO_iL99V6blMAmkrygP/view?usp=drive_link
    """
    verts_b = verts.unsqueeze(0)     # (1, V, 3)
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
    loss = (torch.sigmoid(slope_param * torch.log(score + 1.01) - offset_param)).mean()
    return -loss