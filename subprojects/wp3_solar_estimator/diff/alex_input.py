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