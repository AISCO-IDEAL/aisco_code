import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid

# --- Configuration ---
device = torch.device("cpu")
learning_rate = 1e-3
epochs = 2000  # Might need more epochs for accuracy
nu = 0.01  # Kinematic viscosity
rho = 1.0  # Density
grid_size = 20  # 20x20 grid
H = 1.0  # Total channel height (plates at y=0 and y=H)
# We'll map y from 0 to H in the grid to -h to +h in the formula
h_formula = H / 2.0
mu = nu * rho  # Dynamic viscosity

# --- Analytical Solution Parameters ---
# We need to *impose* a pressure gradient or velocity boundary conditions
# Let's impose a pressure drop: p(x=0)=1, p(x=L)=0 for L=1
# dp/dx = (p_out - p_in) / L = (0 - 1) / 1 = -1.0
dpdx = -1.0
p0 = 1.0  # Pressure at x=0


# --- 1. Analytical Solution Function ---
def poiseuille_analytical(x_coord, y_coord, H_channel, dpdx_val, mu_val, p0_val):
    """Calculates the analytical solution for Poiseuille flow."""
    # Map y from [0, H] grid coordinates to [-h, h] formula coordinates
    h_form = H_channel / 2.0
    y_formula = y_coord - h_form

    u = (-1.0 / (2.0 * mu_val)) * dpdx_val * (h_form**2 - y_formula**2)
    v = np.zeros_like(u)
    p = dpdx_val * x_coord + p0_val
    return u, v, p


# --- 2. Create Grid Graph Data ---
print("Creating grid graph data...")
x_coords_tensor = torch.linspace(0, 1, grid_size, device=device)
y_coords_tensor = torch.linspace(0, H, grid_size, device=device)  # y goes from 0 to H
xx, yy = torch.meshgrid(x_coords_tensor, y_coords_tensor, indexing="xy")
pos = torch.stack([xx.flatten(), yy.flatten()], dim=1).requires_grad_(True)

edge_index = grid(height=grid_size, width=grid_size, device=device)[0]
node_features = pos  # Use positions directly as input features

data = Data(x=node_features, edge_index=edge_index, pos=pos)

# Identify boundary nodes for the channel (y=0 and y=H are walls, x=0 is inlet, x=1 is outlet)
data.wall_mask = (pos[:, 1] == 0) | (pos[:, 1] == H)
data.inlet_mask = (pos[:, 0] == 0) & ~data.wall_mask  # Inlet nodes (excluding corners)
data.outlet_mask = (
    pos[:, 0] == 1
) & ~data.wall_mask  # Outlet nodes (excluding corners)
data.interior_mask = ~data.wall_mask & ~data.inlet_mask & ~data.outlet_mask
boundary_mask = data.wall_mask | data.inlet_mask | data.outlet_mask  # For checking

print(f"Created graph with {data.num_nodes} nodes and {data.num_edges} edges.")
print(f"Interior nodes: {data.interior_mask.sum()}")
print(f"Wall nodes: {data.wall_mask.sum()}")
print(f"Inlet nodes: {data.inlet_mask.sum()}")
print(f"Outlet nodes: {data.outlet_mask.sum()}")
print(f"Total boundary nodes: {boundary_mask.sum()}")


# --- 3. Define the GNN Model ---
class SimpleGNN(nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, output_features)

    def forward(self, data):
        x_features, edge_index = data.x, data.edge_index
        x_features = torch.relu(self.conv1(x_features, edge_index))
        x_features = self.conv2(x_features, edge_index)  # Output: [u, v, p]
        return x_features


# --- 4. Physics-Informed Loss Calculation ---
# (navier_stokes_residuals function remains the same as before)
def navier_stokes_residuals(outputs, pos, nu, rho):
    """Calculates 2D Navier-Stokes residuals using automatic differentiation."""
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    p = outputs[:, 2:3]
    pos.requires_grad_(True)

    grad_outputs = torch.ones_like(u)
    grad_u = torch.autograd.grad(
        u, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    grad_v = torch.autograd.grad(
        v, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    grad_p = torch.autograd.grad(
        p, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]

    du_dx = grad_u[:, 0:1] if grad_u is not None else torch.zeros_like(u)
    du_dy = grad_u[:, 1:2] if grad_u is not None else torch.zeros_like(u)
    dv_dx = grad_v[:, 0:1] if grad_v is not None else torch.zeros_like(v)
    dv_dy = grad_v[:, 1:2] if grad_v is not None else torch.zeros_like(v)
    dp_dx = grad_p[:, 0:1] if grad_p is not None else torch.zeros_like(p)
    dp_dy = grad_p[:, 1:2] if grad_p is not None else torch.zeros_like(p)

    du_dxx = torch.autograd.grad(
        du_dx, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    du_dyy = torch.autograd.grad(
        du_dy, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    dv_dxx = torch.autograd.grad(
        dv_dx, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]
    dv_dyy = torch.autograd.grad(
        dv_dy, pos, grad_outputs=grad_outputs, create_graph=True, allow_unused=True
    )[0]

    laplacian_u = (du_dxx[:, 0:1] if du_dxx is not None else 0) + (
        du_dyy[:, 1:2] if du_dyy is not None else 0
    )
    laplacian_v = (dv_dxx[:, 0:1] if dv_dxx is not None else 0) + (
        dv_dyy[:, 1:2] if dv_dyy is not None else 0
    )

    continuity = du_dx + dv_dy
    momentum_x = u * du_dx + v * du_dy + (1 / rho) * dp_dx - nu * laplacian_u
    momentum_y = u * dv_dx + v * dv_dy + (1 / rho) * dp_dy - nu * laplacian_v

    return continuity, momentum_x, momentum_y


def physics_informed_loss(data, outputs, nu, rho, p0_val, dpdx_val):
    """Combines Poiseuille boundary loss and PDE residual loss."""
    loss_bcs = torch.tensor(0.0, device=device)
    loss_pde = torch.tensor(0.0, device=device)

    # --- Boundary Condition Loss (MSE) ---
    # Walls (u=0, v=0 at y=0 and y=H)
    if data.wall_mask.any():
        wall_preds = outputs[data.wall_mask]
        target_wall_uv = torch.zeros_like(wall_preds[:, 0:2])
        loss_bcs += nn.functional.mse_loss(wall_preds[:, 0:2], target_wall_uv)

    # Inlet Pressure (p = p0 at x=0)
    if data.inlet_mask.any():
        inlet_preds = outputs[data.inlet_mask]
        target_inlet_p = torch.full_like(inlet_preds[:, 2:3], p0_val)
        loss_bcs += nn.functional.mse_loss(inlet_preds[:, 2:3], target_inlet_p)

    # Outlet Pressure (p = p0 + dpdx * L at x=1, L=1)
    if data.outlet_mask.any():
        outlet_preds = outputs[data.outlet_mask]
        target_outlet_p = torch.full_like(outlet_preds[:, 2:3], p0_val + dpdx_val * 1.0)
        loss_bcs += nn.functional.mse_loss(outlet_preds[:, 2:3], target_outlet_p)

    # We don't enforce inlet/outlet velocity, let the pressure BCs drive it.

    # --- PDE Residual Loss (MSE) ---
    if data.interior_mask.any():
        cont_res, xmom_res, ymom_res = navier_stokes_residuals(
            outputs, data.pos, nu, rho
        )
        zeros_target = torch.zeros_like(cont_res[data.interior_mask])

        loss_pde = (
            nn.functional.mse_loss(cont_res[data.interior_mask], zeros_target)
            + nn.functional.mse_loss(xmom_res[data.interior_mask], zeros_target)
            + nn.functional.mse_loss(ymom_res[data.interior_mask], zeros_target)
        )

    # --- Combine Losses ---
    total_loss = loss_bcs + loss_pde
    return total_loss, loss_bcs, loss_pde


# --- 5. Training Setup ---
model = SimpleGNN(input_features=2, hidden_features=64, output_features=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("\nStarting Training...")
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    all_outputs = model(data)
    loss, loss_bc, loss_pde = physics_informed_loss(
        data, all_outputs, nu, rho, p0, dpdx
    )
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0 or epoch == epochs - 1:
        bc_val = loss_bc.item() if isinstance(loss_bc, torch.Tensor) else loss_bc
        pde_val = loss_pde.item() if isinstance(loss_pde, torch.Tensor) else loss_pde
        print(
            f"Epoch {epoch}: Total Loss: {loss.item():.4e}, BC Loss: {bc_val:.4e}, PDE Loss: {pde_val:.4e}"
        )

print("Training Finished.")

# --- 6. Inference & Comparison with Analytical Solution ---
model.eval()
with torch.no_grad():
    final_predictions = model(data).cpu().numpy()

# Get node coordinates
pos_np = data.pos.detach().cpu().numpy()
x_np, y_np = pos_np[:, 0], pos_np[:, 1]

# Calculate analytical solution at node coordinates
u_analytical, v_analytical, p_analytical = poiseuille_analytical(
    x_np, y_np, H, dpdx, mu, p0
)

# Extract GNN predictions
u_pred, v_pred, p_pred = (
    final_predictions[:, 0],
    final_predictions[:, 1],
    final_predictions[:, 2],
)

# --- Calculate Errors ---
mae_u = np.mean(np.abs(u_pred - u_analytical))
mae_v = np.mean(np.abs(v_pred - v_analytical))
mae_p = np.mean(np.abs(p_pred - p_analytical))

print("\n--- Comparison ---")
print(f"Mean Absolute Error (u): {mae_u:.4e}")
print(f"Mean Absolute Error (v): {mae_v:.4e}")  # Should be very small
print(f"Mean Absolute Error (p): {mae_p:.4e}")

# --- Visualize Results and Comparison ---
xx_plot = x_np.reshape(grid_size, grid_size)
yy_plot = y_np.reshape(grid_size, grid_size)
u_pred_plot = u_pred.reshape(grid_size, grid_size)
v_pred_plot = v_pred.reshape(grid_size, grid_size)
p_pred_plot = p_pred.reshape(grid_size, grid_size)
u_ana_plot = u_analytical.reshape(grid_size, grid_size)
p_ana_plot = p_analytical.reshape(grid_size, grid_size)

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
levels = 50  # Number of contour levels

# Predicted U
cf = axes[0, 0].contourf(xx_plot, yy_plot, u_pred_plot, cmap="viridis", levels=levels)
fig.colorbar(cf, ax=axes[0, 0])
axes[0, 0].set_title("Predicted u")
axes[0, 0].set_ylabel("y")

# Predicted V
cf = axes[0, 1].contourf(xx_plot, yy_plot, v_pred_plot, cmap="viridis", levels=levels)
fig.colorbar(cf, ax=axes[0, 1])
axes[0, 1].set_title("Predicted v")

# Predicted P
cf = axes[0, 2].contourf(xx_plot, yy_plot, p_pred_plot, cmap="viridis", levels=levels)
fig.colorbar(cf, ax=axes[0, 2])
axes[0, 2].set_title("Predicted p")

# Analytical U
cf = axes[1, 0].contourf(xx_plot, yy_plot, u_ana_plot, cmap="viridis", levels=levels)
fig.colorbar(cf, ax=axes[1, 0])
axes[1, 0].set_title("Analytical u")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")

# Analytical V (should be zero)
axes[1, 1].contourf(
    xx_plot, yy_plot, np.zeros_like(v_pred_plot), cmap="viridis", levels=levels
)
axes[1, 1].set_title("Analytical v (0)")
axes[1, 1].set_xlabel("x")

# Analytical P
cf = axes[1, 2].contourf(xx_plot, yy_plot, p_ana_plot, cmap="viridis", levels=levels)
fig.colorbar(cf, ax=axes[1, 2])
axes[1, 2].set_title("Analytical p")
axes[1, 2].set_xlabel("x")


plt.suptitle("GNN Prediction vs Analytical Solution (Poiseuille Flow)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for title
plt.show()
