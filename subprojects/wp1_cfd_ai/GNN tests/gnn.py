import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from torch import nn, optim
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
# --- (Previous script code up to the end of training and inference) ---

# --- 6. Inference & Comparison ---
model.eval()
with torch.no_grad():
    final_predictions = model(data).cpu().numpy()
pos_np = data.pos.detach().cpu().numpy()
x_np, y_np = pos_np[:, 0], pos_np[:, 1]
u_pred, v_pred, p_pred = (
    final_predictions[:, 0],
    final_predictions[:, 1],
    final_predictions[:, 2],
)

# --- Visualize Results with Velocity Vectors ---
print("Visualizing results using Matplotlib...")

xx_plot = x_np.reshape(grid_size, grid_size)
yy_plot = y_np.reshape(grid_size, grid_size)
u_pred_plot = u_pred.reshape(grid_size, grid_size)
v_pred_plot = v_pred.reshape(grid_size, grid_size)
p_pred_plot = p_pred.reshape(grid_size, grid_size)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

# Plot Pressure Contours as background
cf = ax.contourf(xx_plot, yy_plot, p_pred_plot, cmap="viridis", levels=50, alpha=0.7)
fig.colorbar(cf, label="Pressure (p)")

# Plot Velocity Vectors (Quiver Plot)
# Use a subset of points for quiver to avoid clutter (e.g., every 2nd point)
skip = 2
ax.quiver(
    xx_plot[::skip, ::skip],
    yy_plot[::skip, ::skip],
    u_pred_plot[::skip, ::skip],
    v_pred_plot[::skip, ::skip],
    color="red",
    scale=20,  # Adjust scale for vector length
    alpha=0.8,
)

# Draw the "Object" (Channel Walls)
ax.plot([0, 1], [0, 0], color="black", linewidth=3)  # Bottom wall
ax.plot([0, 1], [H, H], color="black", linewidth=3)  # Top wall (Lid)
ax.plot([0, 0], [0, H], color="gray", linewidth=2)  # Inlet side (optional)
ax.plot([1, 1], [0, H], color="gray", linewidth=2)  # Outlet side (optional)


ax.set_title("Predicted Pressure Contours and Velocity Vectors")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal", adjustable="box")  # Make aspect ratio realistic
plt.tight_layout()
plt.show()

# --- 7. Visualize Results using PyVista ---
try:
    import pyvista as pv

    print("\nVisualizing results using PyVista...")

    # ... (Create grid_pv as before) ...
    points_pv = np.hstack((pos_np, np.zeros((pos_np.shape[0], 1))))  # Add z=0
    grid_pv = pv.StructuredGrid(xx_plot, yy_plot, np.zeros_like(xx_plot))
    grid_pv["u"] = u_pred_plot.flatten(order="F")
    grid_pv["v"] = v_pred_plot.flatten(order="F")
    grid_pv["p"] = p_pred_plot.flatten(order="F")
    grid_pv["velocity"] = np.vstack(
        (grid_pv["u"], grid_pv["v"], np.zeros(grid_pv.n_points))
    ).T

    # --- Create PyVista Plotter ---
    plotter = pv.Plotter(window_size=[800, 600])

    # --- Add the Grid (Object/Domain Representation), colored by pressure ---
    plotter.add_mesh(
        grid_pv, scalars="p", cmap="viridis", scalar_bar_args={"title": "Pressure"}
    )

    # --- Add Glyphs (Arrows) for Velocity ---
    subset_indices = np.arange(grid_pv.n_points)
    np.random.shuffle(subset_indices)
    subset_indices = subset_indices[: min(500, grid_pv.n_points // 5)]

    points_subset = grid_pv.points[subset_indices]
    vectors_subset = grid_pv["velocity"][subset_indices]

    if points_subset.size > 0 and vectors_subset.size > 0:
        # 1. Create a PolyData object for the subset points
        subset_points_pv = pv.PolyData(points_subset)
        # 2. Add the vectors
        subset_points_pv["velocity"] = vectors_subset
        # --- CHANGE HERE ---
        # 3. Calculate magnitudes and add them as SCALARS *before* glyphing
        subset_points_pv["magnitude"] = np.linalg.norm(vectors_subset, axis=1)
        # --- END CHANGE ---

        # 4. Create glyphs using the .glyph() method
        #    It will automatically use the active vectors ('velocity') for orientation
        #    and the active scalars ('magnitude') for coloring.
        subset_points_pv.active_vectors_name = (
            "velocity"  # Explicitly set active vectors
        )
        # Set magnitude as the active scalars for coloring glyphs
        subset_points_pv.active_scalars_name = "magnitude"

        # Adjust factor for glyph size
        max_mag = subset_points_pv["magnitude"].max()
        scale_factor = (
            (grid_pv.bounds[1] - grid_pv.bounds[0]) * 0.05 / max_mag
            if max_mag > 1e-9
            else 0.05
        )
        scale_factor = min(scale_factor, 1.0)

        # Create the glyphs, they will inherit the 'magnitude' scalar data
        glyphs = subset_points_pv.glyph(
            orient="velocity", scale="magnitude", factor=scale_factor
        )

        # REMOVE THIS LINE - scalars are now part of 'glyphs' automatically
        # glyphs["scalars"] = np.linalg.norm(vectors_subset, axis=1)

        # Add the glyphs mesh to the plotter, coloring by the inherited 'magnitude'
        plotter.add_mesh(
            glyphs,
            scalars="magnitude",
            cmap="coolwarm",
            scalar_bar_args={"title": "Velocity Mag."},
        )
    else:
        print("Warning: No points or vectors found for glyph visualization.")

    # --- Add Object Boundaries (Lines) ---
    bottom_wall = pv.Line([0, 0, 0], [1, 0, 0])
    top_wall = pv.Line([0, H, 0], [1, H, 0])
    inlet_wall = pv.Line([0, 0, 0], [0, H, 0])
    outlet_wall = pv.Line([1, 0, 0], [1, H, 0])
    plotter.add_mesh(bottom_wall, color="black", line_width=5)
    plotter.add_mesh(top_wall, color="black", line_width=5)
    plotter.add_mesh(inlet_wall, color="grey", line_width=3)
    plotter.add_mesh(outlet_wall, color="grey", line_width=3)

    # --- Customize and Show Plot ---
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.add_text(
        "CFD Prediction (Pressure Contours, Velocity Glyphs)",
        position="upper_edge",
        font_size=10,
    )
    plotter.show()

except ImportError:
    print("\nPyVista not installed. Skipping PyVista visualization.")
except Exception as e:
    print(f"\nError during PyVista visualization: {e}")
