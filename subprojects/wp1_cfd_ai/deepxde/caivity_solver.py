import matplotlib
matplotlib.use('Agg')  # <-- ADD THIS LINE
import matplotlib.pyplot as plt
import numpy as np
import torch

import deepxde as dde

# --- 1. Set the Backend ---
dde.backend.set_default_backend("pytorch")


# --- 2. Define the Physics (PDE) ---
nu = 0.01  # Reynolds number = 100


def pde(x, y):
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]

    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_y = dde.grad.jacobian(y, x, i=0, j=1)
    dv_x = dde.grad.jacobian(y, x, i=1, j=0)
    dv_y = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)

    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dv_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dv_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = du_x + dv_y
    x_momentum = u * du_x + v * du_y + dp_x - nu * (du_xx + du_yy)
    y_momentum = u * dv_x + v * dv_y + dp_y - nu * (dv_xx + dv_yy)

    return [continuity, x_momentum, y_momentum]


# --- 3. Define the Geometry & Boundary Conditions ---

geom = dde.geometry.Rectangle([0, 0], [1, 1])


# --- Boundary checking functions ---
def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)


def boundary_other(x, on_boundary):
    return on_boundary and not dde.utils.isclose(x[1], 1.0)


# --- Boundary condition definitions ---

# Lid: u=1, v=0
bc_top_u = dde.DirichletBC(geom, lambda x: 1.0, boundary_top, component=0)
bc_top_v = dde.DirichletBC(geom, lambda x: 0.0, boundary_top, component=1)
# Other walls: u=0, v=0
bc_other_u = dde.DirichletBC(geom, lambda x: 0.0, boundary_other, component=0)
bc_other_v = dde.DirichletBC(geom, lambda x: 0.0, boundary_other, component=1)


# -------------------
# --- CODE CHANGE ---
# -------------------
# We no longer need the 'boundary_pressure_pin' function.
# Define the pressure "pin" point explicitly as a 2D array
pressure_pin_point = np.array([[0.0, 0.0]])

# Use PointSetBC to apply the condition to that *exact* point
# NEW, CORRECT line
# We provide the target value 0.0 as an array [0.0]
bc_pressure = dde.PointSetBC(pressure_pin_point, np.array([0.0]), component=2)
# -----------------------
# --- END CODE CHANGE ---
# -----------------------


# Group all boundary conditions
bcs = [bc_top_u, bc_top_v, bc_other_u, bc_other_v, bc_pressure]


# --- 4. Define the AI Model and Data ---

data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=2500,
    num_boundary=100,  # This 100 now only applies to the 4 DirichletBCs
    solution=None,
    num_test=None,
)

net = dde.maps.FNN([2] + [20] * 8 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)


# --- 5. Train the Model ---

print("Starting Adam optimization...")
model.compile("adam", lr=1e-3)
# Note: I'm lowering iterations to 10,000 for a faster (but less accurate) result.
# Feel free to change it back to 20,000.
losshistory, train_state = model.train(iterations=4000, display_every=1000)

print("Starting L-BFGS optimization...")
model.compile("L-BFGS")
losshistory, train_state = model.train()
print("Training complete.")

# --- 6. Plot the Results ---
print("Generating plot...")

resolution = 100
x_plot = np.linspace(0, 1, resolution)
y_plot = np.linspace(0, 1, resolution)
X, Y = np.meshgrid(x_plot, y_plot)
XY_pairs = np.vstack((X.ravel(), Y.ravel())).T

solution = model.predict(XY_pairs)
u = solution[:, 0].reshape(resolution, resolution)
v = solution[:, 1].reshape(resolution, resolution)
p = solution[:, 2].reshape(resolution, resolution)

# Plot Velocity
plt.figure(figsize=(10, 8))
# Use 'color=u' instead of 'C=u'
plt.quiver(X, Y, u, v, color=u) # <-- FIX HERE
plt.colorbar(label="u-velocity (m/s)")
plt.title(f"2D Lid-Driven Cavity Flow (Re={1 / nu}) - Velocity")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.axis("scaled")
plt.savefig("cavity_velocity.png")
print("Saved velocity plot to cavity_velocity.png")

# Plot Pressure
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, p, levels=50, cmap="viridis")
plt.colorbar(label="Pressure (Pa)")
plt.title(f"2D Lid-Driven Cavity Flow (Re={1 / nu}) - Pressure")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.axis("scaled")
plt.savefig("cavity_pressure.png")
print("Saved pressure plot to cavity_pressure.png")
