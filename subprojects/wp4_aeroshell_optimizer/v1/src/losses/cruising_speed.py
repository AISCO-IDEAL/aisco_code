import torch


class PhysicalRootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, drag_force_at_25ms, solar_power, rolling_friction_force, c, max_iter, tol):
        # We solve: f(u) = drag_force_normalized * u^3 + rolling_friction_force * u + (c - solar_power) = 0
        with torch.no_grad():
            # Initialization: Since drag and solar_power are > 0,
            # A more adaptive starting guess
            u_0 = (solar_power - c) / (rolling_friction_force + 1e-6) # positive power increases speed.
            u = torch.clamp(u_0, min=1.0, max=60.0)  # Keep it in a sane range

            # Newton's Method (5-10 iterations are typically sufficient given the monotonicity)
            for _ in range(max_iter):
                f = drag_force_at_25ms * u**3 / (25**2) + rolling_friction_force * u + c - solar_power
                df_du = 3 / (25**2) * drag_force_at_25ms * u**2 + rolling_friction_force
                delta_u = f / (df_du + 1e-9)  # Avoid division by zero
                u = u - delta_u

                # Ensure we remain within the physical domain (speed must be positive)
                u = torch.clamp(u, min=1e-6)  # TODO: is clamping feasible?
                if torch.max(torch.abs(delta_u)) < tol:
                    break

        ctx.save_for_backward(u, drag_force_at_25ms, solar_power)
        ctx.rolling_friction_force = rolling_friction_force
        ctx.c = c
        return u

    @staticmethod
    def backward(ctx, dloss_du):
        u, drag_force_at_25ms, solar_power = ctx.saved_tensors
        rolling_friction_force = ctx.rolling_friction_force

        # Partial derivatives of the function F = u^2 / (25m/s)^2 * drag_force_at_25ms*u + rolling_friction_force*u + c - solar_power
        df_du = 3 / (25**2) * drag_force_at_25ms * u**2 + rolling_friction_force
        df_ddrag_force_at_25ms = u**3 / (25**2)
        df_dsolar_power = -torch.ones_like(solar_power)

        # Calculate implicit derivative
        # grad_output is dLoss/du
        # We need dLoss/ddrag_force_at_25ms = dLoss/du * du/ddrag_force_at_25ms
        # using du/ddrag_force_at_25ms = - (df/ddrag_force_at_25ms) / (df/du)

        i_f_t = -dloss_du / df_du  # implicit function theorem term: ∂L/∂u​ = -∂f/∂u * ∂f/∂x

        dloss_ddrag_force_at_25ms = i_f_t * df_ddrag_force_at_25ms
        dloss_dsolar_power = i_f_t * df_dsolar_power

        # rolling_friction_force and c are constants relative to this solver -> None
        return dloss_ddrag_force_at_25ms, dloss_dsolar_power, None, None, None, None


def cruising_speed(
    solar_power: torch.Tensor,
    drag_force_at_25ms: torch.Tensor,
    mass: float,
    c_losses: float,
    rolling_friction_coefficient: float = 0.006,
    max_iter: int = 12,
    tol: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
    Combines solar power prediction and drag force prediction into
    a single physical objective: the estimated cruising speed.

    Args:
        solar_power: Solar power prediction  from Solar Estimator [W]
        drag_force_at_25ms: drag force at 25 m/s from CFD AI [N]
        mass: Vehicle mass [kg]
        c_losses: Constant power losses (electronics, etc.) [W]
        rolling_friction_coefficient: Friction coefficient for the tires [-]

    Returns:
        (cruising_speed_objective, diagnostics):
            cruising_speed_objective: Estimated speed in [m/s]
            diagnostics: dict of intermediate values for debugging
    """
    # Rolling friction coefficient based on ACE experience, TODO: refine and make speed-dependent if needed
    # Power loss due to rolling friction
    rolling_friction_force = mass * 9.81 * rolling_friction_coefficient

    # Bias for solar charging (e.g., to account for conservative estimates or nighttime charging)
    solar_power_scaled = 1.25 * solar_power

    # Apply the differentiable root finder to solve for cruising speed
    solver = PhysicalRootFinder.apply
    cruising_speed_objective = solver(drag_force_at_25ms, solar_power_scaled, rolling_friction_force, c_losses, max_iter, tol)

    speed_val = cruising_speed_objective.item()
    diagnostics = {
        "speed (m/s)": speed_val,
        "speed (km/h)": speed_val * 3.6,
        "drag_force (N)": drag_force_at_25ms.item(),
        "solar_power (W)": solar_power_scaled.item(),
        "rolling_friction (N)": rolling_friction_force,
        "speed_is_nan": bool(torch.isnan(cruising_speed_objective)),
        "speed_is_inf": bool(torch.isinf(cruising_speed_objective)),
    }

    return cruising_speed_objective, diagnostics
