import torch

class PhysicalRootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, drag_force_at_25ms, solar_power, rolling_friction_force, c):
        # We solve: f(u) = drag_force_normalized * u^3 + rolling_friction_force * u + (c - solar_power) = 0
        with torch.no_grad():
            # Initialization: Since drag and solar_power are > 0, 
            # a starting value of 1.0 is usually safe for convergence.
            u = torch.ones_like(drag_force_at_25ms)

            # Newton's Method (5-10 iterations are typically sufficient given the monotonicity)
            for _ in range(8):
                f = drag_force_at_25ms * u**3/ (25**2) + rolling_friction_force * u + c - solar_power
                df_du = 3 / (25**2) * drag_force_at_25ms * u**2 + rolling_friction_force
                u = u - f / df_du
                
                # Ensure we remain within the physical domain (speed must be positive)
                u = torch.clamp(u, min=1e-6) # TODO: is clamping feasible?

        ctx.save_for_backward(u, drag_force_at_25ms, solar_power, rolling_friction_force, c)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, drag_force_at_25ms, solar_power, rolling_friction_force, c = ctx.saved_tensors

        # Partial derivatives of the function F = u^2 / (25m/s)^2 * drag_force_at_25ms*u + rolling_friction_force*u + c - solar_power
        df_du = 3 / (25**2) * drag_force_at_25ms * u**2 + rolling_friction_force
        df_ddrag_force_at_25ms = u**3 / (25**2)
        df_dsolar_power = -torch.ones_like(solar_power)

        # Calculate implicit derivative
        # grad_output is dLoss/du
        # We need dLoss/ddrag_force_at_25ms = dLoss/du * du/ddrag_force_at_25ms
        # using du/ddrag_force_at_25ms = - (df/ddrag_force_at_25ms) / (df/du)

        inv_df_du = -grad_output / df_du

        grad_drag_force_at_25ms = inv_df_du * df_ddrag_force_at_25ms
        grad_solar_power = inv_df_du * df_dsolar_power

        # rolling_friction_force and c are constants relative to this solver -> None
        return grad_drag_force_at_25ms, grad_solar_power, None, None

def cruising_speed(solar_power: torch.Tensor, drag_force_at_25ms: torch.Tensor, mass: float, c_losses: float, rolling_friction_coefficient: float) -> torch.Tensor:
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
        cruising_speed_objective: Estimated speed in [m/s]
    """
    # Rolling friction coefficient based on ACE experience; 
    # could be refined with lab data interpolation in the future.
    rolling_friction_coefficient = 0.006 

    # Power loss due to rolling friction
    rolling_friction_force = mass * 9.81 * rolling_friction_coefficient

    # Bias for solar charging (e.g., to account for conservative estimates or nighttime charging)
    solar_power = 1.25 * solar_power

    # Apply the differentiable root finder to solve for cruising speed
    solver = PhysicalRootFinder.apply
    cruising_speed_objective = solver(drag_force_at_25ms, solar_power, rolling_friction_force, c_losses)

    return cruising_speed_objective