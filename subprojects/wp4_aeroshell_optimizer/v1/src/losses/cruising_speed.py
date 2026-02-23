import torch


## start GEMINI Block
class PhysicalRootFinder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, drag_force_normalized, solar_power, rolling_friction_force, c):
        # Wir lösen: f(x) = drag_force_normalized*u^3 + rolling_friction_force*u + (c - solar_power) = 0
        with torch.no_grad():
            # Initialisierung: Da obj1, solar_power > 0, ist ein Startwert bei 1.0 meist sicher
            u = torch.ones_like(drag_force_normalized)

            # Newton-Verfahren (5-10 Iterationen reichen bei Monotonie meist völlig aus)
            for _ in range(8):
                f = drag_force_normalized * u**3 + rolling_friction_force * u + c - solar_power
                df_du = 3 * drag_force_normalized * u**2 + rolling_friction_force
                u = u - f / df_du
                # Sicherstellen, dass wir im physikalischen Bereich bleiben
                u = torch.clamp(u, min=1e-6)

        ctx.save_for_backward(u, drag_force_normalized, solar_power, rolling_friction_force, c)
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, drag_force_normalized, solar_power, rolling_friction_force, c = ctx.saved_tensors

        # Partielle Ableitungen der Funktion F = drag_force_normalized*u^3 + rolling_friction_force*u + c - solar_power
        df_du = 3 * drag_force_normalized * u**2 + rolling_friction_force
        df_ddrag_force_normalized = u**3
        df_dsolar_power = -torch.ones_like(solar_power)

        # Implizite Ableitung berechnen
        # grad_output ist dLoss/dx
        # Wir brauchen dLoss/ddrag_force_normalized = dLoss/dx * dx/ddrag_force_normalized
        # mit dx/ddrag_force_normalized = - (df/ddrag_force_normalized) / (df/du)

        inv_df_du = -grad_output / df_du

        grad_drag_force_normalized = inv_df_du * df_ddrag_force_normalized
        grad_solar_power = inv_df_du * df_dsolar_power

        # f_rf und c sind Konstanten -> None
        return grad_drag_force_normalized, grad_solar_power, None, None
### end GEMINI Block

def cruising_speed(solar_power: torch.Tensor, drag_force_normalized: torch.Tensor, mass: float, c_losses: float, rolling_friction_coefficient: float) -> torch.Tensor:
    """we combine solar power prediction and drag force into a single objective, the cruising speed

    args:
        solar power prediction
        speed^2 normalized drag force
        mass of silvretta
        other physical parameters
    returns:
        cruising speed
    """

    rolling_friction_coefficient = 0.006 # according to experience of ACE, could be refined with an interpolation of the rolling friction coefficients at different speeds, based on bridgestone's lab data

    # powerloss due to rolling friction
    rolling_friction_force = mass * 9.81 * rolling_friction_coefficient

    # powerloss in electronic drive train

    # bias for solar charging in the night (no drag force whatsoever)
    solar_power = 1.25*solar_power

    # solve for cruising speed
    solve_x = PhysicalRootFinder.apply
    cruising_speed_objective = solve_x(solar_power, drag_force_normalized, rolling_friction_force, c_losses)


    return cruising_speed_objective
