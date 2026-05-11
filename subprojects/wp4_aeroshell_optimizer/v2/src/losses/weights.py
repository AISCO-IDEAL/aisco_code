"""
weights for constraints of aeroshell_optimizer are handeled here

"""

import torch


@torch.no_grad()
def weights_alm(
    lamdas: torch.Tensor,
    mus: torch.Tensor,
    nus: torch.Tensor,
    alphas: torch.Tensor,
    gamma: float,
    gamma_aggressive: float,
    constrained_losses: torch.Tensor,
    thresholds: torch.Tensor | None = None,
    use_alm_aggressive: bool = False,
    alm_aggressive_capu: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = 1e-16

    # --- SCHUTZ VOR BROADCASTING FEHLERN ---
    # Wir machen aus allen Tensoren 1D-Vektoren (z.B. Shape [3]).
    # Das verhindert, dass [3, 1] und [3] zu einer [3, 3] Matrix explodieren.
    lams = lamdas.view(-1)
    ms = mus.view(-1)
    ns = nus.view(-1)
    alps = alphas.view(-1)
    closs = constrained_losses.view(-1)

    if use_alm_aggressive:
        # --- CAPU LOGIK ---
        eta = alm_aggressive_capu
        # 1. Update Mu nur wenn nötig (Sufficient Decrease Condition)
        update_mask = torch.abs(closs) > (eta * ns)
        ms[update_mask] *= gamma_aggressive
        ms.clamp_(max=10000.0)

        # 2. Update Nu (Benchmark) in-place
        ns.mul_(alps).add_((1 - alps) * torch.abs(closs))

        # Lambda update for aggressive mode (signed c)
        lams.addcmul_(ms, closs)

    else:
        # --- RMS LOGIK (Klassisch Berzins) ---
        # Optional deadzone: if |c_i| < threshold_i, freeze nu/mu/lambda for that constraint.
        # thresholds=None disables the deadzone (pure adaptive formula).
        if thresholds is not None:
            ts = thresholds.view(-1)
            if ts.numel() != closs.numel():
                raise ValueError(
                    f"ALM thresholds size mismatch: expected {closs.numel()} values for active constraints, got {ts.numel()}."
                )
            active = closs.abs() >= ts
            ns_orig = ns.clone()
            ms_orig = ms.clone()
            lam_orig = lams.clone()

        ns.mul_(alps).add_((1 - alps) * (closs * closs))
        new_mus = (gamma / (torch.sqrt(ns) + epsilon)).clamp_(max=10000.0)
        ms.copy_(new_mus)
        # Lambda update: λ += μ · |C| (per spec; abs is equivalent for non-negative constraint losses)
        lams.addcmul_(ms, closs.abs())

        # Restore frozen constraints (deadzone: undo updates where |c| < threshold)
        if thresholds is not None:
            ns.copy_(torch.where(active, ns, ns_orig))
            ms.copy_(torch.where(active, ms, ms_orig))
            lams.copy_(torch.where(active, lams, lam_orig))

    # Wir geben die originalen Tensoren zurück (die jetzt aktualisiert sind)
    return lamdas, mus, nus
