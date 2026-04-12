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

        # 2. Update Nu (Benchmark) in-place
        ns.mul_(alps).add_((1 - alps) * torch.abs(closs))

    else:
        # --- RMS LOGIK (Klassisch Berzins) ---
        # 1. Update Nu (EMA des quadratischen Fehlers) in-place
        ns.mul_(alps).add_((1 - alps) * (closs * closs))

        # 2. Update Mu basierend auf Nu
        new_mus = gamma / (torch.sqrt(ns) + epsilon)

        if thresholds is not None:
            ts = thresholds.view(-1)
            min_mu = gamma / (ts + epsilon)
            new_mus = torch.max(new_mus, min_mu)

        ms.copy_(new_mus)

    # --- LAMBDA UPDATE ---
    # Das entspricht: lamda = lamda + mu * c(x)
    # addcmul_ macht das in-place und extrem effizient
    lams.addcmul_(ms, closs)

    # Wir geben die originalen Tensoren zurück (die jetzt aktualisiert sind)
    return lamdas, mus, nus
