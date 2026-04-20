"""
Augmented Lagrangian Method (ALM) state manager.
Extracted from total_loss.py to allow independent evaluation and multiplier updates
for population-based training loops (like GINN).
"""

from __future__ import annotations

from typing import Any

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.weights import weights_alm


class ALMManager:
    def __init__(
        self,
        constraint_names: list[str],
        gi: dict[str, Any],
        device: torch.device,
        active_constraint_categories: list[str] | None = None,
        configured_constraint_categories: list[str] | None = None,
    ) -> None:
        self.constraint_names = constraint_names
        self.device = device
        self.nc: int = len(constraint_names)
        self.active_constraint_categories: list[str] = active_constraint_categories or list(constraint_names)
        self.configured_constraint_categories: list[str] = configured_constraint_categories or list(
            self.active_constraint_categories
        )

        if len(self.active_constraint_categories) != self.nc:
            raise ValueError(
                "active_constraint_categories length must match number of active constraints: "
                f"{len(self.active_constraint_categories)} != {self.nc}."
            )

        lam_list = self._as_list(
            gi.get("initial_lambda", 0.01),
            self.nc,
            active_constraint_categories=self.active_constraint_categories,
            configured_constraint_categories=self.configured_constraint_categories,
        )
        mu_list = self._as_list(
            gi.get("initial_mu", 1.0),
            self.nc,
            active_constraint_categories=self.active_constraint_categories,
            configured_constraint_categories=self.configured_constraint_categories,
        )
        nu_list = self._as_list(
            gi.get("initial_nu", 1.0),
            self.nc,
            active_constraint_categories=self.active_constraint_categories,
            configured_constraint_categories=self.configured_constraint_categories,
        )

        self.alpha: float = gi.get("alpha", 0.99)
        self.gamma: float = gi.get("gamma", 0.01)
        self.gamma_aggressive: float = gi.get("gamma_aggressive", 1.1)
        self.use_alm_aggressive: bool = gi.get("ALM_aggressive") == "True"
        self.alm_aggressive_capu: float = gi.get("ALM_aggressive_capu", 0.25)

        alm_thresholds = gi.get("alm_thresholds")
        if alm_thresholds is None:
            self.thresholds_tensor = None
        else:
            threshold_list = self._as_list(
                alm_thresholds,
                self.nc,
                active_constraint_categories=self.active_constraint_categories,
                configured_constraint_categories=self.configured_constraint_categories,
            )
            self.thresholds_tensor = torch.tensor(threshold_list, device=device, dtype=torch.float32)

        self.lamdas: torch.Tensor = torch.tensor(lam_list, device=device, dtype=torch.float32).unsqueeze(1)
        self.mus: torch.Tensor = torch.tensor(mu_list, device=device, dtype=torch.float32).unsqueeze(1)
        self.nus: torch.Tensor = torch.tensor(nu_list, device=device, dtype=torch.float32).unsqueeze(1)
        self.alphas: torch.Tensor = torch.full((self.nc, 1), self.alpha, device=device)

    @staticmethod
    def _as_list(
        val: float | list[float],
        n: int,
        active_constraint_categories: list[str] | None = None,
        configured_constraint_categories: list[str] | None = None,
    ) -> list[float]:
        """Accept scalar or list, with optional category-aware remapping.

        List handling supports two formats:
        1) active-only order (length == n),
        2) full configured constraint-category order (length == len(configured_constraint_categories)).
        """
        if isinstance(val, (list, tuple)):
            values = [float(v) for v in val]
            if len(values) == 1:
                return values * n
            if len(values) == n:
                return values

            if (
                active_constraint_categories is not None
                and configured_constraint_categories is not None
                and len(values) == len(configured_constraint_categories)
            ):
                configured_index: dict[str, int] = {
                    category: idx for idx, category in enumerate(configured_constraint_categories)
                }
                missing_categories = [category for category in active_constraint_categories if category not in configured_index]
                if missing_categories:
                    raise ValueError(
                        f"Active constraint categories missing from configured constraint categories: {missing_categories}."
                    )
                return [values[configured_index[category]] for category in active_constraint_categories]

            if len(values) < n:
                raise ValueError(f"Expected at least {n} ALM values, got {len(values)}.")

            return values[:n]
        return [float(val)] * n

    def compute_penalty(self, constraints: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Compute the ALM penalty independently per constraint without updating state.
        Formula: lambda * c + 0.5 * mu * c^2
        """
        penalties: dict[str, torch.Tensor] = {}
        for i, cname in enumerate(self.constraint_names):
            ci: torch.Tensor = constraints[i]
            lam_i: torch.Tensor = self.lamdas[i].detach().clone()
            mu_i: torch.Tensor = self.mus[i].detach().clone()
            penalties[f"alm_{cname}"] = lam_i * ci + 0.5 * mu_i * ci**2
        return penalties

    def update_multipliers(self, constraints: list[torch.Tensor]) -> None:
        """
        Update the ALM multipliers (lambda, mu, nu) based on the supplied constraints.
        Constraints should be detached before passing.
        """
        c_for_alm: torch.Tensor = torch.stack([c.detach() for c in constraints])
        self.lamdas, self.mus, self.nus = weights_alm(
            lamdas=self.lamdas,
            mus=self.mus,
            nus=self.nus,
            alphas=self.alphas,
            gamma=self.gamma,
            gamma_aggressive=self.gamma_aggressive,
            constrained_losses=c_for_alm,
            thresholds=self.thresholds_tensor,
            use_alm_aggressive=self.use_alm_aggressive,
            alm_aggressive_capu=self.alm_aggressive_capu,
        )

    def get_telemetry(self) -> dict[str, float]:
        """Return a dictionary of the current multipliers suitable for logging."""
        log_dict: dict[str, float] = {}
        for i, cname in enumerate(self.constraint_names):
            log_dict[f"alm_params/lambda_{cname}"] = self.lamdas[i].item()
            log_dict[f"alm_params/mu_{cname}"] = self.mus[i].item()
            log_dict[f"alm_params/nu_{cname}"] = self.nus[i].item()
        return log_dict
