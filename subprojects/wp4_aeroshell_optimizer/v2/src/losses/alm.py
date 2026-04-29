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
    """
    Manages the state of the Augmented Lagrangian Method (ALM) for constraint optimisation.

    This class maintains the multipliers (lambda, mu, nu) for each constraint and provides methods to compute the ALM penalty and update the multipliers based on observed constraint values.

    Methods:
        compute_penalty: Compute the ALM penalty for given constraint values.
        update_multipliers: Update the ALM multipliers based on observed constraint values.
        get_telemetry() -> dict[str, float]: Get the current multipliers for logging purposes.


    """

    def __init__(
        self,
        constraint_names: list[str],
        gi: dict[str, Any],
        device: torch.device,
        active_constraint_categories: list[str],
        configured_constraint_categories: list[str],
    ) -> None:
        """
        Initialisation of the ALMManager.

        Needs:
        - constraint_names: List of all constraint names, in a fixed order.
        - gi: the ["general_info"] entry of the loss config. Should contain:
            - initial_lambda: scalar (not recommended) or list of initial lambda values.
            - initial_mu: scalar (not recommended) or list of initial mu values.
            - initial_nu: scalar (not recommended) or list of initial nu values.
            - alpha: ALM alpha parameter (default 0.99).
            - gamma: ALM gamma parameter (default 0.01).
            - gamma_aggressive: ALM aggressive gamma multiplier (default 1.1).
            - ALM_aggressive: whether to use aggressive multiplier updates (default False).
            - ALM_aggressive_capu: cap for aggressive multiplier updates (default 0.25).
            - alm_thresholds: optional per-constraint deadzone. In standard mode: if |c_i| < threshold_i, nu/mu/lambda for that constraint are not updated (frozen). Set to null/absent for pure adaptive formula with no deadzone. Ignored in aggressive mode (which uses alm_aggressive_capu instead).
        - device: torch.device to store the multipliers on.
        - active_constraint_categories: Comes from the TotalLossV2.constraint_category_order property. This returns which categories are considered active.
        - configured_constraint_categories: List of ALL constraints defined in the loss config. Needs to be provided so the initial ALM Parameter lists can stay set in the same length and order regardless of disabled constraints, the params will be properly remapped to the active constraints based on their categories.
        """
        self.constraint_names = constraint_names
        self.device = device
        self.nc: int = len(constraint_names)
        # assume all constraints are active if no active categories are provided, otherwise filter by active categories
        self.active_constraint_categories: list[str] = active_constraint_categories or list(constraint_names)
        # set variable containing all defined constraint categories for proper mapping of initial multipliers, even if not all are active.
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

    def compute_penalty(
        self, constraints: list[torch.Tensor], active_indices: list[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Compute the ALM penalty independently per constraint without updating state.
        Formula: lambda * c + 0.5 * mu * c^2

        Args:
            constraints: List of Tensors, one tensor per active constraint, in the order of active_indices.
            active_indices: Indices into self.constraint_names that are currently active.
                If None, all constraints are assumed active and constraints must have length self.nc.
        """
        indices = active_indices if active_indices is not None else list(range(self.nc))
        penalties: dict[str, torch.Tensor] = {}
        for j, i in enumerate(indices):
            cname = self.constraint_names[i]
            ci: torch.Tensor = constraints[j]
            lam_i: torch.Tensor = self.lamdas[i].detach().clone()
            mu_i: torch.Tensor = self.mus[i].detach().clone()
            penalties[f"alm_{cname}"] = lam_i * ci + 0.5 * mu_i * ci**2
        return penalties

    def update_multipliers(self, constraints: list[torch.Tensor], active_indices: list[int] | None = None) -> None:
        """
        Update the ALM multipliers (lambda, mu, nu) based on the supplied constraints.
        Multipliers for indices not in active_indices are frozen (not updated).
        Constraints should be detached before passing.

        Args:
            constraints: One tensor per active constraint, in the order of active_indices.
            active_indices: Indices into self.constraint_names to update.
                If None, all constraints are updated and constraints must have length self.nc.
        """
        indices = active_indices if active_indices is not None else list(range(self.nc))
        if not indices:
            return
        c_for_alm: torch.Tensor = torch.stack([c.detach() for c in constraints])
        new_lam, new_mu, new_nu = weights_alm(
            lamdas=self.lamdas[indices],
            mus=self.mus[indices],
            nus=self.nus[indices],
            alphas=self.alphas[indices],
            gamma=self.gamma,
            gamma_aggressive=self.gamma_aggressive,
            constrained_losses=c_for_alm,
            thresholds=self.thresholds_tensor[indices] if self.thresholds_tensor is not None else None,
            use_alm_aggressive=self.use_alm_aggressive,
            alm_aggressive_capu=self.alm_aggressive_capu,
        )
        for j, i in enumerate(indices):
            self.lamdas[i] = new_lam[j]
            self.mus[i] = new_mu[j]
            self.nus[i] = new_nu[j]

    def get_telemetry(self) -> dict[str, float]:
        """Return a dictionary of the current multipliers suitable for logging."""
        log_dict: dict[str, float] = {}
        for i, cname in enumerate(self.constraint_names):
            log_dict[f"alm_params/lambda_{cname}"] = self.lamdas[i].item()
            log_dict[f"alm_params/mu_{cname}"] = self.mus[i].item()
            log_dict[f"alm_params/nu_{cname}"] = self.nus[i].item()
        return log_dict
