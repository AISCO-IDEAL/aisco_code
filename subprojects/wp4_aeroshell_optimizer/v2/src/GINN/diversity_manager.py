"""Diversity-aware population-based trainer for latent vector optimization."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.optimizer import Optimizer

from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass import ConcatBaseModel


class DiversityTrainer:
    """Jointly optimize model parameters and a population of latent vectors.

    Manages optimization of:
    - Model parameters via gradient descent
    - Population of latent vectors Z (size N) acts as fixed condition anchors

    Loss computation per epoch includes:
    - Per-vector loss: sum of TotalLoss.get_total_loss() for each latent vector
    - Diversity loss: encourages separation between latent vectors
    - Combined loss: weighted sum of per-vector and diversity components
    """

    def __init__(
        self,
        model: ConcatBaseModel,
        total_loss: TotalLoss,
        population_size: int,
        z_dim: int,
        device: torch.device | str = "cuda",
        model_lr: float = 1e-4,
        diversity_weight: float = 0.1,
    ) -> None:
        """Initialize the diversity trainer.

        Args:
            model: The neural network model (e.g., ConcatWIREV2) to optimize.
            total_loss: TotalLoss instance managing all loss components.
            population_size: Number of latent vectors in the population (N).
            z_dim: Dimensionality of each latent vector.
            device: Compute device (default: "cuda").
            model_lr: Learning rate for model optimizer.
            z_lr: Learning rate for latent vector optimizer.
            diversity_weight: Weight for diversity loss term (\u03bb_div).
            diversity_metric: Metric for diversity loss ("pairwise_euclidean").
        """
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self._model: ConcatBaseModel = model.to(self._device)
        self._total_loss: TotalLoss = total_loss

        self._population_size: int = population_size
        self._z_dim: int = z_dim
        self._diversity_weight: float = diversity_weight

        # Initialize population of latent vectors (fixed, non-trainable parameter space)
        self._Z: torch.Tensor = torch.randn(population_size, z_dim, device=self._device, dtype=torch.float32)

        # Create optimizer (only for model parameters)
        self._model_optimizer: Optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=model_lr,
        )

    def _compute_diversity_loss(self) -> Tensor:
        """Compute diversity loss to encourage separation in latent space.

        Currently a placeholder to be defined later based on explicit GINN constraints.
        Returns:
            Diversity loss (scalar tensor).
        """
        # TODO: Implement mathematically sound diversity loss
        # (e.g. contrastive mapping of SDF variations via volume discrepancy or deep features)
        return torch.tensor(0.0, device=self._device, requires_grad=True)

    def _compute_per_vector_loss(self, epoch: int) -> Tensor:
        """Compute total loss for each latent vector in the population.

        Returns:
            Sum of per-vector losses (scalar tensor).
        """
        total_per_vector_loss: Tensor = torch.tensor(0.0, device=self._device)

        # Iterate through each latent vector in the population
        for i in range(self._population_size):
            z_i: Tensor = self._Z[i]  # Shape: [z_dim]
            self._model.set_latent_vector(z_i)

            # Loss accumulation for z_i
            loss_i: Tensor = self._total_loss.get_total_loss(
                model=self._model,
                epoch_name=epoch,
                skip_telemetry=True,  # Important: skip wandb logging inside inner loop
            )
            total_per_vector_loss = total_per_vector_loss + loss_i

        # Average over population
        return total_per_vector_loss / self._population_size

    def step(self, epoch: int = -1) -> dict[str, float]:
        """Perform one optimization step.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary with loss metrics.
        """
        # Zero gradients
        self._model_optimizer.zero_grad()

        # Compute per-vector loss
        per_vector_loss: Tensor = self._compute_per_vector_loss(epoch)

        # Compute diversity loss
        diversity_loss: Tensor = self._compute_diversity_loss()

        # Combined loss
        combined_loss: Tensor = per_vector_loss + self._diversity_weight * diversity_loss

        # Backward pass
        combined_loss.backward()

        # Optimizer steps
        self._model_optimizer.step()

        return {
            "per_vector_loss": per_vector_loss.item(),
            "diversity_loss": diversity_loss.item(),
            "total_loss": combined_loss.item(),
        }

    def get_population(self) -> Tensor:
        """Retrieve current population of latent vectors."""
        return self._Z.detach().clone()
