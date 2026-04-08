"""Diversity-aware population-based trainer for latent vector optimization."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_loss import field_diversity_loss
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.concat_baseclass import ConcatBaseModel


class DiversityTrainer:
    """Population-based trainer that optimizes a shared SDF model over multiple latent vectors.

    Training is split into two memory-efficient phases each step:

    Phase 1 — Per-vector backward (expensive):
        For each z_i, compute TotalLoss (CFD, raytracing, constraints) and immediately
        call backward(). Only 1 expensive computation graph lives in VRAM at a time.
        Gradients accumulate in model.parameters().grad across the loop.

    Phase 2 — Diversity backward (cheap):
        After all expensive graphs are freed, do N lightweight WIRE-only forward passes
        (no CFD, no raytracing) at a shared set of random domain points.
        All N small graphs are held simultaneously — they are negligible in VRAM.
        A single backward() injects the diversity gradient signal.
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
        diversity_num_points: int = 2048,
        diversity_bbox_size: float = 10.0,
        diversity_norm_order: int = 2,
        diversity_neighbor_agg_fn: str = "min",
    ) -> None:
        """Initialize the diversity trainer.

        Args:
            model: The SDF network (e.g. ConcatWIREV2).
            total_loss: TotalLoss instance managing all loss components.
            population_size: Number of latent vectors in the population (N).
            z_dim: Dimensionality of each latent vector.
            device: Compute device.
            model_lr: Learning rate for the model optimizer.
            diversity_weight: Weight for the diversity loss term (λ_div).
            diversity_num_points: Number of random points for SDF field sampling in diversity loss.
                With 20 GB VRAM and WIRE-only passes, 2048-8192 points are all cheap.
            diversity_bbox_size: Side length of the cubic sampling domain for diversity points.
                Should match the scale used in other losses (e.g. get_points_params.bbox_size).
            diversity_norm_order: p for L^p field distance between SDF fields.
            diversity_neighbor_agg_fn: Aggregation over inter-shape distances:
                'min' (nearest-neighbor, matches GINN paper), 'mean', or 'sum'.
        """
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self._model: ConcatBaseModel = model.to(self._device)
        self._total_loss: TotalLoss = total_loss

        self._population_size: int = population_size
        self._z_dim: int = z_dim
        self._diversity_weight: float = diversity_weight

        self._diversity_num_points: int = diversity_num_points
        self._diversity_bbox_size: float = diversity_bbox_size
        self._diversity_norm_order: int = diversity_norm_order
        self._diversity_neighbor_agg_fn: str = diversity_neighbor_agg_fn

        # Fixed population of latent vectors — not trained, serve as diverse condition anchors
        self._Z: Tensor = torch.randn(population_size, z_dim, device=self._device, dtype=torch.float32)

        self._model_optimizer: Optimizer = torch.optim.Adam(self._model.parameters(), lr=model_lr)

    # ------------------------------------------------------------------
    # Phase 1 — per-vector backward (one expensive graph at a time)
    # ------------------------------------------------------------------

    def _per_vector_backward(self, epoch: int) -> tuple[float, list[Tensor] | None]:
        """Iterate over the population and immediately backward each z's loss.

        Keeps only 1 expensive computation graph (CFD + raytracing) in VRAM at a time.
        Gradients accumulate in model.parameters().grad across iterations.

        Returns:
            - Mean per-vector loss value (float, for logging — no graph retained).
            - List of mean constraint tensors across population (detached), or None if no ALM.
        """
        use_alm: bool = getattr(self._total_loss, "_use_alm", False)
        all_raw_constraints: list[list[Tensor]] = []
        total_loss_val: float = 0.0

        for i in range(self._population_size):
            self._model.set_latent_vector(self._Z[i])

            if use_alm:
                if self._total_loss.alm_manager is None:
                    raise RuntimeError("TotalLoss ALM manager not initialized but use_alm=True.")

                loss_out, raw_constraints_dict = self._total_loss.get_total_loss(
                    model=self._model,
                    epoch_name=epoch,
                    skip_telemetry=True,
                )
                constraint_tensors = [raw_constraints_dict[name] for name in self._total_loss.alm_manager.constraint_names]
                penalties_dict = self._total_loss.alm_manager.compute_penalty(constraint_tensors)
                loss_i = loss_out + sum(penalties_dict.values())

                # Detach before storing — graph will be freed after backward()
                all_raw_constraints.append([c.detach() for c in constraint_tensors])
            else:
                loss_out = self._total_loss.get_total_loss(
                    model=self._model,
                    epoch_name=epoch,
                    skip_telemetry=True,
                )
                loss_i = loss_out[0] if isinstance(loss_out, tuple) else loss_out

            total_loss_val += loss_i.item()

            # Immediately backward — frees this z's graph before the next iteration
            (loss_i / self._population_size).backward()

        mean_constraints: list[Tensor] | None = None
        if use_alm and all_raw_constraints:
            num_c = len(all_raw_constraints[0])
            mean_constraints = [torch.stack([pop[c_idx] for pop in all_raw_constraints]).mean() for c_idx in range(num_c)]

        return total_loss_val / self._population_size, mean_constraints

    # ------------------------------------------------------------------
    # Phase 2 — diversity backward (cheap WIRE-only, all N graphs together)
    # ------------------------------------------------------------------

    def _compute_diversity_loss(self) -> Tensor:
        """Evaluate SDF fields for all z_i and compute the field-based diversity loss.

        Only the WIRE backbone is called here — TotalLoss (CFD, raytracing, etc.) is NOT
        invoked. All N small computation graphs are held simultaneously, but they are
        negligible in VRAM compared to the expensive per-vector graphs freed in Phase 1.

        Returns:
            Scalar diversity loss (negative → minimize to maximize shape diversity).
        """
        # Random points covering the design domain — no grad on coordinates
        points = (torch.rand(self._diversity_num_points, 3, device=self._device) - 0.5) * self._diversity_bbox_size

        field_values: list[Tensor] = []
        for i in range(self._population_size):
            self._model.set_latent_vector(self._Z[i])
            y_i = self._model(points)  # WIRE-only forward pass — cheap
            if isinstance(y_i, tuple):
                y_i = y_i[0]
            field_values.append(y_i.squeeze(-1))  # [N_pts]

        Y = torch.stack(field_values, dim=0)  # [population_size, N_pts]
        return field_diversity_loss(
            Y,
            norm_order=self._diversity_norm_order,
            neighbor_agg_fn=self._diversity_neighbor_agg_fn,
        )

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def step(self, epoch: int = -1) -> dict[str, float]:
        """Perform one memory-efficient optimization step.

        Phase 1: sequential per-z expensive backward (1 graph alive at a time).
        Phase 2: diversity backward using cheap WIRE-only passes (N small graphs).

        Args:
            epoch: Current epoch number (passed to TotalLoss for telemetry).

        Returns:
            Dict with 'per_vector_loss', 'diversity_loss', 'total_loss'.
        """
        self._model_optimizer.zero_grad()

        # Phase 1 — per-vector: expensive, sequential, 1 graph at a time
        per_vector_loss_val, mean_constraints = self._per_vector_backward(epoch)

        # Update ALM multipliers using mean constraints across population
        if mean_constraints is not None and self._total_loss.alm_manager is not None:
            self._total_loss.alm_manager.update_multipliers(mean_constraints)

        # Phase 2 — diversity: cheap WIRE-only, all N small graphs, then backward
        diversity_loss_val: float = 0.0
        if self._diversity_weight > 0.0:
            diversity_loss = self._compute_diversity_loss()
            (self._diversity_weight * diversity_loss).backward()
            diversity_loss_val = diversity_loss.item()

        self._model_optimizer.step()

        return {
            "per_vector_loss": per_vector_loss_val,
            "diversity_loss": diversity_loss_val,
            "total_loss": per_vector_loss_val + self._diversity_weight * diversity_loss_val,
        }

    def get_population(self) -> Tensor:
        """Retrieve current population of latent vectors (detached copy)."""
        return self._Z.detach().clone()
