"""Diversity-aware population-based trainer for latent vector optimization."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from subprojects.wp4_aeroshell_optimizer.v2.src.GINN.diversity_loss import field_diversity_loss
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.total_loss import TotalLoss
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat


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
        model: SupportsLatentConcat,
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
        force_z0_to_origin: bool = True,
        z_resample_cadence: int = 0,
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
            force_z0_to_origin: If True, z[0] is always pinned to the origin.
            z_resample_cadence: Resample latent vectors every N epochs (0 disables resampling).
        """
        self._device: torch.device = torch.device(device) if isinstance(device, str) else device
        self._model: SupportsLatentConcat = model.to(self._device)
        self._total_loss: TotalLoss = total_loss

        self._population_size: int = population_size
        self._z_dim: int = z_dim
        self._diversity_weight: float = diversity_weight

        self._diversity_num_points: int = diversity_num_points
        self._diversity_bbox_size: float = diversity_bbox_size
        self._diversity_norm_order: int = diversity_norm_order
        self._diversity_neighbor_agg_fn: str = diversity_neighbor_agg_fn
        self._force_z0_to_origin: bool = force_z0_to_origin
        self._z_resample_cadence: int = max(0, int(z_resample_cadence))

        generator_device = self._device.type if self._device.type in {"cpu", "cuda"} else "cpu"
        self._resample_generator = torch.Generator(device=generator_device)

        # Fixed population of latent vectors — not trained, serve as diverse condition anchors
        self._Z: Tensor = torch.randn(population_size, z_dim, device=self._device, dtype=torch.float32)
        if self._force_z0_to_origin and self._population_size > 0:
            self._Z[0] = torch.zeros(z_dim, device=self._device, dtype=torch.float32)

        self._model_optimizer: Optimizer = torch.optim.Adam(self._model.parameters(), lr=model_lr)

    def _resample_population(self, epoch: int) -> None:
        """Resample latent population using a local generator to avoid global RNG side effects."""
        if self._population_size <= 0:
            return

        if self._z_resample_cadence <= 0:
            return

        start_idx = 1 if self._force_z0_to_origin else 0
        if start_idx >= self._population_size:
            if self._force_z0_to_origin:
                self._Z[0] = torch.zeros(self._z_dim, device=self._device, dtype=torch.float32)
            return

        resample_step = epoch // self._z_resample_cadence
        self._resample_generator.manual_seed(resample_step)

        self._Z[start_idx:] = torch.randn(
            self._population_size - start_idx,
            self._z_dim,
            device=self._device,
            dtype=torch.float32,
            generator=self._resample_generator,
        )

        if self._force_z0_to_origin:
            self._Z[0] = torch.zeros(self._z_dim, device=self._device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Phase 1 — per-vector backward (one expensive graph at a time)
    # ------------------------------------------------------------------

    def _per_vector_backward(self, epoch: int) -> tuple[float, list[Tensor] | None, dict[str, float]]:
        """Iterate over the population and immediately backward each z's loss.

        Keeps only 1 expensive computation graph (CFD + raytracing) in VRAM at a time.
        Gradients accumulate in model.parameters().grad across iterations.

        Returns:
            - Mean per-vector loss value (float, for logging — no graph retained).
            - List of mean constraint tensors across population (detached), or None if no ALM.
            - Dict of averaged raw metrics across the population.
        """
        use_alm: bool = getattr(self._total_loss, "_use_alm", False)
        external_alm: bool = bool(getattr(self._total_loss, "_external_alm", False))
        all_raw_constraints: list[list[Tensor]] = []
        total_loss_val: float = 0.0

        summed_raw_metrics: dict[str, float] = {}
        summed_weighted_metrics: dict[str, float] = {}

        for i in range(self._population_size):
            self._model.set_latent_vector(self._Z[i])
            penalties_dict: dict[str, Tensor] = {}

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

            # Accumulate metrics for telemetry
            for k, v in self._total_loss.latest_raw_metrics.items():
                summed_raw_metrics[k] = summed_raw_metrics.get(k, 0.0) + v.item()
            for k, v in self._total_loss._last_weighted_losses.items():
                summed_weighted_metrics[k] = summed_weighted_metrics.get(k, 0.0) + v.item()
            if external_alm:
                for k, v in penalties_dict.items():
                    summed_weighted_metrics[k] = summed_weighted_metrics.get(k, 0.0) + v.item()

            # Immediately backward — frees this z's graph before the next iteration
            (loss_i / self._population_size).backward()

        mean_constraints: list[Tensor] | None = None
        if use_alm and all_raw_constraints:
            num_c = len(all_raw_constraints[0])
            mean_constraints = [torch.stack([pop[c_idx] for pop in all_raw_constraints]).mean() for c_idx in range(num_c)]

        avg_raw_metrics = {k: v / self._population_size for k, v in summed_raw_metrics.items()}
        avg_weighted_metrics = {k: v / self._population_size for k, v in summed_weighted_metrics.items()}
        avg_total_loss = total_loss_val / self._population_size

        # Manually overwrite and trigger telemetry
        self._model.set_latent_vector(self._Z[0])

        loss_log = {f"raw_losses/{k}": v for k, v in avg_raw_metrics.items()}
        loss_log["raw_losses/total_loss"] = avg_total_loss

        self._total_loss._last_weighted_losses = {
            k: torch.tensor(v, device=self._device) for k, v in avg_weighted_metrics.items()
        }
        self._total_loss._handle_telemetry(
            loss_log=loss_log,
            epoch_name=epoch,
            device=self._device,
            model=self._model,
            total_loss=torch.tensor(avg_total_loss, device=self._device),
            skip_gradients=True,
        )

        return avg_total_loss, mean_constraints, avg_raw_metrics

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
        if self._z_resample_cadence > 0 and epoch > 0 and epoch % self._z_resample_cadence == 0:
            self._resample_population(epoch)

        self._model_optimizer.zero_grad()

        # Phase 1 — per-vector: expensive, sequential, 1 graph at a time
        per_vector_loss_val, mean_constraints, avg_raw_metrics = self._per_vector_backward(epoch)

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

        metrics = {
            "per_vector_loss": per_vector_loss_val,
            "diversity_loss": diversity_loss_val,
            "total_loss": per_vector_loss_val + self._diversity_weight * diversity_loss_val,
        }
        metrics.update(avg_raw_metrics)
        return metrics

    def get_population(self) -> Tensor:
        """Retrieve current population of latent vectors (detached copy)."""
        return self._Z.detach().clone()
