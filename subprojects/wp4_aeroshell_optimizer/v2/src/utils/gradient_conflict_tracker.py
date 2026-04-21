from pathlib import Path

import numpy as np
import torch


class GradientConflictTracker:
    """Computes and logs per-component gradient vectors and their cosine similarities."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pending_vectors: dict[int, dict[str, np.ndarray]] = {}

    def capture_vectors(
        self,
        named_components: dict[str, torch.Tensor],
        trainable_params: list[torch.nn.Parameter],
        epoch: int,
    ) -> dict[str, np.ndarray]:
        """
        Computes per-component gradients via torch.autograd.grad() and stores them in
        _pending_vectors[epoch]. Does NOT save to disk — call finalize_epoch() after
        adding any additional vectors (e.g. diversity).

        Args:
            named_components: dict mapping loss name → live weighted loss tensor (with autograd graph)
            trainable_params: list of parameters with requires_grad=True
            epoch: epoch number

        Returns:
            dict mapping param id → accumulated gradient (ready to assign to param.grad for optimizer step)
        """
        if not named_components:
            return {}

        component_items = list(named_components.items())
        grad_vectors: dict[str, np.ndarray] = {}
        per_component_grads: dict[str, tuple] = {}

        for i, (loss_name, component) in enumerate(component_items):
            retain = i < len(component_items) - 1
            try:
                grads = torch.autograd.grad(
                    component,
                    trainable_params,
                    retain_graph=retain,
                    create_graph=False,
                    allow_unused=True,
                )
            except RuntimeError as e:
                print(f"Warning: autograd.grad failed for {loss_name}: {e}")
                continue

            grad_vec = np.concatenate([
                (g.detach().cpu().numpy().flatten() if g is not None else np.zeros(p.numel(), dtype=np.float32))
                for g, p in zip(grads, trainable_params, strict=True)
            ])
            grad_vectors[loss_name] = grad_vec
            per_component_grads[loss_name] = grads

        loss_names = list(grad_vectors.keys())
        if not loss_names:
            print("Warning: No valid gradients computed; skipping conflict save.")
            return {}

        self._pending_vectors[epoch] = grad_vectors

        return self._accumulate_grads_for_optimizer(per_component_grads, trainable_params, loss_names)

    def add_named_vector(self, epoch: int, name: str, grad_vec: np.ndarray) -> None:
        """
        Appends a pre-computed flat gradient vector to the pending epoch data.
        Must be called after capture_vectors() and before finalize_epoch().

        Args:
            epoch: epoch number matching a prior capture_vectors() call
            name: label for this gradient component (e.g. "diversity")
            grad_vec: flattened numpy gradient array with same length as other vectors
        """
        if epoch not in self._pending_vectors:
            print(f"Warning: No pending vectors for epoch {epoch}; add_named_vector skipped.")
            return
        self._pending_vectors[epoch][name] = grad_vec

    def finalize_epoch(self, epoch: int) -> None:
        """
        Computes cosine similarity matrix and gradient norms from all pending vectors
        for the given epoch, saves to disk, and clears the pending entry.

        Args:
            epoch: epoch number to finalize
        """
        if epoch not in self._pending_vectors:
            return

        grad_vectors = self._pending_vectors.pop(epoch)
        loss_names = list(grad_vectors.keys())
        cos_sim_matrix = self._compute_cosine_similarities(grad_vectors)
        grad_norms = self._compute_gradient_norms(grad_vectors)
        self._save_to_disk(cos_sim_matrix, grad_norms, loss_names, epoch)

    def save_from_grad_vectors(
        self,
        named_grad_vectors: dict[str, np.ndarray],
        epoch: int,
    ) -> None:
        """
        Saves pre-computed gradient vectors (e.g., from delta_grad in WIRE_v2/BasicMLP).
        Useful for paths that already have per-component gradient vectors without needing autograd.grad.

        Args:
            named_grad_vectors: dict mapping loss name → flattened gradient numpy array
            epoch: epoch number for file naming
        """
        if not named_grad_vectors:
            return

        loss_names = list(named_grad_vectors.keys())
        cos_sim_matrix = self._compute_cosine_similarities(named_grad_vectors)
        grad_norms = self._compute_gradient_norms(named_grad_vectors)

        self._save_to_disk(cos_sim_matrix, grad_norms, loss_names, epoch)

    def _compute_cosine_similarities(self, grad_vectors: dict[str, np.ndarray]) -> np.ndarray:
        """Returns NxN cosine similarity matrix (range [-1, 1])."""
        loss_names = list(grad_vectors.keys())
        n = len(loss_names)
        cos_sim = np.zeros((n, n), dtype=np.float32)

        for i, (_name_i, vec_i) in enumerate(grad_vectors.items()):
            for j, (_name_j, vec_j) in enumerate(grad_vectors.items()):
                if i == j:
                    cos_sim[i, j] = 1.0
                else:
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    if norm_i > 1e-8 and norm_j > 1e-8:
                        cos_sim[i, j] = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                    else:
                        cos_sim[i, j] = 0.0

        return cos_sim

    def _compute_gradient_norms(self, grad_vectors: dict[str, np.ndarray]) -> np.ndarray:
        """Returns 1D array of L2 gradient norms."""
        return np.array([np.linalg.norm(vec) for vec in grad_vectors.values()], dtype=np.float32)

    def _save_to_disk(
        self,
        cos_sim_matrix: np.ndarray,
        grad_norms: np.ndarray,
        loss_names: list[str],
        epoch: int,
    ) -> None:
        """Saves conflict data to .npz file."""
        file_path = self.output_dir / f"epoch_{epoch:04d}.npz"
        np.savez(
            file_path,
            cos_sim=cos_sim_matrix,
            grad_norms=grad_norms,
            loss_names=np.array(loss_names, dtype=str),
            epoch=epoch,
        )

    def _accumulate_grads_for_optimizer(
        self,
        per_component_grads: dict[str, tuple],
        trainable_params: list[torch.nn.Parameter],
        loss_names: list[str],
    ) -> dict[str, np.ndarray]:
        """
        Accumulates per-component gradient tuples into param.grad attributes
        (mathematically equivalent to summing component losses and calling .backward()).

        Returns dict mapping param→grad for inspection if needed.
        """
        param_to_grad = {}

        for param_idx, param in enumerate(trainable_params):
            accumulated = None
            for loss_name in loss_names:
                grads = per_component_grads.get(loss_name)
                if grads is None or param_idx >= len(grads):
                    continue
                g = grads[param_idx]
                if g is not None:
                    accumulated = g.detach().clone() if accumulated is None else accumulated + g.detach()

            if accumulated is not None:
                param.grad = accumulated
                param_to_grad[id(param)] = accumulated.detach().cpu().numpy()

        return param_to_grad
