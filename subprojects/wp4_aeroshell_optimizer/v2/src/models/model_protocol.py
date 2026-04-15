from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Protocol, Self, runtime_checkable

import torch


@runtime_checkable
class SupportsModelContract(Protocol):
    """
    The main shared model contract used by pretraining, loading, and storage helpers.
    EVERY Model used in our codebase should implement this contract
    """

    def get_model_info_dict(self) -> dict[str, Any]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> Any: ...

    def to(self, device: torch.device | str) -> Self: ...

    # Implemented for comfort reasons, purposefully did not add cuda() here, should be handled with .to(device)
    def cpu(self) -> Self: ...

    # comes from torch.nn.Module.train()
    def train(self) -> Self: ...
    # comes from torch.nn.Module.eval()
    def eval(self) -> Self: ...

    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class SupportsLatentConcat(SupportsModelContract, Protocol):
    """Optional contract for models that manage an active latent vector."""

    def set_latent_vector(self, z: torch.Tensor) -> None: ...

    def get_latent_vector(self) -> torch.Tensor: ...


@runtime_checkable
class SupportsOptimise(SupportsModelContract, Protocol):
    """Optional contract for models that expose a legacy optimise() entry point."""

    def optimise(self, *args: Any, **kwargs: Any) -> Any: ...
