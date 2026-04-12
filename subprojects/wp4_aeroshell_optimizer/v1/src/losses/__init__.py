"""Losses package for wp4 v1."""

from .cruising_speed import cruising_speed
from .weights import weights_alm

__all__ = [
    "cruising_speed",
    "weights_alm",
]
