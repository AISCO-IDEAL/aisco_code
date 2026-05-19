"""
Adaptive gradient clipping (AutoClip).

Inspired by: https://arxiv.org/abs/2007.14469

Tracks a rolling history of recent gradient norms and clips to their
empirical percentile instead of a fixed value. This allows the clip
threshold to adapt as the loss landscape changes — avoiding the static
ceiling that neutralises ALM multiplier growth in late training.

Usage::

    clip = AutoClip(percentile=0.8, hist_len=50, min_len=50, fallback_clip=100.0)
    # ...inside each train step, after all .backward() calls but before opt.step():
    raw_norm = torch.nn.utils.get_total_norm(model.parameters(), norm_type=2.0)
    clip.update(raw_norm.item())
    threshold = clip.get_clip_value()
    torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)

"""

from __future__ import annotations

import math

import torch


class AutoClip:
    """Adaptive gradient clipping based on rolling percentile of gradient norms.

    Before ``min_len`` updates have been recorded the ``fallback_clip`` value is
    returned, giving a static-clip warm-up period that matches legacy behaviour.

    When ``enabled=False`` the clip value is ``float('inf')`` (no clipping).

    Args:
        percentile: Fraction of the history to use as the clip threshold (e.g. 0.8
            for the 80th percentile).  Must be in (0, 1].
        hist_len: Maximum number of gradient norms to keep in the rolling buffer.
        min_len: Minimum number of updates required before switching from
            ``fallback_clip`` to the adaptive percentile threshold.
        fallback_clip: Clip value used during warm-up (before ``min_len`` updates).
            Set to ``float('inf')`` to disable clipping entirely during warm-up.
        enabled: If False, ``get_clip_value()`` always returns ``float('inf')``.
    """

    def __init__(
        self,
        percentile: float = 0.8,
        hist_len: int = 50,
        min_len: int = 50,
        fallback_clip: float = 100.0,
        enabled: bool = True,
    ) -> None:
        if not (0.0 < percentile <= 1.0):
            raise ValueError(
                f"percentile must be in (0, 1], got {percentile}. "
                "Set gradient_clipping.auto_clip_percentile in training_config.yaml."
            )
        if hist_len < 1:
            raise ValueError(
                f"hist_len must be >= 1, got {hist_len}. Set gradient_clipping.auto_clip_hist_len in training_config.yaml."
            )
        if min_len < 0:
            raise ValueError(
                f"min_len must be >= 0, got {min_len}. Set gradient_clipping.auto_clip_min_len in training_config.yaml."
            )

        self._percentile: float = percentile
        self._hist_len: int = hist_len
        self._min_len: int = min_len
        self._fallback_clip: float = fallback_clip
        self._enabled: bool = enabled

        # Circular buffer for gradient norm history
        self._history: torch.Tensor = torch.zeros(hist_len)
        self._write_idx: int = 0
        self._count: int = 0  # total number of updates recorded

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, gradient_norm: float) -> None:
        """Record a new gradient norm observation.

        Args:
            gradient_norm: The pre-clip gradient L2 norm for this step.
        """
        self._history[self._write_idx] = gradient_norm
        self._write_idx = (self._write_idx + 1) % self._hist_len
        self._count += 1

    def get_clip_value(self) -> float:
        """Return the current gradient clipping threshold.

        Returns:
            * ``float('inf')`` if ``enabled=False``.
            * ``fallback_clip`` if fewer than ``min_len`` observations have been recorded.
            * The empirical ``percentile``-th quantile of the rolling history otherwise.
        """
        if not self._enabled:
            return math.inf

        if self._count < self._min_len:
            return self._fallback_clip

        # Only use the filled portion of the buffer
        filled = min(self._count, self._hist_len)
        # Circular buffer: filled values are in write order; for quantile we just need the values
        valid = self._history if filled == self._hist_len else self._history[:filled]

        return float(torch.quantile(valid, self._percentile).item())

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of gradient norm observations recorded so far."""
        return self._count

    @classmethod
    def from_config(cls, config: dict) -> AutoClip:
        """Construct an AutoClip instance from the ``gradient_clipping`` sub-dict
        of ``training_config.yaml``.

        Expected keys::

            gradient_clipping:
              mode: "autoclip"           # "static" | "autoclip" | "none"
              static_clip_norm: 100      # fallback before history + static mode value
              auto_clip_percentile: 0.8
              auto_clip_hist_len: 50
              auto_clip_min_len: 50

        Args:
            config: The ``gradient_clipping`` sub-dict loaded from YAML.

        Returns:
            An ``AutoClip`` instance configured from the dict.  When
            ``mode`` is ``"static"`` the instance always returns
            ``static_clip_norm`` (warm-up never ends because ``min_len``
            is set to infinity).  When ``mode`` is ``"none"`` the
            instance is disabled.
        """
        mode: str = config.get("mode", "static")
        static_norm: float = float(config.get("static_clip_norm", 100.0))

        if mode == "none":
            return cls(enabled=False)

        if mode == "static":
            # Emulate static clipping: always return static_norm
            # by setting min_len so large it is never reached.
            return cls(
                percentile=0.8,
                hist_len=50,
                min_len=2**31,  # never reached
                fallback_clip=static_norm,
                enabled=True,
            )

        if mode == "autoclip":
            return cls(
                percentile=float(config.get("auto_clip_percentile", 0.8)),
                hist_len=int(config.get("auto_clip_hist_len", 50)),
                min_len=int(config.get("auto_clip_min_len", 50)),
                fallback_clip=static_norm,
                enabled=True,
            )

        raise ValueError(
            f"gradient_clipping.mode '{mode}' is not recognised. "
            "Valid values: 'static', 'autoclip', 'none'. "
            "Set gradient_clipping.mode in training_config.yaml."
        )
