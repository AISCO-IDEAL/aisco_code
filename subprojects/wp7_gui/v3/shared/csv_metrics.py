"""CSV metrics writer for per-run epoch tracking and global summary.

Two outputs:
- Per-run: ``{run_dir}/epoch_metrics.csv`` — one row per epoch, appended live.
- Global:  ``{summary_path}``             — one row per GUI run (upserted), final
  epoch values so every run appears at its latest known state.
"""

from __future__ import annotations

import contextlib
import csv
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_PER_RUN_FIELDS = ("epoch", "timestamp", "drag_value", "solar_power_w", "cruising_speed_ms")
_SUMMARY_FIELDS = ("run_id", "cfg_session", "timestamp", "epoch", "drag_value", "solar_power_w", "cruising_speed_ms")


def _fmt(value: float | None) -> str:
    """Format a float metric for CSV; None becomes an empty string."""
    if value is None:
        return ""
    return repr(value)


def _parse_float(raw: str) -> float | None:
    """Parse a CSV cell back to float, returning None for empty/invalid cells."""
    stripped = raw.strip()
    if not stripped:
        return None
    with contextlib.suppress(ValueError):
        return float(stripped)
    return None


class CSVMetricsWriter:
    """Thread-safe writer for epoch metrics CSVs.

    Args:
        summary_path: Absolute path to the global ``gui_runs_summary.csv``.
    """

    def __init__(self, summary_path: Path) -> None:
        self._summary_path = summary_path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Per-run CSV
    # ------------------------------------------------------------------

    def write_epoch_row(
        self,
        run_dir: Path,
        epoch: int,
        drag_value: float | None,
        solar_power_w: float | None,
        cruising_speed_ms: float | None,
    ) -> None:
        """Append one epoch row to ``run_dir/epoch_metrics.csv``.

        Creates the file with a header on first call; appends thereafter.

        Args:
            run_dir: WP4 output directory for the current run.
            epoch: Zero-based epoch index.
            drag_value: Predicted drag force in Newtons, or None if unavailable.
            solar_power_w: Solar output in Watts, or None if unavailable.
            cruising_speed_ms: Cruising speed estimate in m/s, or None if unavailable.
        """
        csv_path = run_dir / "epoch_metrics.csv"
        needs_header = not csv_path.exists()
        row = {
            "epoch": epoch,
            "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "drag_value": _fmt(drag_value),
            "solar_power_w": _fmt(solar_power_w),
            "cruising_speed_ms": _fmt(cruising_speed_ms),
        }
        try:
            with csv_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=_PER_RUN_FIELDS)
                if needs_header:
                    writer.writeheader()
                writer.writerow(row)
        except OSError as exc:
            LOGGER.warning("Could not write epoch CSV %s: %s", csv_path, exc)

    # ------------------------------------------------------------------
    # Global summary CSV
    # ------------------------------------------------------------------

    def write_summary_row(
        self,
        run_id: str,
        cfg_session: str,
        epoch: int,
        drag_value: float | None,
        solar_power_w: float | None,
        cruising_speed_ms: float | None,
    ) -> None:
        """Upsert one row in the global summary CSV.

        If a row for ``run_id`` already exists it is updated in place; otherwise
        a new row is appended.  Thread-safe via an instance-level lock.

        Args:
            run_id: Unique identifier for this run (e.g. W&B run name).
            cfg_session: GUI override session basename (e.g. ``gui_override_20260521_123456``).
            epoch: Current epoch (typically the latest/final epoch).
            drag_value: Drag force in Newtons, or None.
            solar_power_w: Solar output in Watts, or None.
            cruising_speed_ms: Cruising speed in m/s, or None.
        """
        new_row: dict[str, str] = {
            "run_id": run_id,
            "cfg_session": cfg_session,
            "timestamp": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "epoch": str(epoch),
            "drag_value": _fmt(drag_value),
            "solar_power_w": _fmt(solar_power_w),
            "cruising_speed_ms": _fmt(cruising_speed_ms),
        }
        with self._lock:
            self._summary_path.parent.mkdir(parents=True, exist_ok=True)
            existing: list[dict[str, str]] = []
            if self._summary_path.exists():
                try:
                    with self._summary_path.open(newline="", encoding="utf-8") as fh:
                        existing = list(csv.DictReader(fh))
                except OSError as exc:
                    LOGGER.warning("Could not read summary CSV %s: %s", self._summary_path, exc)

            updated = False
            for i, row in enumerate(existing):
                if row.get("run_id") == run_id:
                    existing[i] = new_row
                    updated = True
                    break
            if not updated:
                existing.append(new_row)

            try:
                with self._summary_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=_SUMMARY_FIELDS)
                    writer.writeheader()
                    writer.writerows(existing)
            except OSError as exc:
                LOGGER.warning("Could not write summary CSV %s: %s", self._summary_path, exc)

    # ------------------------------------------------------------------
    # Read latest metrics
    # ------------------------------------------------------------------

    def read_latest_metrics(self, run_dir: Path) -> dict[str, int | float | None]:
        """Return the last row of ``run_dir/epoch_metrics.csv`` as a dict.

        Returns an empty dict when the file does not exist or is empty.

        Args:
            run_dir: WP4 output directory for the current run.

        Returns:
            Dict with keys ``epoch``, ``drag_value``, ``solar_power_w``,
            ``cruising_speed_ms``, or ``{}`` if unavailable.
        """
        csv_path = run_dir / "epoch_metrics.csv"
        if not csv_path.exists():
            return {}
        try:
            with csv_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        except OSError as exc:
            LOGGER.warning("Could not read epoch CSV %s: %s", csv_path, exc)
            return {}
        if not rows:
            return {}
        last = rows[-1]
        epoch_raw = last.get("epoch", "").strip()
        epoch: int | None = None
        with contextlib.suppress(ValueError):
            epoch = int(epoch_raw)
        if epoch is None:
            return {}
        return {
            "epoch": epoch,
            "drag_value": _parse_float(last.get("drag_value", "")),
            "solar_power_w": _parse_float(last.get("solar_power_w", "")),
            "cruising_speed_ms": _parse_float(last.get("cruising_speed_ms", "")),
        }
