"""Log-file tailing and epoch-progress parsing.

Reads a flat text log written by run_local.sh stdout. Kept pure (no subprocess
refs) so it works identically for live runs and finished-run viewer.
"""
from __future__ import annotations

import re
from pathlib import Path

_EPOCH_PATTERNS = [
    re.compile(r"\[Epoch\s+(\d+)\]"),
    re.compile(r"Epoch\s+(\d+)/"),
]
_COMPLETED_MARKERS = ("Model optimisation completed.", "Completed job successfully.")


def tail_log(log_path: Path, max_chars: int = 60_000) -> tuple[str, bool]:
    """Return (text, truncated). Text is the last max_chars of the log file."""
    if not log_path.exists():
        return "", False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[-max_chars:], True
    return text, False


def parse_epoch(log_text: str) -> int:
    """Return the highest epoch number seen in log_text, or -1 if none found."""
    best = -1
    for pattern in _EPOCH_PATTERNS:
        matches = pattern.findall(log_text)
        if matches:
            best = max(best, max(int(m) for m in matches))
    return best


def is_completed(log_text: str) -> bool:
    return any(marker in log_text for marker in _COMPLETED_MARKERS)


def get_progress(log_path: Path, total_epochs: int) -> dict[str, object]:
    """Return progress dict matching v2's /run-local/progress contract."""
    current_epoch = -1
    progress_pct = 0.0
    completed = False

    if log_path.exists():
        log_text, _ = tail_log(log_path)
        current_epoch = parse_epoch(log_text)
        if current_epoch >= 0:
            progress_pct = min(100.0, ((current_epoch + 1) / max(total_epochs, 1)) * 100.0)
        completed = is_completed(log_text)
        if completed:
            progress_pct = 100.0

    return {
        "num_epochs": total_epochs,
        "current_epoch": current_epoch,
        "progress_percent": round(progress_pct, 1),
        "completed": completed,
    }
