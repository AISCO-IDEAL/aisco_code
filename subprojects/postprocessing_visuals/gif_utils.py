#!/usr/bin/env python3
"""Interactive GIF utility for trimming or dynamic speed-up.

Features
--------
- Prompts for input GIF path.
- Prompts for operation:
  1) Remove frames between user-provided start/end frame numbers.
  2) Dynamic speed-up from 1x at first frame to target x at last frame.
     Supports linear or exponential ramp (asks exponent).
- Saves result in the same folder as input GIF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageSequence


def _prompt_non_empty(prompt: str) -> str:
    while True:
        val = input(prompt).strip()
        if val:
            return val
        print("Input cannot be empty.")


def _prompt_int(prompt: str, min_value: int | None = None, max_value: int | None = None) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            val = int(raw)
        except ValueError:
            print("Please enter an integer.")
            continue

        if min_value is not None and val < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        if max_value is not None and val > max_value:
            print(f"Please enter a value <= {max_value}.")
            continue
        return val


def _prompt_float(prompt: str, min_value: float | None = None) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            val = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue

        if min_value is not None and val < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return val


def _load_gif(path: Path) -> tuple[list[Image.Image], list[int], dict[str, Any]]:
    frames: list[Image.Image] = []
    durations: list[int] = []

    with Image.open(path) as img:
        base_info = dict(img.info)
        default_duration = int(base_info.get("duration", 100))

        for frame in ImageSequence.Iterator(img):
            copied = frame.copy()
            frames.append(copied)
            durations.append(int(frame.info.get("duration", default_duration)))

    return frames, durations, base_info


def _save_gif(
    out_path: Path,
    frames: list[Image.Image],
    durations: list[int],
    base_info: dict[str, Any],
) -> None:
    if not frames:
        raise ValueError("No frames to save.")

    loop = int(base_info.get("loop", 0))

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        disposal=2,
        optimize=False,
    )


def _remove_frame_range(
    frames: list[Image.Image],
    durations: list[int],
    start_idx: int,
    end_idx: int,
) -> tuple[list[Image.Image], list[int]]:
    kept_frames: list[Image.Image] = []
    kept_durations: list[int] = []

    for i, (frm, dur) in enumerate(zip(frames, durations, strict=True)):
        if start_idx <= i <= end_idx:
            continue
        kept_frames.append(frm)
        kept_durations.append(dur)

    return kept_frames, kept_durations


def _speedup_durations_linear(durations: list[int], target_speedup: float) -> list[int]:
    n = len(durations)
    if n <= 1:
        return durations[:]

    out: list[int] = []
    for i, dur in enumerate(durations):
        t = i / (n - 1)
        factor = 1.0 + ((target_speedup - 1.0) * t)
        new_dur = max(10, round(dur / factor))
        out.append(new_dur)
    return out


def _speedup_durations_exponential(durations: list[int], target_speedup: float, exponent: float) -> list[int]:
    n = len(durations)
    if n <= 1:
        return durations[:]

    out: list[int] = []
    for i, dur in enumerate(durations):
        t = i / (n - 1)
        shaped = t**exponent
        factor = 1.0 + ((target_speedup - 1.0) * shaped)
        new_dur = max(10, round(dur / factor))
        out.append(new_dur)
    return out


def main() -> None:
    print("GIF Utils")
    print("=========")

    while True:
        gif_path = Path(_prompt_non_empty("Path to input GIF: ")).expanduser()
        if gif_path.exists() and gif_path.is_file() and gif_path.suffix.lower() == ".gif":
            break
        print("Please provide an existing .gif file path.")

    frames, durations, base_info = _load_gif(gif_path)
    total = len(frames)
    if total == 0:
        raise SystemExit("Input GIF has no frames.")

    print(f"Loaded {total} frames.")
    print("Choose operation:")
    print("  1) Remove frame range")
    print("  2) Dynamic speed-up")

    op = _prompt_int("Selection (1 or 2): ", min_value=1, max_value=2)

    if op == 1:
        start = _prompt_int(f"Start frame index to remove (0..{total - 1}): ", min_value=0, max_value=total - 1)
        end = _prompt_int(f"End frame index to remove ({start}..{total - 1}): ", min_value=start, max_value=total - 1)

        out_frames, out_durations = _remove_frame_range(frames, durations, start, end)
        if not out_frames:
            raise SystemExit("Removal would delete all frames. Aborting.")

        out_path = gif_path.with_name(f"{gif_path.stem}_trimmed.gif")
        _save_gif(out_path, out_frames, out_durations, base_info)
        print(f"Saved: {out_path}")
        return

    target = _prompt_float("Target speed-up at last frame (e.g. 2.0, 4.0): ", min_value=1.0)
    print("Speed profile:")
    print("  1) Linear")
    print("  2) Exponential")
    profile = _prompt_int("Selection (1 or 2): ", min_value=1, max_value=2)

    if profile == 1:
        out_durations = _speedup_durations_linear(durations, target)
    else:
        exponent = _prompt_float("Exponent (>0), e.g. 2.0: ", min_value=0.000001)
        out_durations = _speedup_durations_exponential(durations, target, exponent)

    out_path = gif_path.with_name(f"{gif_path.stem}_speedup.gif")
    _save_gif(out_path, frames, out_durations, base_info)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
