"""Rename silvretta run folders to their trailing run number, zero-padded for sorting.

Run from anywhere; targets the absolute path on /cluster/work.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path("/cluster/work/fuge/AISCO/AISCOModel_Pretraining/silvretta")
TRAILING_INT = re.compile(r"(-?\d+)$")


def main(dry_run: bool = True) -> None:
    if not ROOT.is_dir():
        raise SystemExit(f"Not a directory: {ROOT}")

    entries = [p for p in ROOT.iterdir() if p.is_dir()]

    numbers: list[int] = []
    src_to_num: dict[Path, int] = {}
    for p in entries:
        m = TRAILING_INT.search(p.name)
        if not m:
            raise ValueError(f"No trailing number in: {p.name}")
        # Leading "-" in names like "-11" is a separator, not a sign.
        n = abs(int(m.group(1)))
        numbers.append(n)
        src_to_num[p] = n

    if len(set(numbers)) != len(numbers):
        raise ValueError(f"Duplicate run numbers: {sorted(numbers)}")

    width = max(2, len(str(max(numbers))))
    targets = {src: ROOT / f"{n:0{width}d}" for src, n in src_to_num.items()}

    print(f"Renaming {len(targets)} folders in {ROOT} (dry_run={dry_run}):")
    for src, dst in sorted(targets.items(), key=lambda kv: kv[1].name):
        marker = "  (no change)" if src == dst else ""
        print(f"  {src.name!r:30s} -> {dst.name!r}{marker}")

    if dry_run:
        print("\nDry run only. Re-run with --apply to perform the renames.")
        return

    # Two-phase rename via temp names to avoid collisions (e.g. existing "-11" -> "11").
    tmp_map = {src: ROOT / f".__tmp_rename__{i}" for i, src in enumerate(targets)}
    for src, tmp in tmp_map.items():
        src.rename(tmp)
    for src, tmp in tmp_map.items():
        tmp.rename(targets[src])
    print("Done.")


if __name__ == "__main__":
    main(dry_run="--apply" not in sys.argv)
