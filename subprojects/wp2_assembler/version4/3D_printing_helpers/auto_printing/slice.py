#!/usr/bin/env python3
"""
Slice a 3MF file for BambuLab X1C using Bambu Studio CLI.

Usage:
    python slice.py [input.3mf] [--output dir]

Defaults to slicing 3DBenchy.3mf located next to this script.
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

APPIMAGE = Path("/home/aisco/Downloads/Bambu_Studio_ubuntu-24.04_V02.05.01.52.AppImage")
BENCHY_3MF = Path(__file__).parent / "3DBenchy.3mf"

# Profiles bundled inside Bambu Studio (relative to resources/profiles/BBL/)
MACHINE_PROFILE = "machine/Bambu Lab X1 Carbon 0.4 nozzle.json"
PROCESS_PROFILE = "process/0.20mm Standard @BBL X1C.json"
FILAMENT_PROFILE = "filament/Generic PLA High Speed @BBL X1C.json"

CACHE_DIR = Path.home() / ".cache" / "bambu-studio-extracted"


def get_profiles_dir() -> Path:
    """
    Return the path to the bundled Bambu Studio BBL profiles directory.

    Checks for a currently mounted AppImage first (fast), then falls back to
    extracting the AppImage to a cache directory (one-time, ~30 s).
    """
    # 1. Look for a currently mounted AppImage (Bambu Studio already running)
    mounts = glob.glob("/tmp/.mount_Bambu_*")
    for mount in mounts:
        candidate = Path(mount) / "resources" / "profiles" / "BBL"
        if candidate.is_dir():
            return candidate

    # 2. Use cached extraction
    extracted_profiles = CACHE_DIR / "resources" / "profiles" / "BBL"
    if extracted_profiles.is_dir():
        return extracted_profiles

    # 3. Extract AppImage to cache (one-time operation)
    print(f"Extracting AppImage to {CACHE_DIR} (one-time setup, please wait)…")
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(APPIMAGE), "--appimage-extract"],
        cwd=str(CACHE_DIR.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    squashfs = CACHE_DIR.parent / "squashfs-root"
    if squashfs.is_dir():
        squashfs.rename(CACHE_DIR)
    elif result.returncode != 0:
        print("AppImage extraction failed:", result.stderr, file=sys.stderr)
        sys.exit(1)

    if not extracted_profiles.is_dir():
        print(f"Error: profiles not found after extraction at {extracted_profiles}", file=sys.stderr)
        sys.exit(1)

    return extracted_profiles


def slice_3mf(input_3mf: Path, output_dir: Path) -> int:
    """Slice *input_3mf* with X1C settings and write output to *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_dir = get_profiles_dir()
    machine_json = profiles_dir / MACHINE_PROFILE
    process_json = profiles_dir / PROCESS_PROFILE
    filament_json = profiles_dir / FILAMENT_PROFILE

    for p in (machine_json, process_json, filament_json):
        if not p.exists():
            print(f"Error: profile not found: {p}", file=sys.stderr)
            sys.exit(1)

    cmd = [
        str(APPIMAGE),
        "--load-settings", f"{machine_json};{process_json}",
        "--load-filaments", str(filament_json),
        "--slice", "0",
        "--outputdir", str(output_dir),
        str(input_3mf),
    ]

    print("Slicing:", input_3mf.name)
    print("Printer: BambuLab X1 Carbon, 0.4 mm nozzle")
    print("Process:", PROCESS_PROFILE.rsplit("/", maxsplit=1)[-1].replace(".json", ""))
    print("Filament:", FILAMENT_PROFILE.rsplit("/", maxsplit=1)[-1].replace(".json", ""))
    print("Output dir:", output_dir)
    print()

    result = subprocess.run(cmd, text=True, check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slice a 3MF file for BambuLab X1C using Bambu Studio CLI."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(BENCHY_3MF),
        help="Path to the input .3mf file (default: 3DBenchy.3mf next to this script)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for sliced files (default: <input_dir>/sliced_output)",
    )
    args = parser.parse_args()

    if not APPIMAGE.exists():
        print(f"Error: Bambu Studio AppImage not found at {APPIMAGE}", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output).resolve() if args.output else input_path.parent / "sliced_output"

    rc = slice_3mf(input_path, output_path)
    if rc == 0:
        print("\nDone. Sliced files written to:", output_path)
    else:
        print(f"\nBambu Studio exited with code {rc}", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
