"""Post-process gradient conflict data into chord diagrams and GIF animation.

Usage:
    python postprocess_gradient_conflict.py <conflict_dir> [--fps 5] [--output output.gif]
"""

import argparse
from pathlib import Path

import imageio
import numpy as np

from subprojects.wp4_aeroshell_optimizer.v2.src.visuals.gradient_conflict_chord import plot_chord_diagram


def build_conflict_gif(conflict_dir: Path, output_path: Path, fps: int = 5) -> None:
    """
    Build GIF from per-epoch gradient conflict .npz files.

    Args:
        conflict_dir: Directory containing epoch_*.npz files
        output_path: Path to save output GIF
        fps: Frames per second for GIF
    """
    conflict_dir = Path(conflict_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(conflict_dir.glob("epoch_*.npz"))
    if not npz_files:
        print(f"No epoch_*.npz files found in {conflict_dir}")
        return

    print(f"Found {len(npz_files)} epochs to visualize")

    png_dir = output_path.parent / "chord_frames"
    png_dir.mkdir(parents=True, exist_ok=True)

    png_files = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        cos_sim = data["cos_sim"]
        grad_norms = data["grad_norms"]
        loss_names = list(data["loss_names"])
        epoch = int(data["epoch"])

        png_path = png_dir / f"epoch_{epoch:04d}.png"
        plot_chord_diagram(cos_sim, grad_norms, loss_names, epoch, png_path)
        png_files.append(str(png_path))
        print(f"Rendered epoch {epoch}")

    print(f"Stitching {len(png_files)} frames into GIF...")
    images = [imageio.imread(f) for f in png_files]
    imageio.mimsave(output_path, images, fps=fps)
    print(f"Saved GIF to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create conflict GIF from .npz files")
    parser.add_argument("conflict_dir", type=Path, help="Directory containing epoch_*.npz files")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second for GIF")
    parser.add_argument(
        "--output", type=Path, default=None, help="Output GIF path (default: <conflict_dir>/conflict_evolution.gif)"
    )

    args = parser.parse_args()
    output = args.output or (args.conflict_dir / "conflict_evolution.gif")
    build_conflict_gif(args.conflict_dir, output, fps=args.fps)
