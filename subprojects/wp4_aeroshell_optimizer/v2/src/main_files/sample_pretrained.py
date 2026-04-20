#!/usr/bin/env python
"""
Sample a mesh from any SDF model and save as PLY.

Usage:
    python -m subprojects.wp4_aeroshell_optimizer.v2.src.main_files.sample_pretrained.py
        --model /cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_dataset/bunny_pretrained.pth
        --out /cluster/work/fuge/AISCO/pretraining_output_sampled/bunny_sampled2.ply
        --res 64
        --box 1000
"""

import argparse
from pathlib import Path

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.models.WIRE_v2 import WIREV2
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.kaolin_conversions import (
    get_diff_mesh,
    kaolin_to_PLY,
)


def sample(model_path: str, out_path: str, res: int, box: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_config = load_config(v="v2", cfg_name="WIRE_v2.yaml")
    model = WIREV2(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    box_dim = [-box, box, -box, box, -box, box]
    print(f"Extracting mesh: res={res}, box=±{box}mm ...")
    verts, faces = get_diff_mesh(model, grid_res=res, box_dim=box_dim, device=device)
    print(f"  vertices: {verts.shape[0]}, faces: {faces.shape[0]}")

    out = Path(out_path)
    kaolin_to_PLY(verts, faces, filename=out.name, mesh_dir=out.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_dataset/bunny_pretrained3.pth",
        help="Path to .pth checkpoint",
    )  # required=True,
    parser.add_argument(
        "--out", default="/cluster/work/fuge/AISCO/pretraining_output_sampled/bunny_sampled3.ply", help="Output PLY path"
    )
    parser.add_argument("--res", type=int, default=64, help="Grid resolution (FlexiCubes)")
    parser.add_argument("--box", type=int, default=1000, help="Bounding box half-size in mm")
    args = parser.parse_args()
    sample(args.model, args.out, args.res, args.box)
