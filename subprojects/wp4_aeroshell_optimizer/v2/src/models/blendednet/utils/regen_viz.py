"""Regenerate cross-section viz from a saved checkpoint."""

import argparse
import os
import sys

import torch

# add parent dir to path so imports work from utils/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sdf_net import BWBSDFNet
from train_sdf import _load_vtk_meshes, _pick_extreme_geometries, load_dataset, normalize_params
from viz_training import save_cross_section_figure


def main():
    p = argparse.ArgumentParser(description="regenerate viz from checkpoint")
    p.add_argument("--checkpoint", required=True, help="path to .pt checkpoint")
    p.add_argument("--data-dir", required=True, help="path to dataset dir")
    p.add_argument("--output-dir", default=None, help="output dir (default: same as checkpoint)")
    p.add_argument("--step", type=int, default=None, help="step label (default: from checkpoint)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    step = args.step or ckpt.get("step", 0)

    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    cfg["output_dir"] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    # load data
    points, sdf_gt, params, param_min, param_max, geom_ids = load_dataset(
        args.data_dir, device,
    )

    # pick viz geometries + load meshes
    vis_geom_indices = _pick_extreme_geometries(params, param_min, param_max)
    geom_index_path = os.path.join(args.data_dir, "geom_index.json")
    vtk_meshes = _load_vtk_meshes(geom_index_path, geom_ids, vis_geom_indices)

    # rebuild model
    model = BWBSDFNet(
        fourier_bands=cfg.get("fourier_bands", 10),
        hidden_dim=cfg.get("hidden_dim", 512),
        cond_dim=cfg.get("cond_dim", 9),
        n_blocks=cfg.get("n_blocks", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded step {step}")

    # generate viz
    path = save_cross_section_figure(
        model, points, params, param_min, param_max,
        vis_geom_indices, geom_ids, vtk_meshes,
        cfg, step, device,
    )
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
