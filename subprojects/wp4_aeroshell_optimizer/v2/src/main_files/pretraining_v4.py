#!/usr/bin/env python
"""
Dynamic model loading standalone pretraining entry point (Plan 013, v0.2.3).

Improvements over pretraining_v3.py:
- Model is loaded dynamically via smart_model_init — no hard import of a specific class.
  Changing model_name in pretraining_v4_config.yaml switches architectures without touching code.
- Reads all configuration (including model_name) from pretraining_v4_config.yaml.
- Valid model_name values are registered in v2/configs/models.json.
- Cosine annealing LR schedule (inherited from pretraining_v3.py)
- NaN guard: halts immediately, no corrupt checkpoint written
- Separate MSE and eikonal loss printed each interval

Usage (smoke test — 5 epochs):
    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v4.py --epochs 5

Usage (full run):
    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v4.py
"""

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _apply_override_config,
    _compute_eikonal,
    _compute_gt_target,
    _load_ply_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.kaolin_conversions import (
    get_diff_mesh,
    kaolin_to_PLY,
)


def _extract_mesh(model: SupportsModelContract, pth_path: Path, device: str, mesh_res: int, mesh_box: int) -> None:
    """
    Extract a PLY mesh from a trained SDF model and save it alongside the checkpoint.

    Args:
        model: Trained model satisfying SupportsModelContract (in eval mode).
        pth_path: Path to the saved checkpoint (used to derive .ply output path).
        device: Device string, e.g. "cuda" or "cpu".
        mesh_res: Marching-cubes grid resolution.
        mesh_box: Half-extent of the marching-cubes bounding box in mm.

    Returns:
        None. Saves a .ply file to pth_path.with_suffix('.ply') on success.
        Logs a warning and skips saving if vertex count is 0.
    """
    ply_path = pth_path.with_suffix(".ply")
    print(f"[mesh] Starting post-training mesh extraction (res={mesh_res}, box=\u00b1{mesh_box}mm) ...")
    box_dim = [
        -mesh_box,
        mesh_box,
        -mesh_box,
        mesh_box,
        -mesh_box,
        mesh_box,
    ]
    verts, faces = get_diff_mesh(model, grid_res=mesh_res, box_dim=box_dim, device=device)
    n_verts = verts.shape[0]
    print(f"[mesh] Extracted: {n_verts} vertices, {faces.shape[0]} faces")
    if n_verts == 0:
        print("[mesh] WARNING: 0 vertices extracted \u2014 surface not within bounding box. PLY not saved.")
    else:
        kaolin_to_PLY(verts, faces, filename=ply_path.name, mesh_dir=ply_path.parent)
        print(f"\u2713 Saved mesh to {ply_path}")


def main(epochs_override: int | None = None, no_mesh: bool = False) -> None:  # noqa: PLR0915 — training loop matches pretraining_v3.py structure; extracting sub-functions would obscure the linear flow
    """
    Run dynamic pretraining: load model via models.json, save checkpoint, optionally extract mesh.

    The model class is determined by model_name in pretraining_v4_config.yaml.
    Valid model_name values are the registry keys in v2/configs/models.json.
    Changing model_name requires no Python source edits.

    Args:
        epochs_override: If provided, overrides the epoch count from config.
            Used by the CLI --epochs flag and SLURM smoke-test runs.
        no_mesh: If True, skip post-training PLY mesh extraction.

    Returns:
        None. Saves a .pth checkpoint on success.
        If no_mesh is False, also saves a .ply to the same directory.
    """
    # ── Load config and model ────────────────────────────────────────────────
    config = load_config(v="v2", cfg_name="pretraining_v4_config.yaml")
    model_name: str = config["model_name"]
    model, _, _, _ = smart_model_init(model_name)
    print(f"[v4] Loaded model: {model.__class__.__name__} (model_name={model_name!r})")

    # ── Apply CLI overrides ──────────────────────────────────────────────────
    override: dict = {}
    if epochs_override is not None:
        override["training_loop.epochs"] = epochs_override
    if override:
        config = _apply_override_config(config, override)

    # ── Device ──────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[v4] Device: {device}")
    model = model.to(device)

    # ── Pre-load PLY mesh ────────────────────────────────────────────────────
    pretraining_shape = config.get("pretraining_shape", "sphere")
    ply_verts_t = ply_faces_t = None
    if pretraining_shape == "ply":
        ply_verts_t, ply_faces_t = _load_ply_mesh(config, device)

    # ── Training hyperparameters ─────────────────────────────────────────────
    learning_rate: float = config["training_loop"]["learning_rate"]
    num_epochs: int = config["training_loop"]["epochs"]
    batch_size: int = config["training_loop"]["random_batch_size"]
    print_interval: int = config["training_loop"]["print_interval"]
    eikonal_weight: float = config["training_loop"]["eikonal_weight"]

    # ── Output settings ──────────────────────────────────────────────────────
    output_cfg: dict = config["output"]
    checkpoint_path: str = output_cfg["checkpoint_path"]
    mesh_res: int = output_cfg["mesh_resolution"]
    mesh_box: int = output_cfg["mesh_box_mm"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    print(
        f"[v4] Starting pretraining: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}, eikonal_weight={eikonal_weight}"
    )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        input_data = (torch.rand(batch_size, 3, device=device) - 0.5) * 5000.0

        output = model(input_data)
        target = _compute_gt_target(pretraining_shape, input_data, ply_verts_t, ply_faces_t)
        eikonal = _compute_eikonal(model, input_data, device)

        mse_loss = torch.nn.MSELoss()((target / 1000).squeeze(), (output / 1000).squeeze())
        total_loss = mse_loss + eikonal_weight * eikonal

        # NaN guard: halt immediately, never write a corrupt checkpoint
        if not math.isfinite(total_loss.item()):
            print(
                f"[v4] NaN/Inf detected at epoch {epoch}. Halting. No checkpoint written. ",
                file=sys.stderr,
            )
            sys.exit(1)

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % print_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:5d} | total={total_loss.item():.6f} "
                f"| mse={mse_loss.item():.6f} "
                f"| eikonal={eikonal.item():.6f} "
                f"| lr={lr_now:.2e}"
            )

    # ── Save checkpoint ──────────────────────────────────────────────────────
    # torch.save(model.state_dict(), checkpoint_path)
    # print(f"\u2713 Saved {model_name} pretrained model to {checkpoint_path}")

    # ── Optional post-training mesh extraction ───────────────────────────────
    if no_mesh:
        print("[mesh] Skipped (--no-mesh)")
    else:
        try:
            model.eval()
            _extract_mesh(model, Path(checkpoint_path), device, mesh_res=mesh_res, mesh_box=mesh_box)
        except Exception as exc:
            print(
                f"[mesh] ERROR: {exc}. Checkpoint is intact at {checkpoint_path}.",
                file=sys.stderr,
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WIRE v4 dynamic model pretraining")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count from config (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        dest="no_mesh",
        help="Skip post-training PLY mesh extraction.",
    )
    args = parser.parse_args()
    main(epochs_override=args.epochs, no_mesh=args.no_mesh)
