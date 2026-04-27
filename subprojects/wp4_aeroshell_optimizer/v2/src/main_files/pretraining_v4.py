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
import json
import math
import sys
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from subprojects.wp3_solar_estimator.mesh_extractors.non_diff_algos.marching_cubes import MarchingCubesExtractor
from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_v1_1 import (
    _apply_override_config,
    _compute_eikonal,
    _compute_gt_target,
    _load_ply_mesh,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsModelContract
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import (
    apply_sweep_overrides,
    get_master_config,
    load_config,
    register_all_configs,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.kaolin_conversions import kaolin_to_PLY


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
    extractor = MarchingCubesExtractor(device=device, verbose=True)
    extractor.set_grid(
        bbox=[[-mesh_box, mesh_box], [-mesh_box, mesh_box], [-mesh_box, mesh_box]],
        resolution=mesh_res,
    )
    verts, faces = extractor.extract_mesh(model, isovalue=0.0, batch_size=8192)
    n_verts = verts.shape[0]
    print(f"[mesh] Extracted: {n_verts} vertices, {faces.shape[0]} faces")
    if n_verts == 0:
        print("[mesh] WARNING: 0 vertices extracted \u2014 surface not within bounding box. PLY not saved.")
    else:
        kaolin_to_PLY(verts, faces, filename=ply_path.name, mesh_dir=ply_path.parent)
        print(f"\u2713 Saved mesh to {ply_path}")


def log_model_stats(model: SupportsModelContract, epoch: int) -> None:
    if wandb.run is not None:
        layer_omegas = {}
        for name, param in model.named_parameters():
            if "omega" in name:
                layer_omegas[name] = param.detach().cpu().numpy()
        wandb.log({"pretraining/layer_omegas": wandb.Histogram(layer_omegas)}, step=epoch)

        layer_scales = {}
        for name, param in model.named_parameters():
            if "scale" in name:
                layer_scales[name] = param.detach().cpu().numpy()
        wandb.log({"pretraining/layer_scales": wandb.Histogram(layer_scales)}, step=epoch)


def main(epochs_override: int | None = None, no_mesh: bool = False) -> None:  # noqa: PLR0912, PLR0915 — training loop matches pretraining_v3.py structure; extracting sub-functions would obscure the linear flow
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
    # ── Register configs + wandb init + sweep overrides ─────────────────────
    # All model configs must be registered before apply_sweep_overrides so that
    # sweep params targeting e.g. models/WIRE_v3.hidden_features are patchable,
    # and smart_model_init's load_config call returns the patched version.
    models_json_path = Path.home() / "AISCO_code" / "subprojects" / "wp4_aeroshell_optimizer" / "v2" / "configs" / "models.json"
    models_inventory = json.loads(models_json_path.read_text())
    active_configs: dict[str, str] = {"pretraining_v4": "pretraining_v4_config.yaml"}
    for model_info in models_inventory["models"].values():
        cfg_file: str | None = model_info.get("config_file")
        if cfg_file:
            active_configs[Path(cfg_file).stem] = cfg_file
    register_all_configs({"active_configs": active_configs}, v="v2")

    pt_cfg_for_wandb = load_config(v="v2", cfg_name="pretraining_v4_config.yaml")
    use_wandb: bool = pt_cfg_for_wandb.get("wandb", {}).get("use_wandb", False)
    if use_wandb:
        wandb.init(
            entity=pt_cfg_for_wandb["wandb"]["entity_name"],
            project=pt_cfg_for_wandb["wandb"]["project_name"],
        )

    if wandb.run is not None:
        apply_sweep_overrides(dict(wandb.config))
        wandb.config.update(get_master_config(), allow_val_change=True)

    # ── Load config and model after registry is done──────────────────────────
    config = load_config(v="v2", cfg_name="pretraining_v4_config.yaml")
    torch.manual_seed(config["random_seed"])  # Set random seed to ensure reproducibility of pretraining runs
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
    bb_center = bb_half_extent = None
    if pretraining_shape == "ply":
        ply_verts_t, ply_faces_t = _load_ply_mesh(config, device)
        bb_min = ply_verts_t.min(dim=0).values
        bb_max = ply_verts_t.max(dim=0).values
        bb_center = (bb_min + bb_max) / 2.0
        # isotropic half-extent with 1.5x padding so near-surface samples stay well inside the box
        bb_half_extent = ((bb_max - bb_min) * 0.5 * 1.5).max()
        print(f"[v4] PLY bbox: {bb_min.tolist()} → {bb_max.tolist()}, sampling half-extent ±{bb_half_extent.item():.0f} mm")

    # ── Training hyperparameters ─────────────────────────────────────────────
    learning_rate: float = config["training_loop"]["learning_rate"]
    num_epochs: int = config["training_loop"]["epochs"]
    batch_size: int = config["training_loop"]["random_batch_size"]
    print_interval: int = config["training_loop"]["print_interval"]
    eikonal_weight: float = config["training_loop"]["eikonal_weight"]

    # ── Output settings ──────────────────────────────────────────────────────
    output_cfg: dict = config["output"]
    base_output_dir = Path(output_cfg["checkpoint_path"])
    if wandb.run is not None:
        prefix = wandb.run.sweep_id if wandb.run.sweep_id is not None else ""
        run_dir = base_output_dir / prefix / wandb.run.name if prefix else base_output_dir / wandb.run.name
    else:
        run_dir = base_output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_pth = run_dir / f"{model_name}.pth"
    mesh_res: int = output_cfg["mesh_resolution"]
    mesh_box: int = output_cfg["mesh_box_mm"]
    if pretraining_shape == "ply":
        assert bb_half_extent is not None
        # scale extraction box 10% larger than the fitted mesh region
        mesh_box = int(bb_half_extent * 1.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min")

    print(
        f"[v4] Starting pretraining: epochs={num_epochs}, batch={batch_size}, lr={learning_rate}, eikonal_weight={eikonal_weight}"
    )

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        if pretraining_shape == "ply":
            assert bb_center is not None and bb_half_extent is not None and ply_verts_t is not None
            n_surface = batch_size // 2
            n_uniform = batch_size - n_surface
            uniform_pts = bb_center + (torch.rand(n_uniform, 3, device=device) - 0.5) * (2 * bb_half_extent)
            surf_idx = torch.randint(0, ply_verts_t.shape[0], (n_surface,), device=device)
            # sigma = 5% of half-extent concentrates samples in the near-surface band where SDF varies fastest
            surface_pts = ply_verts_t[surf_idx] + torch.randn(n_surface, 3, device=device) * (bb_half_extent * 0.05)
            input_data = torch.cat([uniform_pts, surface_pts], dim=0)
        else:
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
        scheduler.step(total_loss, epoch=epoch)

        if use_wandb and wandb.run is not None:
            wandb.log(
                {
                    "pretraining/total_loss": total_loss.item(),
                    "pretraining/mse_loss": mse_loss.item(),
                    "pretraining/eikonal_loss": eikonal.item(),
                    "pretraining/lr": scheduler.get_last_lr()[0],
                },
                step=epoch,
            )

        if epoch % print_interval == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:5d} | total={total_loss.item():.6f} "
                f"| mse={mse_loss.item():.6f} "
                f"| eikonal={eikonal.item():.6f} "
                f"| lr={lr_now:.2e}"
            )

    # ── Save checkpoint ──────────────────────────────────────────────────────
    torch.save(model.state_dict(), checkpoint_pth)
    print(f"\u2713 Saved {model_name} pretrained model to {checkpoint_pth}")

    # ── Optional post-training mesh extraction ───────────────────────────────
    if no_mesh:
        print("[mesh] Skipped (--no-mesh)")
    else:
        try:
            model.eval()
            _extract_mesh(model, checkpoint_pth, device, mesh_res=mesh_res, mesh_box=mesh_box)
        except Exception as exc:
            print(
                f"[mesh] ERROR: {exc}. Checkpoint is intact at {checkpoint_pth}.",
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
    args, _ = parser.parse_known_args()  # ignore wandb sweep flags passed as CLI args
    main(epochs_override=args.epochs, no_mesh=args.no_mesh)
