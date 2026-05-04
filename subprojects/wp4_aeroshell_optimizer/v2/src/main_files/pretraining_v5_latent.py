#!/usr/bin/env python
"""CLI entry point for latent-population pretraining (Plan 005 / v0.3.0).

Usage (from AISCO_code repo root with aisco_env active):

    #recommended:
    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v5_latent.py --wandb --vis-sampling

    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v5_latent.py \\
        --config pretraining_latent_config.yaml \\
        --epochs 2000 \\
        --device cpu  \\
        --wandb --wandb-project AISCO-latent-pretraining \\
        --vis-sampling --make-gif

Flags:
    --config        Config filename (basename only) to load from the v2 configs dir.
                    Default: "pretraining_latent_config.yaml"
    --epochs        Override training_loop.epochs from config.
    --no-mesh       Skip per-slot mesh extraction even if output.mesh_resolution is set.
    --device        Override compute device ("cpu" or "cuda"). Default: auto-detect.
    --wandb --wandb-project AISCO-latent-pretraining enables wandb logging
    --vis-sampling  Write sampling visualizations (points + mesh PLY) every vis_cadence epochs.
    --make-gif      After training, call ply_to_gif to render a convergence GIF from the
                    sampling_vis directory.  Requires --vis-sampling to produce PLY input files.
                    Output GIFs are written to <run_dir>/pretraining_latent_gifs/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import (
    _resolve_checkpoint_path,
    pretraining_latent,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Latent-population pretraining for ConcatWIREV2 (AISCO v0.3.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="pretraining_latent_config.yaml",
        help="Config filename (basename) inside the v2 configs directory.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,  # value from config will be used
        help="Override training_loop.epochs from config.",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",  # iff flag is present: no_mesh == true
        dest="no_mesh",
        help="Skip per-slot mesh extraction after training.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device ('cpu' or 'cuda'). Default: auto-detect.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",  # iff flag is present: use_wandb == true
        dest="use_wandb",
        help="Log training metrics to Weights & Biases.",
    )
    parser.add_argument(
        "--wandb-project",
        default="AISCO-latent-pretraining",
        dest="wandb_project",
        help="W&B project name (only used with --wandb).",
    )
    parser.add_argument(
        "--vis-sampling",
        action="store_true",  # iff flag is present: vis_sampling == true
        dest="vis_sampling",
        help="Write sampling visualizations (points + mesh PLY) every vis_cadence epochs.",
    )
    parser.add_argument(
        "--make-gif",
        action="store_true",
        dest="make_gif",
        help=(
            "After training, render a convergence GIF from the sampling_vis directory. "
            "Requires --vis-sampling to be active and vis_cadence > 0 in config. "
            "Output: <run_dir>/pretraining_latent_gifs/."
        ),
    )
    parser.add_argument(
        "--tight-sigma",
        type=float,
        default=None,
        dest="tight_sigma",
        help="Override training_loop.near_surface_tight_sigma (factor x bbox_extent_mm).",
    )
    parser.add_argument(
        "--wide-sigma",
        type=float,
        default=None,
        dest="wide_sigma",
        help=(
            "Override training_loop.near_surface_wide_sigma. "
            "Setting this enables dual-band sampling (DeepSDF-style): "
            "near-surface pts are split evenly between tight and wide Gaussian bands."
        ),
    )
    return parser.parse_args()


def main() -> None:  # noqa: PLR0912
    """Run latent-population pretraining from the command line."""
    args = _parse_args()

    config = load_config(v="v2", cfg_name=args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config["training_loop"]["epochs"] = args.epochs
        print(f"[pretraining_v5_latent] Overriding epochs → {args.epochs}")

    if args.no_mesh:
        # Remove mesh extraction keys so pretraining_latent skips that block
        config.get("output", {}).pop("mesh_resolution", None)
        config.get("output", {}).pop("mesh_box_mm", None)
        print("[pretraining_v5_latent] Mesh extraction disabled (--no-mesh).")

    if args.tight_sigma is not None:
        config["training_loop"]["near_surface_tight_sigma"] = args.tight_sigma
        print(f"[pretraining_v5_latent] Overriding near_surface_tight_sigma → {args.tight_sigma}")

    if args.wide_sigma is not None:
        config["training_loop"]["near_surface_wide_sigma"] = args.wide_sigma
        print(f"[pretraining_v5_latent] Overriding near_surface_wide_sigma → {args.wide_sigma} (dual-band enabled)")

    # Device selection
    device: str = (
        args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    )  # TODO: align with rest of code
    print(f"[pretraining_v5_latent] Device: {device}")

    # Load model
    model_name: str = config.get("model_name", "ConcatWIREV2")
    model, _, _, _ = smart_model_init(model_name)
    model = model.to(device)
    print(f"[pretraining_v5_latent] Loaded model: {model_name}")

    # Guard: model must support latent-concat protocol
    if not isinstance(model, SupportsLatentConcat):
        raise TypeError(f"Model '{model_name}' does not implement SupportsLatentConcat. ")

    # Init wandb if requested
    use_wandb = False
    if args.use_wandb:
        import wandb  # noqa: PLC0415

        wandb.init(
            project=args.wandb_project,
            config={
                "model_name": model_name,
                "latent_population": config.get("latent_population"),
                "training_loop": config.get("training_loop"),
                "z_placement": config.get("z_placement", "unit_circle"),
                "z_radius": config.get("z_radius", 1.0),
            },
            name=f"latent-pretrain-{model_name}",
        )
        use_wandb = True
        if wandb.run is not None:
            print(f"[pretraining_v5_latent] W&B run: {wandb.run.url}", flush=True)
        else:
            print("[pretraining_v5_latent] W&B initialized, but no run URL found.")

    # Run pretraining
    model, Z = pretraining_latent(model, config, device, use_wandb=use_wandb, vis_sampling=args.vis_sampling)

    if use_wandb:
        wandb.finish()

    ckpt_path: Path = _resolve_checkpoint_path(config, model)
    run_dir: Path = ckpt_path.parent
    print(f"[pretraining_v5_latent] Pretraining complete. Z shape={tuple(Z.shape)}, checkpoint: {ckpt_path}")

    # --- Optional: render convergence GIFs from sampling_vis PLY sequence ---
    if args.make_gif:
        if not args.vis_sampling:
            print(
                "[pretraining_v5_latent] WARNING: --make-gif requires --vis-sampling. "
                "No PLY files were written. Skipping GIF generation."
            )
        else:
            from subprojects.wp4_aeroshell_optimizer.v2.src.sampling_visualization.ply_to_gif import (  # noqa: PLC0415
                generate_gifs,
            )

            vis_subdir: str = str(config.get("output", {}).get("visualisation_directory", "sampling_vis"))
            vis_dir: Path = run_dir / vis_subdir
            gif_dir: Path = run_dir / "pretraining_latent_gifs"
            print(f"[pretraining_v5_latent] Generating GIFs from {vis_dir} → {gif_dir}")
            try:
                generate_gifs(vis_dir, gif_dir)
                print(f"[pretraining_v5_latent] GIFs written to {gif_dir}")
            except ValueError as exc:
                print(
                    f"[pretraining_v5_latent] WARNING: GIF generation failed: {exc}. "
                    "Check that vis_cadence > 0 and PLY files exist in the sampling_vis directory."
                )


if __name__ == "__main__":
    main()
