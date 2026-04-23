#!/usr/bin/env python
"""CLI entry point for latent-population pretraining (Plan 005 / v0.3.0).

Usage (from AISCO_code repo root with aisco_env active):

    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v5_latent.py
    python subprojects/wp4_aeroshell_optimizer/v2/src/main_files/pretraining_v5_latent.py \\
        --config pretraining_latent_config.yaml \\
        --epochs 2000 \\
        --device cpu

Flags:
    --config   Config filename (basename only) to load from the v2 configs dir.
               Default: "pretraining_latent_config.yaml"
    --epochs   Override training_loop.epochs from config.
    --no-mesh  Skip per-slot mesh extraction even if output.mesh_resolution is set.
    --device   Override compute device ("cpu" or "cuda"). Default: auto-detect.
"""

from __future__ import annotations

import argparse

import torch


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
        default=None,
        help="Override training_loop.epochs from config.",
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        dest="no_mesh",
        help="Skip per-slot mesh extraction after training.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device ('cpu' or 'cuda'). Default: auto-detect.",
    )
    return parser.parse_args()


def main() -> None:
    """Run latent-population pretraining from the command line."""
    args = _parse_args()

    from subprojects.wp4_aeroshell_optimizer.v2.src.main_files.pretraining_latent import pretraining_latent  # noqa: PLC0415
    from subprojects.wp4_aeroshell_optimizer.v2.src.models.model_protocol import SupportsLatentConcat  # noqa: PLC0415
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config  # noqa: PLC0415
    from subprojects.wp4_aeroshell_optimizer.v2.src.utils.dynamic_loading import smart_model_init  # noqa: PLC0415

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

    # Device selection
    device: str = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pretraining_v5_latent] Device: {device}")

    # Load model
    model_name: str = config.get("model_name", "ConcatWIREV2")
    model, _, _, _ = smart_model_init(model_name)
    model = model.to(device)
    print(f"[pretraining_v5_latent] Loaded model: {model_name}")

    # Guard: model must support latent-concat protocol
    if not isinstance(model, SupportsLatentConcat):
        raise TypeError(
            f"Model '{model_name}' does not implement SupportsLatentConcat. "
            "Only ConcatWIREV2 (and similar latent-concat models) can be used with this CLI."
        )

    # Run pretraining
    model, Z = pretraining_latent(model, config, device)

    print(
        f"[pretraining_v5_latent] Pretraining complete. "
        f"Z shape={tuple(Z.shape)}, checkpoint: {config['output']['checkpoint_path']}"
    )


if __name__ == "__main__":
    main()
