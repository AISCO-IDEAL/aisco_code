# blendednet — conditional 3D SDF for BWB geometries

Neural SDF trained on 999 BlendedNet BWB shapes (9 design params → signed distance field).

## Training

```bash
# quick dry run
python train_sdf.py --steps 500 --scratch /tmp/runs

# full run with AdamW + AMP + validation
python train_sdf.py --scratch /tmp/runs --optimizer adamw --lr 8e-5 --val-split 0.2 --steps 500000 --amp --batch 128
```

`--scratch <dir>` puts checkpoints in `<dir>/bwb_sdf/` and wandb cache in `<dir>/wandb/`.

## Files

| File | Description |
|---|---|
| `sdf_net.py` | `BWBSDFNet` — Fourier(xyz) + 9 params → SDF (configurable width/depth) |
| `config.py` | default hyperparameters |
| `train_sdf.py` | training loop (offline, precomputed data) |
| `viz_training.py` | 8×4 cross-section visualization (3 spanwise XZ + 1 top-down XY) |
| `precompute_sdf.py` | builds `data/train_dataset.pt` from VTK meshes |
| `gpu_sdf.py` | GPU-accelerated mesh SDF (used by precompute) |
| `utils/regen_viz.py` | regenerate viz from checkpoint with current viz code |
| `utils/viz_sdf_interactive.py` | interactive slider viz (local, needs display) |
| `utils/` | inspection/validation scripts |

## CLI flags

```
--hidden-dim 512       # network width
--fourier-bands 10     # Fourier encoding bands
--n-blocks 3           # layers per block (total depth = 2*n_blocks)
--activation gelu      # gelu or siren
--lr 1e-4              # initial learning rate (linear decay to 0)
--optimizer adam        # adam or adamw
--loss-type relative_mse  # relative_mse, absolute_mse, l1, clamped_l1, weighted_mse
--clamp-delta 0.1      # clamp threshold (clamped_l1 only)
--normalize-xyz        # per-axis XYZ normalization to [-1,1]
--val-split 0.2        # fraction of geoms held out for validation (0=off)
--data-dir <path>      # path to dir containing train_dataset.pt
--amp                  # mixed-precision training (fp16 forward, fp32 eikonal)
```

## Data

Not tracked in git. Regenerate with `precompute_sdf.py` (~6 min, 128 cores).

| Version | Pts/geom | Size | Location |
|---|---|---|---|
| v1 (30k) | 30k | 458MB | `blendednet/data/train_dataset.pt` |
| v2 (200k) | 200k | 6GB | `$SCRATCH/blendednet_raw/precomputed_v2/` |
| **v3 (750k)** | **750k** | **11.4GB** | **`$SCRATCH/blendednet_raw/precomputed_v3/`** |

**v3 sampling bands** (750k total):

| Band | Count | σ / method | Purpose |
|---|---|---|---|
| Near-surface | 250k | σ=0.01, normal offset | Zero-level set |
| Mid-field | 100k | σ=0.1, isotropic | Transition region |
| Far-field uniform | 50k | bbox_pad=10.0 | Kill spurious Fourier zeros |
| Gaussian far-field | 30k | σ=0.8×extent | Interior + mid-range coverage |
| Curvature-weighted | 60k | σ=0.005 | High-curvature regions |
| Wing-tip | 40k | σ=0.003 | High-|y| region |
| Edge LE/TE | 220k | σ=0.001 | Leading/trailing edge (new) |

**⚠ Scratch auto-deletes after 15 days.** Keep scripts, regenerate when needed.

## Architecture

- **Fourier + GELU** (default): NeRF-style positional encoding → 2-block MLP with skip connection. 1.39M params at hidden_dim=512, n_blocks=3.
- **SIREN** variant: sin activations, no Fourier encoding. Same skip architecture.
- **Y-symmetry**: hardcoded `|y|` in forward pass — BWB shapes are symmetric about XZ plane.
- **Condition injection**: 9 design params normalized to [-1,1], concatenated with spatial features.
- Eikonal regularization: (|∇SDF| - 1)² on 25% subsample, λ=0.1.

## Ablation results (30k steps, 2026-03-02)

All runs: Y-symmetry on, Fourier+GELU, hidden=512, n_blocks=3, lr=1e-4 linear decay.

| Run | MSE | Eikonal | Sign acc | Key flag |
|---|---|---|---|---|
| baseline | 0.0438 | 0.110 | 98.7% | — |
| normxyz | 0.0419 | 0.114 | 98.7% | `--normalize-xyz` |
| clamped_l1 | 0.0014 | 0.007 | 97.1% | `--loss-type clamped_l1` |
| adamw | 0.0423 | 0.107 | **98.9%** | `--optimizer adamw` |
| normxyz+clampl1 | 0.0018 | 0.017 | 97.0% | both |
| normxyz+adamw | 0.0417 | 0.116 | 98.7% | both |

**Takeaways:**
- **AdamW**: best sign accuracy (98.9%), slightly lower eikonal. Clear winner.
- **Normalize XYZ**: marginal MSE improvement, no sign acc gain. Not worth the complexity.
- **Clamped L1**: dramatically reduces eikonal (0.007 vs 0.110) but sign acc drops to 97.1% — ignores far-field by design. MSE not directly comparable (different loss scale).
- Combos don't beat single ablations. Effects are roughly independent.

## Decisions log (2026-03-10)

### AMP training
- `torch.amp.autocast('cuda')` wraps forward + SDF loss only
- Eikonal loss (`autograd.grad` with `create_graph=True`) stays in fp32 — mixed precision causes gradient instability there
- `GradScaler` handles loss scaling; `unscale_` before grad clipping
- Expected ~2x speedup on RTX 4090 Tensor Cores

### fp16 inference
- `load_model(device, half=True)` converts model to fp16
- `forward()` auto-casts inputs to match model dtype
- Sign accuracy (what matters most) unaffected by fp16

### Far-field spurious zero crossings
- **Cause**: Fourier encoding is periodic → model oscillates where no training signal exists
- **Fix**: `bbox_pad` 0.5 → 10.0 (20x). Uniform far-field in much larger domain.
- **Rejected**: Gaussian far-field with large σ (8.0×extent) — extreme outliers at 100s of units dominated the loss. Reduced to σ=0.8 for interior coverage instead.
- **Also rejected**: far-field regularization loss — more data is simpler and sufficient

### Wingtip / leading-trailing edge accuracy
- **Problem**: high curvature + thin geometry → SDF sign changes over tiny distance, hard to learn
- **Fix**: new `sample_edge_points()` sampler — detects LE/TE by combining triangle curvature with chordwise (x) extremes, uses very tight sigma (0.001) normal offset
- 220k edge points (29% of dataset) — this is the biggest single band
- Previous attempts: tip sampling (high |y|) + curvature weighting helped but not enough — LE/TE are the specific bottleneck

### Dataset v3 design (2026-03-10)
- 750k pts/geom, 999 geoms, 11.4GB
- Heavy near-surface + edge bias (67.6% of points have |sdf| < 0.01)
- 38.4% inside, 61.6% outside (geometry is thin → hard to get interior points)
- Gaussian σ=0.8 added specifically for interior coverage
- Precompute: 128 CPU cores, ~6 min

### Training plan
- 500k steps, AdamW, AMP, batch 128, 20% val split, linear LR decay
- Data: v3 (750k pts/geom) on scratch
- Scratch auto-deletes after 15 days — regenerate with `precompute_sdf.py` when needed

Wandb project: [glue2-playground](https://wandb.ai/taebersold-eth-zurich/glue2-playground)
