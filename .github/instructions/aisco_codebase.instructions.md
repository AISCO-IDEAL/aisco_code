---
description: "Use when working on AISCO_code: adding models, datasets, trainers, layers, preprocessing scripts, configs, or any code in subprojects/wp1_cfd_ai (GAOT2). Covers architecture patterns, data formats, config conventions, naming, WP structure, and CFD domain knowledge."
applyTo: "subprojects/**/*.py, subprojects/**/*.yaml, dataset/**/*.py, src/**/*.py"
---

# AISCO_code Codebase Instructions

## Project Overview

AISCO = AI-based Surrogate modeling In aerodynamic Simulations and design Optimization.
A research monorepo organized into workpackages (WP):

| WP | Directory | Purpose |
|----|-----------|---------|
| WP1 | `subprojects/wp1_cfd_ai/` | CFD surrogate models (GAOT2 — flagship) |
| WP2 | `subprojects/wp2_assembler/` | CAD parametric assembly (CadQuery) |
| WP3 | `subprojects/wp3_solar_estimator/` | Solar irradiance estimation |
| WP4 | `subprojects/wp4_aeroshell_optimizer/` | Aerodynamic shape optimization |
| WP7 | `subprojects/wp7_gui/` | PyQt6 + PyVista visualization GUI |

**The GAOT2 model** (`subprojects/wp1_cfd_ai/AISCO_GAOT/GAOT2/`) is the primary focus. It is the Geometry-Aware Operator Transformer for 3D CFD field prediction.

---

## GAOT2 Architecture: Encode → Process → Decode

The model maps an irregular CFD surface mesh to scalar (force, Cd) and optionally field (pressure coefficient Cp) outputs.

```
Irregular mesh nodes (pos, c, x)
        ↓  MAGNOEncoder (GNO: radius/knn graph → IntegralTransform)
Fixed latent grid [D×H×W, C]
        ↓  ViT Processor (patchify → GroupQueryFlashAttention × L → unpatchify)
Fixed latent grid [D×H×W, C]
        ↓  AttnPool + MultiHeadDecoder  →  {force, cd, ...} scalars
        └─ MAGNODecoder (optional)      →  pressure field at query nodes
```

### Key files

| Component | File |
|-----------|------|
| Top-level model | `src/model/gaot_3d.py` — `GAOT3D` |
| Encoder/Decoder layers | `src/model/layers/magno.py` — `MAGNOEncoder`, `MAGNODecoder` |
| Attention | `src/model/layers/attn.py` — `GroupQueryFlashAttention`, `TransformerBlock` |
| Kernel/GNO | `src/model/layers/integral_transform.py` — `IntegralTransform` |
| Geometry embedding | `src/model/layers/geoembed.py` — `GeometricEmbedding` |
| Scalar conditioning | `src/model/layers/scalar_encoder.py` — `ScalarEncoder` |
| MLP utilities | `src/model/layers/mlp.py` — `LinearChannelMLP`, `ChannelMLP` |
| Trainer base | `src/trainer/base.py` — `TrainerBase` |
| Trainer (static 3D) | `src/trainer/stat.py` — `StaticTrainer3D` |
| W&B logger | `src/trainer/logger.py` — `WandbTracker` |
| Dataset | `src/dataloader/pyg_datasets.py` — `VTKMeshDataset` |
| Batch collation | `src/dataloader/collate_functions.py` — `UnifiedCollateFunction` |
| Data transforms | `src/dataloader/pyg_transforms.py` — `RescalePositionNew` |
| Dataset metadata registry | `src/dataloader/metadata.py` — `DATASET_METADATA` |
| Entry point | `main.py` |

---

## Data Format: PyTorch Geometric `Data` / `Batch`

Every preprocessed sample is a `torch_geometric.data.Data` object stored as `.pt`:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `pos` | `[N, 3]` | Surface mesh node positions (meters) |
| `x` | `[N]` or `[N, D]` | Node features (primary: Cp after squeeze) |
| `c` | `[N, 3]` | Outward unit surface normals |
| `face` | `[3, F]` | Triangle indices (fan-triangulated) |
| `cd` | `[1]` | Drag coefficient |
| `force` | `[1]` | Drag force (NaN for Ahmed, BWB) |
| `v_inf` | `[1]` | Freestream velocity |
| `conditioning` | `[1, 7]` | `[v_inf, extent_x, extent_y, extent_z, aspect_x, aspect_y, aspect_z]` |
| `geom_extent` | `[3]` | Bounding box extents |
| `geom_aspect` | `[3]` | Aspect ratios |
| `filename` | `str` | Sample identifier |

**Pressure coefficient:** `Cp = (p - p_ref) / (0.5 * ρ * v_inf²)` where `ρ = 1.225 kg/m³` (air).

**Euler-specific:** x-coordinate and normals are negated (nose at −x) to match DrivAer orientation.

**Normals in training:** `group_c=None` for most datasets (normals stored but not used as input). Blended Wing Body uses `group_c="c"` (normals ARE input features).

### Default encoder input

`magno.encoder_feature_attr: ["pos", "c"]` → input is `[pos (3D) | normals (3D)]` = 6-dimensional.

---

## MAGNO: Graph Neural Operator

MAGNO (Multiscale Attentional Graph Neural Operator) builds **bipartite graphs** between physical mesh nodes and latent grid tokens.

- **Latent tokens**: a fixed uniform 3D grid `[D, H, W]` (default `[32, 32, 16]`) registered as a model buffer in normalized domain `[-1, 1]³`.
- **Encoder direction**: mesh nodes → latent tokens (each latent token aggregates nearby mesh nodes).
- **Decoder direction**: latent tokens → query nodes (each query node aggregates nearby latent tokens).
- **Graph building** happens in `UnifiedCollateFunction` at collate time (online), or can be precomputed.
- **`IntegralTransform`** computes the kernel-weighted aggregation: relative position → MLP → scatter_sum per query node.

### Neighbor strategies

| Strategy | Description |
|----------|-------------|
| `"radius"` | Radius ball query |
| `"knn"` | k-nearest neighbor |
| `"bidirectional"` | Union of radius + knn (default) |
| `"reverse"` | Use reversed encoder edges for decoder |

### Multi-scale support

`magno.scales: [1.0, 2.0]` builds graphs at multiple radii and aggregates features.

---

## Configuration System

Configs use **YAML + OmegaConf**. All configs live in `config/` (main and experimental variants).

### Top-level config structure

```yaml
setup:
  name: "experiment_name"
  device: "cuda:0"          # or "cpu"
  dtype: "bfloat16"         # float32 | float64 | bfloat16
  trainer_name: "static3d"
  distributed: false
  seed: 42
  wandb: true
  wandb_project: "WP1_default"
  wandb_entity: "aisco"

model:
  name: "gaot_3d"
  args:
    input_size: 6           # pos(3) + normals(3)
    output_size: 1
    latent_tokens: [32, 32, 16]
    magno:
      gno_radius: 0.066
      neighbor_strategy: "bidirectional"
      lifting_channels: 32
      scales: [1.0]
      encoder_feature_attr: ["pos", "c"]
    transformer:
      hidden_size: 256
      num_layers: 10
      patch_size: 2
      positional_embedding: "rope"
      attn_config:
        num_heads: 8
        atten_dropout: 0.1
    decoder:
      hidden_features: 256
      output_heads:
        force: 1
        cd: 1

dataset:
  name: "euler_pressure_field_new"
  base_path: "/cluster/work/fuge/AISCO/..."
  batch_size: 6
  training_strategy: "neural_field"  # supervised | neural_field
  subsample_nodes: null

optimizer:
  name: "adamw"
  args:
    lr: 1.0e-3
    weight_decay: 1.0e-4
```

**CLI overrides** use dot-notation: `python main.py -c config.yaml --model.args.latent_tokens=[64,64,32]`

**Never hardcode paths or hyperparameters** in Python — they belong in YAML config. Reference them as `config["dataset"]["base_path"]`.

---

## Trainer Pattern

All trainers extend `TrainerBase`. Do not bypass the base class lifecycle:

```python
class TrainerBase:
    def __init__(self, args: dict):
        self.init_dataset(dataset_config)   # override in subclass
        self.init_model(model_config)        # override in subclass
        self.init_optimizer(optimizer_config)
        self.tracker = WandbTracker(...)     # always W&B

class StaticTrainer3D(TrainerBase):
    def init_dataset(self, ...): ...   # builds VTKMeshDataset + DataLoader
    def init_model(self, ...): ...     # instantiates GAOT3D
    def train_step(self, batch): ...
    def validate(self, loader, epoch): ...
    def test(self): ...
    def run(self, num_epochs): ...
```

**Dtype/autocast**: bfloat16 uses `torch.amp.autocast` — model stays float32, autocast applies within the forward/backward block.

**DDP**: enabled by `setup.distributed: true` — uses NCCL backend, initialized from environment variables. Don't use `torch.nn.DataParallel`.

**Uncertainty weighting**: Kendall et al. 2018 multi-task loss. `log_sigma_field` and `log_sigma_scalar` are learnable parameters with no weight decay.

---

## Experiment Tracking

Always use W&B (`wandb`). Never use plain `print` for metrics.

```python
# Log per step or epoch
self.tracker.log({"train/loss": loss, "val/mse": mse}, step=epoch)

# Save model checkpoint (handled by tracker)
self.tracker.save_model(model, epoch=epoch)
```

W&B entity is `"aisco"`, project names follow `"WP1_<description>"` pattern.

---

## Dataset Registration

Add new datasets to the `DATASET_METADATA` registry in `src/dataloader/metadata.py`:

```python
DATASET_METADATA["my_new_dataset"] = DatasetMetadata(
    input_features=["pos", "c"],
    output_features=["cp"],
    group_c=None,          # None = normals stored but not used as input
    domain={"x": [-5, 5], "y": [-2, 2], "z": [-1, 2]},
)
```

Then add preprocessing logic in `dataset/<name>/process_<name>.py` following the common convention:
1. Load raw CFD files (VTK, HDF5, PLY, etc.)
2. Compute Cp via `compute_pressure_coefficient()`
3. Compute surface normals (PyVista `compute_normals` or manual fan-triangulation)
4. Compute geometry conditioning via `compute_geom_conditioning(pos, v_inf)`
5. Build `torch_geometric.data.Data(pos=..., x=..., c=..., face=..., ...)` and save as `.pt`

---

## Coding Conventions

- **Python 3.11.6** with strict type hints throughout (checked by mypy)
- **Ruff** for linting — run via pre-commit. No bare `except`, no unused imports.
- **Naming**: `PascalCase` for classes, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for global constants.
- **Imports**: use absolute imports from `src.` when inside GAOT2; `torch_geometric.data.{Data,Batch,Dataset}` for mesh data.
- **Errors**: raise explicit typed exceptions (`TypeError`, `ValueError`, `FileNotFoundError`). Include context in the message: `raise ValueError(f"Unknown strategy: {name!r}")`.
- **Logging**: use `logging.getLogger(__name__)` — no bare `print` in library code. Prints are OK in scripts and notebooks.
- **Private helpers**: prefix with `_` (e.g., `_subsample_random`, `_build_latent_grid`).

### Shape annotation convention

Document tensor shapes inline:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, N, C]  →  out: [B, N, C_out]
```

---

## Graph Building: Edge Index Conventions

`edge_index` is always `[2, E]` where:
- **Encoder**: `edge_index[0]` = physical mesh indices (source), `edge_index[1]` = latent token indices (target)
- **Decoder**: `edge_index[0]` = latent token indices (source), `edge_index[1]` = physical query indices (target)

Always call `coalesce()` on manually constructed edge indices to remove duplicates.

Batch offsets are handled by `UnifiedCollateFunction` — never manually add offsets in model code.

---

## Multi-head Output Format

The model returns a `dict` from `forward()`. Always use string keys matching the decoder head names:

```python
outputs = {
    "force": ...,        # [B, 1]
    "cd": ...,           # [B, 1]
    "field": ...,        # [N_query, 1], only when field_decoder active
}
```

Loss computation in `train_step` iterates over the configured output heads — add new heads by updating `decoder.output_heads` in the config.

---

## Subsampling & Large Meshes

Full-resolution meshes can have 8–10M nodes. Use subsampling strategies:
- `"random"`: fast, varies each epoch (acts as augmentation)
- `"voxel"`: spatially uniform, converges via binary search on voxel size
- `"fps"` / `"poisson_disk"`: higher quality, slower

`subsample_nodes: null` disables subsampling (use for final evaluation).
`subsample_nodes: 20000` is typical for training.

When subsampling is active, `face` connectivity is dropped in `VTKMeshDataset.get()` before subsampling.

---

## SLURM / HPC Conventions

Training runs on ETH Zürich Euler cluster. Job configs use `.slurm` files referencing YAML configs. Processed data lives under `/cluster/work/fuge/AISCO/wp1_cfd_ai/`.

- Always request GPU memory accounting for full-resolution batch: 80GB per GPU for large meshes.
- DDP training: set `setup.distributed: true` and launch with `torchrun`.
- Checkpoints and W&B runs are the source of truth — not local file logs.
