# Differentiable Isosurface Extraction Benchmarking

Comprehensive benchmarking framework for comparing gradient quality of differentiable mesh extraction methods.

## Overview

This toolkit benchmarks **4 differentiable mesh extraction methods**:
- **DISO (DiffMC)** - Differentiable Marching Cubes
- **DISO (DiffDMC)** - Differentiable Dual Marching Cubes
- **FlexiCubes** - Kaolin's flexible isosurface extraction
- **MeshSDF** - Gradient flow through implicit surfaces (custom implementation)

## What It Tests

### 1. Gradient Accuracy
- Compares extracted gradients against analytical ground truth
- Uses finite differences as numerical validation
- Tests on simple shapes (sphere, cube) with known analytical solutions

### 2. Gradient Smoothness/Continuity
- Detects gradient discontinuities at mesh vertices
- Uses nearest-neighbor analysis to measure gradient jumps
- Quantifies smoothness via variance and discontinuity scores

### 3. Gradient Stability
- Tests robustness under small perturbations
- Measures coefficient of variation across trials
- Identifies methods prone to gradient noise

### 4. Physical Meaningfulness
- Validates surface area accuracy
- Checks gradient sign consistency
- Ensures physical properties are preserved

## Directory Structure

```
meshing_benchmarks/
├── analytical_sdfs.py        # Sphere and cube SDF models with analytical gradients
├── mesh_sdf_extractor.py     # MeshSDF implementation (Equation 4 gradient flow)
├── gradient_metrics.py       # Gradient analysis and comparison utilities
└── main_test.py             # Main benchmarking orchestrator

geo_mesh_utils/
└── mesh_properties.py        # Surface area, normals, and geometric utilities
```

## Quick Start

### 1. Install Dependencies

```bash
# Required packages
pip install torch numpy matplotlib scikit-image pandas wandb
```

### 2. Configure Weights & Biases

Edit [main_test.py](main_test.py) and set your wandb credentials in `BenchmarkConfig`:

```python
class BenchmarkConfig:
    def __init__(self):
        self.wandb_project = "your-project-name"  # TODO: Set this
        self.wandb_entity = "your-username"       # TODO: Set this
        # ...
```

### 3. Run Benchmark

```bash
cd /cluster/home/aalthaus/AISCO_code/subprojects/wp3_solar_estimator/meshing_benchmarks
python main_test.py
```

## Configuration

All settings are in `BenchmarkConfig` class in [main_test.py](main_test.py):

```python
self.resolutions = [32, 64, 128, 256]  # Grid resolutions
self.shapes = ["sphere", "cube"]        # Test shapes
self.device = "cuda"                    # GPU/CPU
self.finite_diff_epsilon = 1.0         # Numerical gradient step size
self.stability_trials = 5               # Stability test repetitions
```

## Output

### Console Output
- Real-time progress for each method/resolution
- Summary tables with key metrics
- Error reports for failed tests

### Files Generated
- `benchmark_results/benchmark_results.csv` - Complete results table
- `benchmark_results/*.png` - Visualization plots
  - Gradient accuracy vs resolution
  - Extraction time vs resolution
  - Discontinuity scores
  - Surface area errors

### Weights & Biases
- All scalar metrics logged per method/shape/resolution
- Visualization plots uploaded as images
- Results CSV saved as artifact
- Interactive comparison dashboard

## Key Metrics

| Metric | Description | Lower is Better |
|--------|-------------|-----------------|
| `relative_error_analytical` | Gradient accuracy vs ground truth | ✓ |
| `discontinuity_score` | Gradient smoothness (0-1 scale) | ✓ |
| `gradient_cv` | Stability coefficient of variation | ✓ |
| `surface_area_rel_error` | Mesh quality indicator | ✓ |
| `extraction_time` | Runtime performance | ✓ |

## Analytical Test Cases

### Sphere
- **SDF**: `||x|| - r`
- **Surface Area**: `A = 4πr²`
- **Gradient**: `∂A/∂r = 8πr`

### Cube
- **SDF**: Box distance function
- **Surface Area**: `A = 6s²`
- **Gradient**: `∂A/∂s = 12s`

## Extending the Benchmark

### Add New Shapes

Create a new SDF model in [analytical_sdfs.py](analytical_sdfs.py):

```python
class EllipsoidSDF(nn.Module):
    def forward(self, points):
        # Implement SDF
        ...

    def analytical_surface_area(self):
        # Return ground truth
        ...

    def analytical_gradient_wrt_params(self):
        # Return analytical gradients
        ...
```

Add to `config.shapes` in [main_test.py](main_test.py).

### Add New Methods

Implement extraction in `DifferentiableMeshExtractor.extract()` in [main_test.py](main_test.py):

```python
elif method == "your_method":
    vertices, faces = your_extraction_function(...)
    return vertices, faces
```

Add to `methods` list in `run_benchmark()`.

### Customize Metrics

Add new analysis functions to [gradient_metrics.py](gradient_metrics.py) and call in `benchmark_method()`.

## Architecture Notes

### MeshSDF Implementation

Based on [cvlab-epfl/MeshSDF](https://github.com/cvlab-epfl/MeshSDF):

1. **Extract mesh** using standard Marching Cubes (skimage)
2. **Compute normals** via SDF gradient: `n = ∇_x SDF(x) / ||∇_x SDF(x)||`
3. **Inflow derivative**: `dL/ds = -dL/dx · n` (Equation 4 from paper)
4. **Backward pass** through zero-level set constraint

Key difference from DISO/FlexiCubes: Doesn't modify extraction algorithm; derives gradient flow mathematically.

### Gradient Discontinuity Detection

Uses **nearest-neighbor analysis**:
1. For each vertex, find k nearest neighbors
2. Compute L2 difference between gradient vectors
3. Aggregate: max jump, mean jump, variance
4. Normalize to discontinuity score (0-1)

No voxel-specific boundary detection needed.

## Troubleshooting

### Empty Meshes
- Check bbox encompasses the shape
- Increase grid resolution
- Verify SDF is centered at origin

### Gradient Computation Fails
- Ensure `requires_grad=True` on parameters
- Check if extraction method supports backprop
- Verify loss function is differentiable

### OOM (Out of Memory)
- Reduce `self.resolutions` (skip 256³)
- Decrease batch size in extractors
- Use CPU instead of GPU for testing

### wandb Not Logging
- Run `wandb login` first
- Check entity/project names are correct
- Verify internet connection

## Use Cases

### Research
- Compare novel extraction methods
- Validate gradient implementations
- Generate figures for papers

### Development
- Debug differentiable rendering pipelines
- Optimize aeroshell shapes (original use case)
- Test SDF network training stability

### Production
- Select best method for application
- Tune hyperparameters (grid resolution)
- Monitor gradient quality over training

## Performance Tips

1. **Start small**: Test on 32³ and 64³ first
2. **Use GPU**: 10-50x speedup for large grids
3. **Skip stability tests**: Set `config.stability_trials = 1` for quick runs
4. **Parallelize**: Run different shapes in separate processes

## Citation

If you use this benchmarking framework, please cite the original papers:

**MeshSDF**:
```
@inproceedings{remelli2020meshsdf,
  title={MeshSDF: Differentiable Iso-Surface Extraction},
  author={Remelli, Edoardo and Lukoianov, Artem and Richter, Stephan and
          Guillard, Benoit and Bagautdinov, Timur and Baque, Pierre and Fua, Pascal},
  booktitle={NeurIPS},
  year={2020}
}
```

**DISO**: (Add citation when available)

**FlexiCubes** (Kaolin):
```
@inproceedings{shen2023flexicubes,
  title={FlexiCubes: Flexible Isosurface Extraction},
  author={Shen, Tianchang and others},
  booktitle={CVPR},
  year={2023}
}
```

## Contact

For questions about this benchmarking framework, open an issue or contact the aeroshell optimization team.
