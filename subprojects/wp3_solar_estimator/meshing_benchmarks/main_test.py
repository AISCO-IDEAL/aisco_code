"""
Main benchmarking script for comparing differentiable isosurface extraction methods.

Compares DISO (DiffMC, DiffDMC), FlexiCubes, and MeshSDF across:
- Gradient accuracy vs analytical solutions
- Gradient smoothness/discontinuities
- Runtime performance

Results are logged to Weights & Biases with visualizations.
"""

import warnings
from pathlib import Path

import torch
from wp3_solar_estimator.meshing_benchmarks.tests import BaseTester

# Weights & Biases for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. Install with: pip install wandb", stacklevel=2)


class BenchmarkConfig:
    """Configuration for benchmarking runs."""

    def __init__(self):
        # Start WandB run
        self.wandb_project = "Benchmarking of IsoSurface Extraction"
        self.wandb_entity = "aisco"  # TODO: Set your entity/username (or None for default)
        self.wandb_run_name = None  # Auto-generated if None
        self.wandb_tags = ["gradient-benchmark", "diso", "flexicubes", "meshsdf"]

        # Output
        self.save_meshes = True  # Save extracted meshes as PLY
        self.output_dir = Path("/cluster/work/fuge/AISCO/wp3_solar_estimator/benchmarks")
        self.generate_visualizations = True






if __name__ == "__main__":
    torch.manual_seed(42)  # Set a fixed seed for reproducibility

    # Initialize benchmarking configuration
    config = BenchmarkConfig()

    # Initialize Weights & Biases run
    if WANDB_AVAILABLE:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=config.wandb_run_name, tags=config.wandb_tags)

    # Run benchmarks for each method
    tester = BaseTester()
    try:
        if tester.startup():
            print("Startup check completed successfully. All methods produced meshes without errors.\n")
    except Exception as e:
        print(f"Error during startup check: {e}")

    try:
        tester.stretched_geometry_test()
    except Exception as e:
        print(f"Error during stretched geometry test: {e}")
