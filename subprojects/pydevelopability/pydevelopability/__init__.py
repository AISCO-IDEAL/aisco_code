"""
pydevelopability: Python/PyTorch bindings for "Developability of Triangle Meshes"
"""

from pydevelopability._devcore import (
    EnergyType,
    FixedTopologyProblem,
    Linesearch,
    OptimizationResult,
    StepType,
    build_problem,
    evaluate_energy_and_grad,
    run_optimization,
    total_energy,
)

# Convenience names matching the C++ enum integers
ENERGY_NAMES = {
    0: "HINGE",
    1: "MINWIDTH",
    2: "OLDHINGE",
    3: "OLDMINWIDTH",
    4: "PAIRWISENORMALS",
    5: "MAXPAIRWISENORMALS",
}

__all__ = [
    "ENERGY_NAMES",
    "EnergyType",
    "FixedTopologyProblem",
    "Linesearch",
    "OptimizationResult",
    "StepType",
    "build_problem",
    "evaluate_energy_and_grad",
    "run_optimization",
    "total_energy",
]
