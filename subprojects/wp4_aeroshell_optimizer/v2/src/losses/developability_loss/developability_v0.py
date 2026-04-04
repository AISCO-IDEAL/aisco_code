"""
Developability constraint loss for AISCO.

Computes the developability energy of a mesh extracted from the SDF model
using the "Developability of Triangle Meshes" (Stein et al. 2018) energies
via pydevelopability (PyBind11 wrapper around the C++ implementation).

Energy types:
  0 = HINGE, 1 = MINWIDTH, 2 = OLDHINGE, 3 = OLDMINWIDTH,
  4 = PAIRWISENORMALS, 5 = MAXPAIRWISENORMALS

Post-optimization (after training):
  Optionally runs the C++ L-BFGS/Overton optimizer directly on the mesh
  vertex positions for fine-grained developability improvement.
"""

from __future__ import annotations

import numpy as np
import pydevelopability as pdev
import torch
from pydevelopability.torch_wrapper import CachedDevelopabilityEnergy

from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import (
    DISOExtractor,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss


class DevelopabilityV0(ParentLoss):
    """
    Developability loss: measures how far a mesh is from being developable.

    Config keys (in losses_v0_config.yaml under developability_config):
        energy_type: int           # 0-5, which energy functional
        mesh_resolution: int       # DISO grid resolution (e.g. 64, 128)
        mesh_bbox: list            # [[xmin,xmax],[ymin,ymax],[zmin,zmax]]
        mesh_isovalue: float       # isosurface value (default 0.0)
        mesh_batch_size: int       # SDF eval batch size

        # Post-optimization (applied after final epoch):
        post_opt_enabled: bool     # whether to run C++ optimizer after training
        post_opt_energy_type: int  # energy type for post-optimization
        post_opt_linesearch: int   # 0=NONE, 1=OVERTON, 2=BACKTRACK
        post_opt_step_type: int    # 0=GRADDESC, 1=LBFGS
        post_opt_num_steps: int    # number of optimization steps
    """

    def _setup(self) -> None:
        cfg = self.losses_config.get("developability_config", {})
        self.energy_type = cfg.get("energy_type", 4)  # PAIRWISENORMALS default
        self.mesh_resolution = cfg.get("mesh_resolution", 64)
        self.mesh_bbox = cfg.get("mesh_bbox", [[-5000, 5000]] * 3)
        self.mesh_isovalue = cfg.get("mesh_isovalue", 0.0)
        self.mesh_batch_size = cfg.get("mesh_batch_size", 8192)

        # Post-optimization config
        self.post_opt_enabled = cfg.get("post_opt_enabled", True)
        self.post_opt_energy_type = cfg.get("post_opt_energy_type", self.energy_type)
        self.post_opt_linesearch = cfg.get("post_opt_linesearch", 1)  # OVERTON
        self.post_opt_step_type = cfg.get("post_opt_step_type", 1)  # LBFGS
        self.post_opt_num_steps = cfg.get("post_opt_num_steps", 100)

        # DISO mesh extractor (differentiable)
        self.mesh_extractor = DISOExtractor(
            extractor_type="DiffMC",
            dtype=torch.float32,
            device=self.device,
        )
        self.mesh_extractor.set_grid(
            bbox=torch.tensor(self.mesh_bbox, dtype=torch.float32),
            resolution=self.mesh_resolution,
        )

        # C++ energy evaluator (with topology caching)
        self.energy_fn = CachedDevelopabilityEnergy(energy_type=self.energy_type)

    def _loss(self, model) -> torch.Tensor:
        # Extract mesh from SDF via DISO (differentiable)
        verts, faces = self.mesh_extractor.extract_mesh(
            model=model,
            isovalue=self.mesh_isovalue,
            batch_size=self.mesh_batch_size,
        )

        if verts.shape[0] == 0:
            return torch.tensor(0.0, device=self.device)

        # Compute developability energy (C++ backend, analytical gradient)
        loss = self.energy_fn(verts.double(), faces)

        return loss.to(dtype=torch.float32)

    def post_optimize(self, vertices: np.ndarray, faces: np.ndarray) -> dict:
        """
        Run C++ energy minimization on extracted mesh vertices.

        Called after the final training epoch. Modifies vertex positions directly.

        Args:
            vertices: (V, 3) float64 array
            faces: (F, 3) int32 array

        Returns:
            dict with keys: V (optimized vertices), energy_history, step_statuses
        """
        V = np.ascontiguousarray(vertices, dtype=np.float64)
        F = np.ascontiguousarray(faces, dtype=np.int32)
        problem = pdev.build_problem(V, F)

        result = pdev.run_optimization(
            problem,
            V,
            energy_type=self.post_opt_energy_type,
            linesearch=self.post_opt_linesearch,
            step_type=self.post_opt_step_type,
            num_steps=self.post_opt_num_steps,
        )

        return {
            "V": result.V,
            "energy_history": result.energy_history,
            "step_statuses": result.step_statuses,
        }
