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
  Supports Loop subdivision with mesh saving at each subdivision level.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from subprojects.wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import (
    DISOExtractor,
)
from subprojects.wp4_aeroshell_optimizer.v2.src.losses.parent_loss import ParentLoss

_pdev_module: Any
CachedDevelopabilityEnergy: Any
try:
    import subprojects.pydevelopability.pydevelopability as _pdev_module
    from subprojects.pydevelopability.pydevelopability.torch_wrapper import (
        CachedDevelopabilityEnergy,
    )

    _PDEV_AVAILABLE = True
except ImportError:
    _pdev_module = None
    CachedDevelopabilityEnergy = None
    _PDEV_AVAILABLE = False


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
        if not _PDEV_AVAILABLE:
            msg = (
                "pydevelopability not installed. "
                "Build: cd subprojects/pydevelopability/build_manual && cmake .. && make -j4 && "
                "cp _devcore.*.so ../pydevelopability/"
            )
            raise ImportError(msg)

        self.pdev = _pdev_module

        cfg = self.losses_config.get("developability_config", {})
        self.energy_type = cfg.get("energy_type", 4)  # PAIRWISENORMALS default
        self.mesh_resolution = cfg.get("mesh_resolution", 64)
        self.mesh_bbox = cfg.get("mesh_bbox", [[-5000, 5000]] * 3)
        self.mesh_isovalue = cfg.get("mesh_isovalue", 0.0)
        self.mesh_batch_size = cfg.get("mesh_batch_size", 8192)

        # Post-optimization config
        self.post_opt_enabled = cfg.get("post_opt_enabled", False)
        self.post_opt_energy_type = cfg.get("post_opt_energy_type", self.energy_type)
        self.post_opt_linesearch = cfg.get("post_opt_linesearch", 1)  # OVERTON
        self.post_opt_step_type = cfg.get("post_opt_step_type", 1)  # LBFGS
        self.post_opt_num_steps = cfg.get("post_opt_num_steps", 100)
        self.post_opt_subdivisions = cfg.get("post_opt_subdivisions", 0)
        self.post_opt_output_dir = cfg.get("post_opt_output_dir", "postprocessed_meshes")

        # DISO mesh extractor (differentiable)
        self.mesh_extractor = DISOExtractor(
            extractor_type="DiffMC",
            dtype="float32",
            device=self.device,
        )
        self.mesh_extractor.set_grid(
            bbox=self.mesh_bbox,
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

    def post_optimize(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        run_name: str,
        storage_dir: str,
    ) -> dict:
        """
        Run C++ energy minimization with subdivisions, saving meshes at each level.

        Called after the final training epoch. Modifies vertex positions directly.

        Args:
            vertices: (V, 3) float64 array
            faces: (F, 3) int32 array
            run_name: Name of the run (for output filenames)
            storage_dir: Base storage directory from general_config

        Returns:
            dict with keys:
                V: optimized vertices (final)
                F: optimized faces (final)
                energy_history: list of energy values
                step_statuses: list of step statuses
                saved_meshes: list of saved mesh paths
        """
        V = np.ascontiguousarray(vertices, dtype=np.float64)
        F = np.ascontiguousarray(faces, dtype=np.int32)

        # Create output directory
        output_dir = Path(storage_dir) / self.post_opt_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build problem
        problem = self.pdev.build_problem(V, F)

        # Run optimization with subdivisions
        result = self.pdev.run_optimization(
            problem,
            V,
            energy_type=self.post_opt_energy_type,
            linesearch=self.post_opt_linesearch,
            step_type=self.post_opt_step_type,
            num_steps=self.post_opt_num_steps,
            num_subdivisions=self.post_opt_subdivisions,
        )

        # Save mesh at each subdivision level (D=0,1,2,...)
        saved_meshes = []
        for subdiv_idx in range(len(result.V_per_subdiv)):
            V_subdiv = result.V_per_subdiv[subdiv_idx]
            F_subdiv = result.F_per_subdiv[subdiv_idx]

            # Compute energy at end of this subdivision phase
            step_end = min(
                (subdiv_idx + 1) * self.post_opt_num_steps - 1,
                len(result.energy_history) - 1,
            )
            energy = result.energy_history[step_end]

            # Build filename: <run_name>_energy<E>_linesearch<L>_steptype<S>_iterations<N>_subdivision<D>_E<energy>.ply
            filename = (
                f"{run_name}_"
                f"energy{self.post_opt_energy_type}_"
                f"linesearch{self.post_opt_linesearch}_"
                f"steptype{self.post_opt_step_type}_"
                f"iterations{self.post_opt_num_steps}_"
                f"subdivision{subdiv_idx}_"
                f"E{energy:.4f}.ply"
            )

            filepath = output_dir / filename

            # Save mesh
            mesh = trimesh.Trimesh(vertices=V_subdiv, faces=F_subdiv)
            mesh.export(filepath)
            saved_meshes.append(str(filepath))

            print(f"[DevelopabilityV0] Saved subdivision {subdiv_idx}/{self.post_opt_subdivisions}: {filepath.name}")

        # Print summary
        print(
            f"[DevelopabilityV0] Post-optimization complete: "
            f"Energy {result.energy_history[0]:.6f} -> {result.energy_history[-1]:.6f}"
        )
        print(f"[DevelopabilityV0] Saved {len(saved_meshes)} meshes to {output_dir}")

        return {
            "V": result.V,
            "F": result.F,
            "energy_history": result.energy_history,
            "step_statuses": result.step_statuses,
            "saved_meshes": saved_meshes,
        }
