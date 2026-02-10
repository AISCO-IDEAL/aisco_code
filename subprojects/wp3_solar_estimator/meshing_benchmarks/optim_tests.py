from pathlib import Path

import torch
from wp3_solar_estimator.geo_mesh_utils.mesh_properties import compute_mesh_surface_area
from wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from wp3_solar_estimator.meshing_benchmarks.analytical_sdfs import SphereSDF
from wp3_solar_estimator.meshing_benchmarks.tests import BaseTester
from wp4_aeroshell_optimizer.v1.src.visuals.visual_wrapper import Wrapper

from subprojects.wp3_solar_estimator.mesh_extractors.diso.helpers.storage_functions import tensors_to_mesh


class OptimiseSphereRadius(BaseTester):
    """
    Test case for optimizing the radius of a sphere SDF to match a target surface area.
    This serves as a simple benchmark for validating gradient accuracy of different mesh extraction methods.
    The analytical SDF and its gradients are defined in SphereSDF, which is used as the test model.
    The target surface area is set to that of a sphere with a known radius, and the optimization should ideally converge to that radius.
    This test can be used to compare the convergence behavior and gradient accuracy of different meshing algorithms when optimizing a simple geometric parameter.
    The results can be visualized by plotting the loss curve and the estimated radius over iterations, and comparing it to the true radius.
    """

    def __init__(self):
        super().__init__()
        torch.manual_seed(42)  # Set a fixed seed for reproducibility
        self.meshing_algorithms = {
            "DISO_DiffMC": DISOExtractor(extractor_type="DiffMC"),
            "DISO_DiffDMC": DISOExtractor(extractor_type="DiffDMC"),
            "FlexiCubes": FCExtractor(),
            "MeshSDF": MeshSDFExtractor(),
        }

        self.set_all_grids(bbox=[[-3500.0, 3500.0], [-3500.0, 3500.0], [-3500.0, 3500.0]], resolution=64)

        self.visualiser = Wrapper(
            output_dir=Path("/cluster/work/fuge/AISCO/wp3_solar_estimator/benchmarks/optimisation_test"), verbose=True
        )

    def optimise_sphere(
        self,
        extractor_name: str = "DISO_DiffMC",
        target_radius: float = 2000.0,
        init_radius: float = 1500.0,
        num_iterations: int = 120,
        learning_rate: float = 10.0,
        visualize: bool = True,
    ) -> None:
        """
        Optimizes the radius of the SphereSDF to match the target surface area corresponding to the target radius.
        The loss is defined as the squared difference between the estimated surface area from the extracted mesh and the target surface area.
        """
        # Initialize SphereSDF with a given radius
        model = SphereSDF(radius=init_radius)  # Start with a radius different from the target for optimization

        # Compute target surface area for the given target radius
        target_surface_area = 4 * torch.pi * (target_radius**2)

        # Choose an extractor (e.g., DISO_DiffMC) for optimization
        extractor = self.meshing_algorithms[extractor_name]

        optimizer = torch.optim.Adam([model.radius], lr=learning_rate)

        for iteration in range(num_iterations):
            optimizer.zero_grad()
            verts, faces = extractor.extract_mesh(model)

            # visualize mesh
            if visualize:
                self.visualiser.update({"model": model, "iteration": iteration, "vertices": verts, "faces": faces})

            if verts.numel() == 0 or faces.numel() == 0:
                print(f"Warning: Extracted empty mesh at iteration {iteration}. Skipping loss computation.")
                continue

            # Compute estimated surface area from extracted mesh (using a simple approximation)
            estimated_surface_area = compute_mesh_surface_area(verts, faces)

            # Compute loss as squared difference from target surface area
            loss = (estimated_surface_area - target_surface_area) ** 2

            # Backpropagate and optimize
            loss.backward()
            optimizer.step()

            print(f"Iteration {iteration}: Estimated Radius = {model.radius.item():.2f}, Loss = {loss.item():.4f}")

        # save final mesh for comparison
        final_verts, final_faces = extractor.extract_mesh(model)
        tensors_to_mesh(
            final_verts,
            final_faces,
            save_mesh=True,
            filename=f"{extractor_name}_OptimizedSphere.ply",
            mesh_dir="/cluster/work/fuge/AISCO/wp3_solar_estimator/benchmarks/optimisation_test",
        )
        if visualize:
            self.visualiser.finalize(
                custom_name=f"{extractor_name}", delete_frames=True
            )  # Save GIF and clear frames after finalization

    def optimise_all_extractors(self, **kwargs) -> None:
        for extractor_name in self.meshing_algorithms:
            print(f"\nOptimizing with {extractor_name}...")
            self.optimise_sphere(extractor_name=extractor_name, **kwargs)
