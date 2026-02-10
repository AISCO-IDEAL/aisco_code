import torch
from wp3_solar_estimator.mesh_extractors.diso.diso_wrapper import DISOExtractor
from wp3_solar_estimator.mesh_extractors.kaolin_flexicubes.flexicubes_wrapper import FCExtractor
from wp3_solar_estimator.mesh_extractors.mesh_sdf.mesh_sdf_extractor import MeshSDFExtractor
from wp3_solar_estimator.meshing_benchmarks.analytical_sdfs import SphereSDF
from wp3_solar_estimator.meshing_benchmarks.tests import BaseTester


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
        self.testing_models = {
            "SphereSDF": SphereSDF(radius=2000.0),
        }

        self.set_all_grids(bbox=[[-3500.0, 3500.0], [-3500.0, 3500.0], [-3500.0, 3500.0]], resolution=64)
