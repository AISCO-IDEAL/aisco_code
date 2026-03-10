"""
For validation purposes, runs a basic race simulation to predict the total solar output during the race
Assumes a constant speed given in the australia.yaml config file, and a constant solar output per unit area given in the same config file.
"""



import matplotlib.pyplot as plt
import numpy as np
import torch

from subprojects.wp3_solar_estimator.australia.route_v2 import RouteV2
from subprojects.wp3_solar_estimator.utils.solar_utils import averaging_solar_quality
from subprojects.wp4_aeroshell_optimizer.v1.logging_lib.image_uploads import upload_neural_renderer_image


class SimulationV1:
    def __init__(self, cfg_path: str = "solar_configs/australia.yaml", area_budget_mm2: float = 3500000) -> None:
        self.route = RouteV2(cfg_path=cfg_path)
        self.area_budget_mm2 = area_budget_mm2

    def run(self, vertices: torch.Tensor, mesh_faces: torch.Tensor):
        # disable gradient tracking since this is just a validation simulation
        verts, faces = vertices.detach().clone(), mesh_faces.detach().clone()
        verts.requires_grad_(False)
        faces.requires_grad_(False)

        solar_cells_mask, normals, areas_mm2 = self.allocate_solar_cells(verts, faces, self.route)
        # cull faces, normals and areas that dont get solar cells allocated

        allocated_normals = normals[solar_cells_mask]
        allocated_areas_mm2 = areas_mm2[solar_cells_mask]

        # calculate the solar output for each point
        solar_output_area = np.zeros(
            self.route.num_points
        )  # array to store total solar projected area at each point along the route
        positions = np.zeros((self.route.num_points, 3))  # array to store positions along the route for debugging/analysis
        for point in range(self.route.num_points):
            sun_vec = self.route.get_route_solar_vecs(point)
            position = self.route.get_route_points(point)

            # dot product of all allocated normals with single solar vector to get solar quality, then multiply by area to get total solar projected area at this point
            solar_qualities = allocated_normals * sun_vec
            solar_projected_area = (solar_qualities * allocated_areas_mm2.unsqueeze(-1)).sum()

            positions[point] = position.tolist()
            solar_output_area[point] = solar_projected_area.item()

        self.plot_results(positions, solar_output_area)

    def allocate_solar_cells(
        self, verts: torch.Tensor, faces: torch.Tensor, route: RouteV2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each face, compute the average solar quality across its vertices

        Returns:
            solar_cells_allocated: (M,) boolean tensor mask indicating which faces get solar cells allocated
            normals: (M, 3) tensor of unit face normals, for later use in computing solar quality
            areas_mm2: (M,) tensor of face areas in mm^2, for later use in computing total solar output
        """
        solar_vectors = route.get_route_solar_vecs()

        v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
        edge_a, edge_b = v1 - v0, v2 - v0
        face_cross = torch.cross(edge_a, edge_b, dim=-1)
        face_cross_norm = torch.linalg.norm(face_cross, dim=-1)
        face_area_mm2 = 0.5 * face_cross_norm

        # Compute illumination quality (cosine of angle with sun direction, clamped to [0, 1])
        face_normals_unit = face_cross / (face_cross_norm.clamp_min(1e-6)).unsqueeze(-1)  # (M, 3) unit normals of faces
        sun_directions_unit = solar_vectors / (torch.linalg.norm(solar_vectors, dim=-1, keepdim=True).clamp_min(1e-6))

        agg_face_solar_quality = averaging_solar_quality(face_normals_unit, sun_directions_unit)

        # Sort solar cells based on quality and select top N based on area budget
        quality_order_desc = torch.argsort(agg_face_solar_quality, descending=True)
        sorted_areas = face_area_mm2[quality_order_desc]

        # Deterministic-safe cumulative area
        cumulative_area = torch.cumsum(sorted_areas.cpu(), dim=0)
        # Find which faces we can allocate (better quality first) before exceeding the area budget
        allocated_mask_sorted = cumulative_area <= self.area_budget_mm2
        # Map back to original face order
        solar_cells_allocated = torch.zeros_like(allocated_mask_sorted, dtype=torch.bool)
        solar_cells_allocated[quality_order_desc] = allocated_mask_sorted

        return solar_cells_allocated, face_normals_unit, face_area_mm2

    def plot_results(self, positions: np.ndarray, solar_output_area: np.ndarray):
        # goal here is to make a gif of the australia route map with points colored by solar output, and a plot of solar output and lat/lon over the route points.
        # note: normalises mm2 to m2 with 1e-6 calculation
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(positions[:, 1], positions[:, 0], c=solar_output_area, cmap="inferno")
        plt.colorbar(label="Solar Output (m^2)")

        plt.title("Simulated Solar Output and Route Positions")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        # ensure equal aspect ratio for lat/lon
        plt.gca().set_aspect("equal", adjustable="box")

        # log to wandb as image
        upload_neural_renderer_image(plt, "simulation_results.png")
