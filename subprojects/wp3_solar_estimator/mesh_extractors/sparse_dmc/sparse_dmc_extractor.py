"""This module defines the SparseDMCExtractor, which serves as a wrapper around the core mesh extraction function sparse_sdf_to_mesh. The extractor is designed to be compatible with different versions of the loss function used in the meshing process. It provides a unified interface for extracting meshes from signed distance functions (SDFs) while allowing for flexibility in the underlying loss computations."""

import torch

from subprojects.wp3_solar_estimator.mesh_extractors.sparse_dmc.core import sparse_sdf_to_mesh


class SparseDMCExtractor:
    """Wrapper class for the SparseDMC mesh extraction method. This class interfaces with the core mesh extraction function sparse_sdf_to_mesh, allowing it to be used seamlessly within the solar loss computation framework."""

    def __init__(self, bbox, resolution, device, batch_size=1):
        self.xyz_min = tuple(bbox[i][0] for i in range(3))  # Extract min values for x, y, z from bbox
        self.xyz_max = tuple(bbox[i][1] for i in range(3))  # Extract max values for x, y, z from bbox
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device
        print(
            f"SparseDMCExtractor initialized with bbox={bbox}, resolution={resolution}, device={device}, batch_size={batch_size}"
        )

    def extract_mesh(self, model) -> tuple[torch.Tensor, torch.Tensor]:
        """Extracts a mesh from SDF model(s) using the sparse DMC method.
        This function calls the core mesh extraction function sparse_sdf_to_mesh, which is implemented by Tim Aebersold.
        As until now only supports batch_size=1, it will return the first mesh in the list of meshes returned by sparse_sdf_to_mesh.
        """
        vertices_list, faces_list, batch_size = sparse_sdf_to_mesh(
            sdf_fn=model,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            resolution=self.resolution,
            batch_size=self.batch_size,
            device=self.device,
        )

        if batch_size >= 1:
            verts = vertices_list[0]
            faces = faces_list[0]
        else:
            raise ValueError("Batch size must be at least 1 to extract mesh.")

        print(f"SparseDMCExtractor extracted mesh with {verts.shape[0]} vertices and {faces.shape[0]} faces.")

        return verts, faces

    def set_grid(self, bbox: list[list[int | float]], resolution: int) -> None:
        """
        Configure extraction domain for sparse DMC.

        Args:
            bbox: [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            resolution: Target grid resolution (must be power of 2).
        """
        if len(bbox) != 3 or any(len(axis) != 2 for axis in bbox):
            raise ValueError("bbox must be [[xmin, xmax], [ymin, ymax], [zmin, zmax]].")

        xyz_min = tuple(float(bbox[i][0]) for i in range(3))
        xyz_max = tuple(float(bbox[i][1]) for i in range(3))

        if any(lo >= hi for lo, hi in zip(xyz_min, xyz_max, strict=True)):
            raise ValueError("Each bbox axis must satisfy min < max.")

        if resolution < 2 or (resolution & (resolution - 1)) != 0:
            raise ValueError(f"resolution must be a power of 2, got {resolution}.")

        self.xyz_min = xyz_min
        self.xyz_max = xyz_max
        self.resolution = int(resolution)
