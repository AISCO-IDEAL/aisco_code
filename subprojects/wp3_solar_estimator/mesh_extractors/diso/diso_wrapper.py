from collections.abc import Callable

import diso
import torch

from subprojects.wp3_solar_estimator.mesh_extractors.diso.helpers.grid_formatting import (
    create_bbox_grid,
    renormalise_vertices,
    sample_sdf,
    sdf_samples_to_grid,
)


class DISOExtractor:
    def __init__(
        self,
        extractor_type: str = "DiffMC",
        dtype: str = "float32",
        device: str = "cuda",
        verbose: bool = True,
        deform: bool = False,
        fixed_box: bool = True,
    ):
        """

        DISOExtractor wrapper class, based on https://github.com/SarahWeiii/diso/blob/main/README.md
        Note that we put the extractor on cuda device per default (this is where it gets the performance from)

        extractor_type: Type of the extractor, either "DiffMC" or "DiffDMC"
        dtype: Data type for the extractor, either "float32" or "float64"
        device: Device to put the extractor on, e.g. "cuda" or "cpu"
        verbose: Whether to print information during extraction
        deform: Whether to use deformation of grid or not

        fixed_box: Whether to use a fixed bounding box for the grid or not, see set_grid function

        """

        # Initialise the DISO Object

        if dtype not in ["float32", "float64"]:
            raise Warning(f"Unsupported dtype {dtype}, use 'float32' or 'float64'")
        if extractor_type == "DiffMC":
            self.extractor = diso.DiffMC(dtype=torch.float32 if dtype == "float32" else torch.float64)
        elif extractor_type == "DiffDMC":
            self.extractor = diso.DiffDMC(dtype=torch.float32 if dtype == "float32" else torch.float64)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        self.extractor = self.extractor.to(device)
        self.device = device  # Store device directly instead of extracting from parameters

        ### Settings for extractor
        self.deform = deform
        self.verbose = verbose

        # if using fixed_box:
        if not fixed_box:
            raise NotImplementedError("Currently only fixed_box=True is supported")

        self._fixed_box = fixed_box
        self._grid: torch.Tensor | None = None
        self._bbox: list[list[int]] | None = None
        self._resolution: int | None = None

    def set_grid(self, bbox: list[list[int]], resolution: int):
        """

        Sets the grid for the extractor on which the SDF will be sampled.

        bbox: Bounding box for the grid, e.g. [[-5000, 5000], [-5000, 5000], [-5000, 5000]]
        resolution: Resolution of the grid, e.g. 64 for a 64x64x64 grid

        """
        if self._fixed_box:
            self._grid = create_bbox_grid(bbox, resolution, self.device)
            self._bbox = bbox
            self._resolution = resolution
        else:
            raise NotImplementedError("Currently only fixed_box=True is supported")

    def extract_mesh(self, model: Callable, isovalue: float = 0.0, batch_size: int = 2048):
        """
        Extracts a mesh from the given SDF model using the configured extractor.

        model: Callable SDF function that takes in (N, 3) points and returns (N, 1) SDF values
        isovalue: The isovalue to extract the mesh at
        batch_size: Number of points to sample in each batch

        Returns:
            verts: Extracted vertices of the mesh
            faces: Extracted faces of the mesh
        """

        if self._fixed_box and (self._grid is None or self._resolution is None):
            raise Warning("Extractor grid for fixed grid not set, please run set_grid once before extracting mesh")

        # Sample SDF on gridpoints
        sdf_values = sample_sdf(model, self._grid, batch_size=batch_size)
        # Reshape SDF values to grid
        sdf_grid = sdf_samples_to_grid(sdf_values, self._resolution)
        # Run DISO
        verts, faces = self.extractor(sdf_grid, isovalue=isovalue)
        if self.verbose:
            print(f"Extracted mesh with {verts.shape[0]} vertices and {faces.shape[0]} faces")
        # Renormalise vertices back to original coordinate space
        verts = renormalise_vertices(verts, bbox=self._bbox)
        return verts, faces
