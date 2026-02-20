"""
Ultra-efficient non-differentiable marching cubes mesh extraction using PyMCubes.

Optimized for high-resolution (512³+) repeated mesh extraction from neural SDFs.
Uses GPU-batched SDF evaluation with CPU-based PyMCubes for maximum throughput.
"""

import warnings

import torch

try:
    import diso

    HAS_DISO = True
except ImportError:
    HAS_DISO = False

try:
    import mcubes  # PyMCubes - fast C++ marching cubes

    HAS_PYMCUBES = True
except ImportError:
    # raise a warning that PyMCubes is not available and falling back to sklearn
    warnings.warn(
        "PyMCubes not found. Falling back to sklearn.measure.marching_cubes (much slower). Install PyMCubes for best performance.",
        stacklevel=2,
    )
    HAS_PYMCUBES = False
    from skimage import measure  # Fallback to sklearn

from subprojects.wp3_solar_estimator.mesh_extractors.diso.helpers.grid_formatting import (
    create_bbox_grid,
    renormalise_vertices,
    sample_sdf,
    sdf_samples_to_grid,
)


class MarchingCubesExtractor:
    """
    High-performance non-differentiable marching cubes mesh extractor.

    Uses PyMCubes (C++ backend) for ~10x speedup over sklearn at high resolutions.
    Falls back to sklearn.measure.marching_cubes if PyMCubes unavailable.

    Optimizations:
    - torch.no_grad() for SDF evaluation (no gradient tracking overhead)
    - Large batch sizes (default 8192) for fewer kernel launches
    - GPU-accelerated SDF evaluation, CPU-based mesh extraction
    - Reuses existing grid helper functions for consistency

    Usage:
        extractor = MarchingCubesExtractor(device="cuda", verbose=True)
        extractor.set_grid(bbox=[[-5000, 5000], [-5000, 5000], [-5000, 5000]], resolution=512)
        verts, faces = extractor.extract_mesh(model, isovalue=0.0, batch_size=8192)
    """

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = True,
        fixed_box: bool = True,
    ):
        """
        Initialize the marching cubes extractor.

        Args:
            device: Device for SDF evaluation, e.g. "cuda" or "cpu"
            verbose: Whether to print extraction information
            fixed_box: Whether to use fixed bounding box (only True supported)
        """
        self.device = device
        self.verbose = verbose
        self._fixed_box = fixed_box

        # Grid state
        self._bbox = None
        self._grid = None
        self._resolution = None

        # Print which backend is being used
        if self.verbose:
            backend = "PyMCubes (C++)" if HAS_PYMCUBES else "sklearn (fallback)"
            print(f"MarchingCubesExtractor initialized with {backend} backend on {device}")

    def set_grid(self, bbox: list[list[int]], resolution: int):
        """
        Sets the sampling grid for SDF evaluation.

        Args:
            bbox: Bounding box [[xmin,xmax], [ymin,ymax], [zmin,zmax]]
            resolution: Grid resolution per axis (e.g., 512 for 512³ grid)
        """
        if not self._fixed_box:
            raise NotImplementedError("Currently only fixed_box=True is supported")

        self._grid = create_bbox_grid(bbox, resolution, self.device)
        self._bbox = bbox
        self._resolution = resolution

        if self.verbose:
            total_points = resolution**3
            print(f"Grid set: {resolution}³ = {total_points:,} points, bbox={bbox}")

    def extract_mesh(self, model: callable, isovalue: float = 0.0, batch_size: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract isosurface mesh from neural SDF using marching cubes.

        Args:
            model: Callable SDF function (N, 3) -> (N, 1) or (N,) returning SDF values
            isovalue: Isosurface level to extract (typically 0.0 for zero-level set)
            batch_size: Points per batch for SDF evaluation (larger = faster, more memory)
                       Default 8192 optimized for 512³ on 16GB GPU

        Returns:
            verts: (V, 3) tensor of vertex positions in bbox coordinate space
            faces: (F, 3) tensor of triangle face indices (long dtype)
        """
        # Validate grid is configured
        if self._fixed_box and (self._grid is None or self._resolution is None):
            raise RuntimeError("Extractor grid not set. Call set_grid(bbox, resolution) before extracting mesh.")

        # Step 1: Sample SDF on grid with no gradient tracking (performance boost)
        with torch.no_grad():
            sdf_values = sample_sdf(model, self._grid, batch_size=batch_size)

        # Step 2: Reshape to 3D grid
        sdf_grid = sdf_samples_to_grid(sdf_values, self._resolution)

        # Step 3: Convert to numpy and run marching cubes
        sdf_grid_np = sdf_grid.squeeze().cpu().numpy()

        if HAS_PYMCUBES:
            # PyMCubes backend (fast C++ implementation)
            verts, faces = mcubes.marching_cubes(sdf_grid_np, isovalue)
        else:
            # Fallback to sklearn
            verts, faces, _, _ = measure.marching_cubes(
                sdf_grid_np,
                level=isovalue,
                spacing=(1.0, 1.0, 1.0),  # We'll normalize manually
            )

        # Step 4: Normalize vertices from grid index space to bbox space
        # PyMCubes/sklearn return vertices in [0, resolution-1] index coordinates
        # First normalize to [0, 1]
        verts_normalized = verts / (self._resolution - 1)

        # Convert to torch and use existing helper to scale to bbox
        verts_torch = torch.from_numpy(verts_normalized).float()
        verts_torch = renormalise_vertices(verts_torch, bbox=self._bbox)

        # Step 5: Convert faces to torch and move to device
        faces_torch = torch.from_numpy(faces).long()

        # Move to requested device
        verts_torch = verts_torch.to(self.device)
        faces_torch = faces_torch.to(self.device)

        if self.verbose:
            backend = "PyMCubes" if HAS_PYMCUBES else "sklearn"
            print(f"[{backend}] Extracted mesh: {verts_torch.shape[0]:,} vertices, {faces_torch.shape[0]:,} faces")

        return verts_torch, faces_torch

    def extract_mesh_chunked(
        self,
        model: callable,
        isovalue: float = 0.0,
        batch_size: int = 8192,
        chunks: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract mesh using spatial chunking for ultra-high resolutions (1024³+).

        Divides the bounding box into sub-volumes, extracts meshes separately,
        then merges them. Note: Creates duplicate vertices at chunk boundaries.

        Args:
            model: Callable SDF function
            isovalue: Isosurface level
            batch_size: Points per batch for SDF evaluation
            chunks: Number of chunks per axis (e.g., 2 = 2³ = 8 sub-volumes)

        Returns:
            verts: Merged vertices (may contain duplicates at seams)
            faces: Merged faces with adjusted indices
        """
        if chunks == 1:
            # No chunking needed
            return self.extract_mesh(model, isovalue, batch_size)

        if not HAS_PYMCUBES:
            print("Warning: Chunked extraction works best with PyMCubes. Using sklearn fallback.")

        # TODO: Implement spatial chunking if needed for 1024³+
        # For now, raise NotImplementedError and suggest increasing batch_size
        raise NotImplementedError(
            "Chunked extraction not yet implemented. For 1024³ grids, try:\n"
            "1. Increase batch_size to max GPU memory allows\n"
            "2. Use CPU device if GPU memory insufficient\n"
            "3. Reduce resolution to 512³ or 768³"
        )


class MeshExtractor:
    """
    Unified mesh extractor with a switch between MC and DMC backends.

    Algorithms:
        - "mc": Marching Cubes using PyMCubes/sklearn (non-differentiable, fast)
        - "dmc": Dual Marching Cubes using DISO (non-differentiable, similar perf)
    """

    def __init__(
        self,
        algorithm: str = "mc",
        device: str = "cuda",
        verbose: bool = True,
        fixed_box: bool = True,
        dmc_dtype: str = "float32",
    ) -> None:
        self.algorithm = algorithm.lower()
        self.device = device
        self.verbose = verbose
        self._fixed_box = fixed_box

        self._bbox = None
        self._grid = None
        self._resolution = None

        if self.algorithm == "dmc":
            if not HAS_DISO:
                raise ImportError("DISO is not available. Install diso to use DMC.")
            if dmc_dtype not in ["float32", "float64"]:
                raise ValueError("dmc_dtype must be 'float32' or 'float64'")
            dtype = torch.float32 if dmc_dtype == "float32" else torch.float64
            self._dmc = diso.DiffDMC(dtype=dtype).to(device)
            self._mc = None
            if self.verbose:
                print(f"MeshExtractor initialized with DMC backend on {device}")
        else:
            self._mc = MarchingCubesExtractor(device=device, verbose=verbose, fixed_box=fixed_box)
            self._dmc = None

    def set_grid(self, bbox: list[list[int]], resolution: int):
        if not self._fixed_box:
            raise NotImplementedError("Currently only fixed_box=True is supported")

        if self.algorithm == "dmc":
            self._grid = create_bbox_grid(bbox, resolution, self.device)
            self._bbox = bbox
            self._resolution = resolution
            if self.verbose:
                total_points = resolution**3
                print(f"Grid set: {resolution}³ = {total_points:,} points, bbox={bbox}")
        else:
            self._mc.set_grid(bbox, resolution)

    def extract_mesh(self, model: callable, isovalue: float = 0.0, batch_size: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
        if self.algorithm == "dmc":
            if self._fixed_box and (self._grid is None or self._resolution is None):
                raise RuntimeError("Extractor grid not set. Call set_grid(bbox, resolution) before extracting mesh.")

            with torch.no_grad():
                sdf_values = sample_sdf(model, self._grid, batch_size=batch_size)
            sdf_grid = sdf_samples_to_grid(sdf_values, self._resolution)

            verts, faces = self._dmc(sdf_grid, isovalue=isovalue)
            verts = renormalise_vertices(verts, bbox=self._bbox)

            if self.verbose:
                print(f"[DMC] Extracted mesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} faces")

            return verts, faces

        return self._mc.extract_mesh(model, isovalue=isovalue, batch_size=batch_size)
