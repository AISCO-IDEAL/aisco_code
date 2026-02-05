import diso
import torch

from subprojects.wp3_solar_estimator.diso_utils.helpers.grid_formatting import (
    create_bbox_grid,
    renormalise_vertices,
    sample_sdf,
    sdf_samples_to_grid,
)


class Extractor:
    def __init__(self,
                 extractor_type: str = "diff_mc",
                 dtype: str = "float32",
                 device: str = "cuda",
                 verbose: bool = True,
                 deform: bool = False,
                 fixed_box: bool = True,

                 ):
        """

        DISO Extractor wrapper class, based on https://github.com/SarahWeiii/diso/blob/main/README.md
        Note that we put the extractor on cuda device per default (this is where it gets the performance from)

        extractor_type: Type of the extractor, either "diff_mc" or "diff_dmc"
        dtype: Data type for the extractor, either "float32" or "float64"
        device: Device to put the extractor on, e.g. "cuda" or "cpu"
        verbose: Whether to print information during extraction
        deform: Whether to use deformation of grid or not

        fixed_box: Whether to use a fixed bounding box for the grid or not, see set_grid function

        """


        #Initialise the DISO Object

        if dtype not in ["float32", "float64"]:
            raise Warning(f"Unsupported dtype {dtype}, use 'float32' or 'float64'")
        if extractor_type == "diff_mc":
            self.extractor = diso.DiffMC(dtype=torch.float32 if dtype == "float32" else torch.float64)
        elif extractor_type == "diff_dmc":
            self.extractor = diso.DiffDMC(dtype=torch.float32 if dtype == "float32" else torch.float64)
        else:
            raise ValueError(f"Unknown extractor type: {extractor_type}")
        self.extractor = self.extractor.to(device)
        self.device = next(self.extractor.parameters()).device

        ### Settings for extractor
        self.deform = deform
        self.verbose = verbose

        # if using fixed_box:
        if not fixed_box:
            raise NotImplementedError("Currently only fixed_box=True is supported")

        self._fixed_box = fixed_box
        self._grid = None
        self._resolution = None



    def set_grid(self, bbox: list[list[int]], resolution: int):
        if self._fixed_box:
            self._grid=create_bbox_grid(bbox,resolution,self.device)
            self._resolution = resolution
        else:
            raise NotImplementedError("Currently only fixed_box=True is supported")



    def extract_mesh(self, model: callable, isovalue: float = 0.0,batch_size: int = 2048):
        if self._fixed_box and (self._grid is None or self._resolution is None):
            raise Warning("Extractor grid for fixed grid not set, please run set_grid once before extracting mesh")

        #Sample SDF on grid
        sdf_values = sample_sdf(model, self._grid, batch_size=batch_size)
        #Reshape SDF values to grid
        sdf_grid = sdf_samples_to_grid(sdf_values, self._resolution)
        #Run DISO
        verts, faces = self.extractor(self._grid, sdf_grid, isovalue=isovalue)
        if self.verbose:
            print(f"Extracted mesh with {verts.shape[0]} vertices and {faces.shape[0]} faces")
        #Renormalise vertices back to original coordinate space
        verts = renormalise_vertices(verts, bbox=[[-5000, 5000], [-5000, 5000], [-5000, 5000]])
        return verts, faces
