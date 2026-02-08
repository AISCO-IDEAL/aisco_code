import diso
import torch

"""
This file is mainly used in diso_example
"""

def get_diff_mc(dtype : str= "float32") -> diso.DiffMC:
    """
    Returns isosurface extractor object based on Marching Cubes
    """
    if dtype == "float64":
        return diso.DiffMC(dtype=torch.float64).cuda()
    elif dtype == "float32":
        return diso.DiffMC(dtype=torch.float32).cuda()
    else:
        raise ValueError(f"We don't want to use dtype {dtype} for diso.DiffMC, use 'float32' or 'float64'")

def get_diff_dmc(dtype: str = "float32") -> diso.DiffDMC:
    """
    Returns isosurface extractor object based on Dual Marching Cubes
    """
    if dtype == "float64":
        return diso.DiffDMC(dtype=torch.float64).cuda()
    elif dtype == "float32":
        return diso.DiffDMC(dtype=torch.float32).cuda()
    else:
        raise ValueError(f"We don't want to use dtype {dtype} for diso.DiffDMC, use 'float32' or 'float64'")
