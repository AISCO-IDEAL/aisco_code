
import time

import torch
import trimesh

from AISCO_code.subprojects.wp3_solar_estimator.mesh_extractors.diso.helpers.diso_fetcher import get_diff_dmc, get_diff_mc
from subprojects.wp4_aeroshell_optimizer.v1.src.models.basic_MLP_v1 import BasicMLP
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh


def load_mesh(filename:str = "FlexiCubesMesh.ply") -> trimesh.Trimesh:
    """
    Loads a mesh for the testing purposes.
    Based on files in /cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data
    """
    return trimesh.load_mesh(f"/cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data/{filename}")

def load_testing_model() -> BasicMLP:
    """
    Loads the fixed (!) model for the unit tests.
    Based on files in /cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data
    """
    model_config={
    "in_dim": 3,
    "hidden": 64,
    "layers": 6,
    }
    model = BasicMLP(**model_config)
    model.load_state_dict(torch.load("/cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data/test_model.pth"))
    return model

def save_testing_mesh(verts: torch.Tensor, faces: torch.Tensor,filename:str = "generic_DISO.ply"):
    """
    Saves a mesh for the testing purposes.
    Based on files in /cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data
    """

    mesh = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
    mesh.export(f"/cluster/work/fuge/AISCO/wp3_solar_estimator/diso_test_data/{filename}")


def main_test():
    #Load model
    model = load_testing_model()
    print(model)
    model= model.cuda()
    model.eval()
    #Get extractors
    diff_mc= get_diff_mc(dtype="float32")
    print(diff_mc)
    diff_dmc= get_diff_dmc(dtype="float32")
    print(diff_dmc)

    # Create a 3D grid for SDF evaluation
    # For coordinate space -5000 to 5000, using 64x64x64 grid
    dimx, dimy, dimz = 64, 64, 64
    grids = torch.stack(
        torch.meshgrid(
            torch.linspace(-5000, 5000, dimx),
            torch.linspace(-5000, 5000, dimy),
            torch.linspace(-5000, 5000, dimz),
            indexing="ij",
        ),
        dim=-1,
    ).cuda()

    # Reshape grid for model evaluation: (64, 64, 64, 3) -> (64*64*64, 3)
    grid_points = grids.reshape(-1, 3)

    # Evaluate SDF on the grid
    print("Starting SDF evaluation...")
    sdf_start_time = time.time()
    with torch.no_grad():
        sdf_values = model(grid_points)
    sdf_end_time = time.time()
    print(f"SDF evaluation took: {sdf_end_time - sdf_start_time:.3f} seconds")

    # Reshape back to grid: (64*64*64, 1) -> (64, 64, 64)
    sdf_grid = sdf_values.reshape(dimx, dimy, dimz).squeeze()
    sdf_grid.requires_grad_(True)

    print(f"SDF grid shape: {sdf_grid.shape}, min: {sdf_grid.min():.3f}, max: {sdf_grid.max():.3f}")

    #Extract meshes using the SDF grid
    print("Starting diff_mc extraction...")
    mc_start_time = time.time()
    mc_verts, mc_faces = diff_mc(sdf_grid, None, isovalue=0)
    mc_end_time = time.time()
    print(f"diff_mc extraction took: {mc_end_time - mc_start_time:.3f} seconds")
    # Scale vertices back to original coordinate space
    mc_verts = mc_verts.cpu() * 10000.0 - 5000.0  # from [0,1] to [-5000, 5000]
    save_testing_mesh(filename="diff_mc_output.ply", verts=mc_verts, faces=mc_faces)
    print("Shapes of MC verts/faces:",mc_verts.shape, mc_faces.shape)

    print("Starting diff_dmc extraction...")
    dmc_start_time = time.time()
    dmc_verts, dmc_faces = diff_dmc(sdf_grid, None, isovalue=0)
    dmc_end_time = time.time()
    print(f"diff_dmc extraction took: {dmc_end_time - dmc_start_time:.3f} seconds")
    # Scale vertices back to original coordinate space
    dmc_verts = dmc_verts.cpu() * 10000.0 - 5000.0  # from [0,1] to [-5000, 5000]
    save_testing_mesh(filename="diff_dmc_output.ply", verts=dmc_verts, faces=dmc_faces)
    print("Shapes of DMC verts/faces:",dmc_verts.shape, dmc_faces.shape)
    fc_start_time = time.time()
    fc_verts, fc_faces = get_diff_mesh(model, grid_res=64)
    fc_end_time = time.time()
    print(f"FC extraction took: {fc_end_time - fc_start_time:.3f} seconds")
    save_testing_mesh(filename="FC_output.ply", verts=fc_verts, faces=fc_faces)
    print("Shapes of FC verts/faces:",fc_verts.shape, fc_faces.shape)


if __name__ == "__main__":
    main_test()
    print("Testing script for diso_utils completed.")
