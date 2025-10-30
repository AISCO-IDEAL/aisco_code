import datetime
import os
import numpy as np
import torch
from losses import SDFLosses  # Assuming you save the above in losses.py
from mesh_to_pointcloud import stl_to_point_cloud
from sdf import SDFNetwork
from utils import visualize_sdf  # Import the new utilities

username="aalthaus"  # Replace with your actual username


def main(load_pretrained_model: bool = False):
    if load_pretrained_model:
        # Load a pre-trained model
        load_dir = f"/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/dummy_models"
        model_path = os.path.join(load_dir, 'sdf_sphere_model_2025-10-24.pth')  # Specify the correct model filename
        model = torch.load(model_path)
        print(f"Model loaded from: {model_path}")

    else:
        model = SDFNetwork()
    loss_fn = SDFLosses(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Adjust weights as needed
    #WTF is going on here? we need points for our loss function though, what happened to them?
    """
    #################### ADD ASSEMBLY POINTS HERE ####################
    # Path to your STL file
    stl_file_path = '/cluster/home/eschaerer/AISCO-code/subprojects/WP2-Assembler/shells/shell0.stl'
    # Generate the point cloud
    point_cloud = stl_to_point_cloud(stl_file_path)
    # Convert to PyTorch tensor, ensure float32, and enable gradient tracking
    point_cloud = torch.from_numpy(point_cloud).float().requires_grad_(True)
    # Save the point cloud to a .npy file
    np.save('shell0_point_cloud.npy', point_cloud.detach().numpy())
    print(f"Point cloud generated and saved as 'shell0_point_cloud.npy' with {len(point_cloud)} points.")

    """
    ##################################################################

    #Replaced the above snippet with this as a quick fix
    points = torch.randn(1000, 3, requires_grad=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs= 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = loss_fn.get_total_loss(model, points)
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}")
    # Save the trained model
    save_dir = f"/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/main_models"
    os.makedirs(save_dir, exist_ok=True)
    datum = datetime.datetime.now()
    save_path = os.path.join(save_dir, f'sdf_model_{datum.strftime("%m_%d_%H_%M")}.pth')
    torch.save(model, save_path)
    print(f"Model saved to: {save_path}")

    # Visualize the SDF
    visualize_sdf(model, grid_size=32, threshold=0.0, filename="sdf_aeroshell_mesh_BiggerSphereStart.ply")

    print("Training complete. A mesh has been generated and saved as 'sdf_aeroshell_mesh.ply'. It can be viewed using 3D Point Cloud Visualizer (PLY, ...) from Florian Nick")

if __name__ == "__main__":
    main(load_pretrained_model=True)
