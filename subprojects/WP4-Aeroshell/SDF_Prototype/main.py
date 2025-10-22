import datetime
import torch
import os
from sdf import SDFNetwork
from losses import SDFLosses  # Assuming you save the above in losses.py
from utils import visualize_sdf  # Import the new utilities

username="aalthaus"  # Replace with your actual username

def main():
    model = SDFNetwork()
    loss_fn = SDFLosses(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Adjust weights as needed
    #################### ADD ASSEMBLY POINTS HERE ####################
    points = torch.randn(1000, 3, requires_grad=True)
    ##################################################################
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(2):
        optimizer.zero_grad()
        total_loss = loss_fn.get_total_loss(model, points)
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Loss: {total_loss.item():.4f}")

    # Save the trained model
    save_dir = "/cluster/home/{username}/AISCO-code/subprojects/WP4-Aeroshell/SDF_Prototype/main_models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'sdf_model_{datetime.date.today()}.pth')
    torch.save(model, save_path)
    print(f"Model saved to: {save_path}")
   

    # Visualize the SDF
    visualize_sdf(model, grid_size=32, threshold=0.0, filename="sdf_aeroshell_mesh.ply")




if __name__ == "__main__":
    main()

