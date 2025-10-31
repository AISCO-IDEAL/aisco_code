import visualtorch
import torch
from sdf import SDFNetwork
import matplotlib.pyplot as plt

def visualize_network(model: SDFNetwork,
                      save_dir: str="AISCO-code/subprojects/wp4_aeroshell_optimizer/data/architecture_visualizations/",
                      filename: str="sdf_network_visualization"):
    """
    Visualizes the architecture of the SDFNetwork using visualtorch and saves the image.
    Args:
        model: The SDFNetwork model to visualize.
        save_dir: Directory to save the visualization image.
        filename: Name of the saved image file (without extension).
    Returns:
        None

    """
    # Create the visualization

    input_shape = (1, 3)  # Assuming input is a single point with 3 coordinates (x, y, z)
    img = visualtorch.graph_view(model, input_shape)

    plt.axis("off")
    plt.tight_layout()
    plt.imshow(img)
    plt.show()
    plt.savefig(f"{save_dir}/{filename}.png", bbox_inches='tight', pad_inches=0, dpi=300)
    return