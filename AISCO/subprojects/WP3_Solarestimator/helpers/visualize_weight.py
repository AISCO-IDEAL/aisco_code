import matplotlib.cm as cm
import matplotlib.colors as colors
import trimesh
import numpy as np
def color_faces_by_weight(mesh: trimesh.Trimesh, weights: np.ndarray, gamma: float = 1.0) -> trimesh.Trimesh:
    """
    Colors the faces of a trimesh object based on a weight array (0.0 to 1.0)
    with a Green (0.0) to Red (1.0) gradient, using gamma correction to
    emphasize differences in a specific range.

    Args:
        mesh (trimesh.Trimesh): The mesh object to color.
        weights (np.ndarray): A 1D float array of shape (n_faces,) with 
                              values between 0.0 and 1.0.
        gamma (float): Controls the non-linear stretching of the color map.
                       > 1.0: Stretches high values (e.g., 0.9 to 1.0) for better visibility.
                       < 1.0: Stretches low values (e.g., 0.0 to 0.1).
                       = 1.0: Linear mapping (default).

    Returns:
        trimesh.Trimesh: The mesh with updated face colors.
    """
    
    if len(weights) != len(mesh.faces):
        raise ValueError(
            f"Weight array length ({len(weights)}) must match the number of faces ({len(mesh.faces)})."
        )
    
    # 1. Normalize and Clip Weights
    normalized_weights = np.clip(weights, 0.0, 1.0)

    # 2. Apply Gamma Correction
    stretched_weights = normalized_weights ** gamma
    # With gamma > 1.0, small numbers stay small (compressed), large numbers stay large (stretched).
    # E.g., for gamma=2: (0.1)^2 = 0.01, (0.9)^2 = 0.81. The high values retain more "color budget".

    # 3. Select Colormap
    # 'RdYlGn_r': Green (low) -> Yellow -> Red (high)
    colormap = cm.get_cmap('cool')#cm.get_cmap('RdYlGn_r')

    # 4. Map stretched weights to RGBA colors (0.0 to 1.0)
    rgba_float_colors = colormap(stretched_weights)

    # 5. Convert to uint8 RGBA (0-255)
    face_colors = (rgba_float_colors * 255).astype(np.uint8)

    # 6. Assign colors
    mesh.visual.face_colors = face_colors
    
    return mesh
