import math
from pathlib import Path

import imageio
import numpy as np
import torch
import time
from subprojects.fixed_renderer import Renderer as NeuralRenderer
from subprojects.wp4_aeroshell_optimizer.v1.src.main_files.pretraining_v1_1 import pretraining
from subprojects.wp4_aeroshell_optimizer.v1.src.models.model_baseclass import BaseModel
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.config_utils import load_config
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.dynamic_loading import dynamic_type_loading
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.hashing import dict_hashing
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.kaolin_conversions import get_diff_mesh
from subprojects.wp4_aeroshell_optimizer.v1.src.utils.storage_utils import allround_save_data, fetch_pretrained_model


def create_dummy_textures(verts, faces, texture_size: int = 2, color=(1.0, 1.0, 1.0), device=None, requires_grad: bool = False):
        """
        Create a dummy per-face 3D texture tensor expected by `fixed_renderer`.

        fixed_renderer expects textures shaped:
            [B, F, ts, ts, ts, 3]
        where `ts` is the 3D texture resolution per axis and 3 is RGB.

        Args:
            verts: tensor [B, V, 3] (used to infer device if not provided)
            faces: tensor [B, F, 3]
            texture_size: int, texture resolution `ts` (small values like 2 or 4 are fine)
            color: tuple of 3 floats (RGB) or single float to fill texture
            device: torch device string or torch.device
            requires_grad: whether textures.requires_grad

        Returns:
            textures: torch.FloatTensor on `device` with shape [B, F, ts, ts, ts, 3]
        """

        if device is None:
            device = verts.device
        B = faces.shape[0]
        F = faces.shape[1]
        ts = int(texture_size)
        # fill value
        if isinstance(color, (int, float)):
            fill = float(color)
            base = torch.full((B, F, ts, ts, ts, 3), fill, dtype=torch.float32, device=device)
        else:
            rgb = torch.tensor(color, dtype=torch.float32, device=device).view(1, 1, 1, 1, 1, 3)
            base = torch.ones((B, F, ts, ts, ts, 3), dtype=torch.float32, device=device) * rgb
        base.requires_grad = requires_grad
        return base
def process_image(images_out, alpha_out):
    image_out_export = 255*images_out.detach().cpu().numpy()[0].transpose((1, 2, 0))  # [image_size, image_size, RGB]
    alpha_out_export = 255*alpha_out.detach().cpu().numpy()[0]
    image_out_export = np.concatenate( (image_out_export, alpha_out_export[:,:,np.newaxis]), -1 )
    return image_out_export.astype(np.uint8)
def get_projection(az, el, distance, focal_length=35, img_w=256, img_h=256, sensor_size_mm = 32.):
    """Calculate 4x3 3D to 2D projection matrix given viewpoint parameters."""

    # Calculate intrinsic matrix.
    f_u = focal_length * img_w  / sensor_size_mm
    f_v = focal_length * img_h  / sensor_size_mm
    u_0 = img_w / 2
    v_0 = img_h / 2
    K = np.matrix(((f_u, 0, u_0), (0, f_v, v_0), (0, 0, 1)))

    # Calculate rotation and translation matrices.
    sa = np.sin(np.radians(az))
    ca = np.cos(np.radians(az))
    R_azimuth = np.transpose(np.matrix(((ca, 0, sa),
                                          (0, 1, 0),
                                          (-sa, 0, ca))))
    se = np.sin(np.radians(el))
    ce = np.cos(np.radians(el))
    R_elevation = np.transpose(np.matrix(((1, 0, 0),
                                          (0, ce, -se),
                                          (0, se, ce))))
    # fix up camera
    se = np.sin(np.radians(180))
    ce = np.cos(np.radians(180))
    R_cam = np.transpose(np.matrix(((ce, -se, 0),
                                          (se, ce, 0),
                                          (0, 0, 1))))
    T_world2cam = np.transpose(np.matrix((0,
                                           0,
                                           distance)))
    RT = np.hstack((R_cam@R_elevation@R_azimuth, T_world2cam))

    return K, RT
def store_image(image_filename, images_out, alpha_out):
    image_out_export = process_image(images_out, alpha_out)
    imageio.imwrite(image_filename, image_out_export)

def main(verbose: bool = True) -> None:
    # Get general config
    # parameters
    angle_deg = 30.0
    dist = 2000.0
    azimuth_deg = 0.0   # change if you want camera rotated around origin

    # compute eye position (elevation = angle_deg above XY plane)
    phi = math.radians(angle_deg)
    theta = math.radians(azimuth_deg)
    x = dist * math.cos(phi) * math.cos(theta)
    y = dist * math.cos(phi) * math.sin(theta)   # if you want camera offset in Y instead of X, swap axes
    z = dist * math.sin(phi)
    eye = torch.tensor([x, y, z], dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu").unsqueeze(0)

    # configure renderer
                   # look_at() uses this directly
    intrinsic, extrinsic = get_projection(azimuth_deg, angle_deg, dist,img_h=512, img_w=512)

    # set up renderer
    K_cuda = torch.tensor(intrinsic[np.newaxis, :, :].copy()).float().cuda().unsqueeze(0)
    R_cuda = torch.tensor(extrinsic[np.newaxis, 0:3, 0:3].copy()).float().cuda().unsqueeze(0)
    t_cuda = torch.tensor(extrinsic[np.newaxis, np.newaxis, 0:3, 3].copy()).float().cuda().unsqueeze(0)
    NR = NeuralRenderer(K=K_cuda, R=R_cuda, t=t_cuda, anti_aliasing=False)
    NR.camera_mode = 'look_at'
    NR.perspective = True            # if you want perspective projection
    NR.viewing_angle = 60.0         # camera field-of-view in degrees (change as needed)
    NR.eye = eye
    # Ensure near/far planes include the object at distance `dist` (2000 units)
    NR.near = 0.1
    NR.far = max(10000.0, dist * 10.0)

    # Load model to test rendering, copied this part from main_v1.1_one_model.py

    general_config = load_config(v="v1", cfg_name="general_config.yaml")
    model_config = load_config(v="v1", cfg_name=general_config["active_configs"]["model_params"])
    if verbose:
        print("Successfully loaded general and model config")

    # Get locations of model definition
    model_name = general_config["model_def"]["model_name"]
    model_definition = general_config["model_def"]["model_file_path"]

    # Import model and get <class 'type'>
    ModelClass = dynamic_type_loading(file_path=model_definition, class_name=model_name)  # noqa: N806

    # Initialise model with config,important that all keys match to a keyword argument in the model __init__()
    model = ModelClass(**model_config)
    if verbose:
        print(f"Successfully initialised model: {model_name}")

    # Move model to device
    device = general_config["general_info"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Check if there is a pretrained model already
    # Note that for now this only includes pretrained spheres

    pretrained_model = fetch_pretrained_model(
        model, pretrained_path=general_config["storage_paths"].get("pretrained_dir", None), verbose=verbose
    )
    if pretrained_model is not None:
        model.load_state_dict(torch.load(pretrained_model, map_location=device))
        if verbose:
            print(f"Loaded pretrained model from {pretrained_model}")
    else:
        if verbose:
            print("No pretrained model found, proceeding by pretraining")
        model = pretraining(model=model)
        # Save pretrained model and mesh for sanity check
        allround_save_data(
            model=model,
            storage_path=general_config["storage_paths"].get(
                "pretrained_dir", "/cluster/work/fuge/AISCO/wp4_aeroshell_optimiser/pretrained_models"
            ),
            save_model=True,
            save_mesh=True,
            verbose=verbose,
            pretraining=True,
        )

    # Extract mesh from model and render it using NeuralRenderer

    print("Storing image...")
    # Also track the time it took from mesh extraction to rendering
    start_time = time.time()
    verts, faces =get_diff_mesh(model, device=device, grid_res=96)
    # convert to batch format
    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)
    mesh_time = time.time()
    print(f"Mesh extraction and conversion took {mesh_time - start_time:.2f} seconds")



    # Example: simple white texture per face with tiny 2x2x2 3D texture
    textures = create_dummy_textures(verts, faces, texture_size=2, color=(1.0, 1.0, 1.0), device=device)

    # Render RGB + depth + alpha using render() which returns (rgb, depth, alpha)
    images_out, depth_out, alpha_out = NR.render(verts, faces, textures)
    store_image(Path("/cluster/home/aalthaus/test_rendering_output_normal.png"), images_out, alpha_out)

    # Also produce a depth-only render if desired
    #depth_images = NR.render_depth(verts, faces)
    #store_image("/cluster/home/aalthaus/test_rendering_output_depth.png", depth_images, depth_images)
    # Also do it RGB
    #rgb_images = NR.render_rgb(verts, faces, textures)
    #store_image("/cluster/home/aalthaus/test_rendering_output_rgb_only.png", rgb_images, rgb_images)

    end_time_rendering = time.time()
    print(f"Rendering took {end_time_rendering - mesh_time:.2f} seconds")

    print("Done")


if __name__ == "__main__":
    main()
