# render_cq_and_ply.py

# DATA PATHS
PLY_FILE_PATH = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh1.ply"
OUTPUT_IMAGE_PATH = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply.png"

# RENDERING PARAMETERS
IMAGE_SIZE = (1200, 900)
BG_COLOR = (255, 255, 255, 0)
DEFAULT_COLORS = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5))  # Red opaque, Green transparent, Blue transparent
SCALE_FACTOR = 0.001
ADDITIONAL_SCALE = 2.0
MESH_TOLERANCE = 0.1
ANGULAR_TOLERANCE = 0.1

# EDGE RENDERING PARAMETERS
SHOW_OUTER_NBZ_EDGES = True  # Only show edges for outerNBZ (green shape)
EDGE_COLOR = (0.0, 0.0, 0.0, 1.0)  # Black edges
EDGE_WIDTH = 5.0

# CAMERA PARAMETERS
CAMERA_EXTRA_DISTANCE = 0.0
CAMERA_TILT_DOWN_DEGREES = -50.0
CAMERA_AZIMUTH_DEGREES = 0.0
CAMERA_ELEVATION_FACTOR = 0.0
CAMERA_DISTANCE_FACTOR = 1.5
CAMERA_FOV_DEGREES = 65.0
CAMERA_X_FACTOR = 1
CAMERA_Y_FACTOR = 1
CAMERA_Z_FACTOR = 1

# LIGHTING PARAMETERS
AMBIENT_INTENSITY = 0.3
KEY_LIGHT_INTENSITY = 5.0
FILL_LIGHT_INTENSITY = 2.0
DIRECTIONAL_INTENSITY = 3.5

# MATERIAL PARAMETERS
METALLIC_FACTOR = 0.0
ROUGHNESS_FACTOR = 0.5

# DISPLAY SETTINGS
DISPLAY_PORT = ':99'
XVFB_SCREEN_SIZE = '1024x768x24'

import io
import os
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio

import cadquery as cq
from cadquery import exporters
from wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ

def setup_headless_rendering():
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['DISPLAY'] = DISPLAY_PORT
    try:
        import subprocess
        result = subprocess.run(['pgrep', 'Xvfb'], capture_output=True)
        if result.returncode != 0:
            subprocess.Popen(['Xvfb', DISPLAY_PORT, '-screen', '0', XVFB_SCREEN_SIZE], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(1)
    except:
        pass
    try:
        import OpenGL
        OpenGL.ERROR_CHECKING = False
        OpenGL.ERROR_LOGGING = False
    except:
        pass

setup_headless_rendering()

def _cq_solid_to_trimesh(solid: cq.Solid, tolerance: float = MESH_TOLERANCE) -> trimesh.Trimesh:
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
        temp_path = tmp_file.name
    try:
        try:
            wp = cq.Workplane().add(solid)
            cq.exporters.export(wp, temp_path, tolerance=tolerance, angularTolerance=ANGULAR_TOLERANCE)
        except (AttributeError, TypeError):
            try:
                with open(temp_path, 'w') as f:
                    stl_str = exporters.exportShape(solid, "STL", tolerance=tolerance, angularTolerance=ANGULAR_TOLERANCE)
                    f.write(stl_str)
            except (TypeError, AttributeError):
                try:
                    solid.exportStl(temp_path, tolerance=tolerance, angularTolerance=ANGULAR_TOLERANCE)
                except (AttributeError, TypeError):
                    try:
                        with open(temp_path, 'w') as f:
                            stl_str = exporters.exportShape(solid.wrapped, "STL", tolerance=tolerance, angularTolerance=ANGULAR_TOLERANCE)
                            f.write(stl_str)
                    except Exception as e:
                        raise RuntimeError(f"Failed to export CadQuery solid to STL: {e}")
        tm = trimesh.load(temp_path, file_type="stl")
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
        return tm
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def _load_ply_as_trimesh(ply_path: str) -> trimesh.Trimesh:
    tm = trimesh.load(ply_path, file_type="ply")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
    return tm

def render_cq_solids_and_ply(
    solid_a: cq.Solid,
    solid_b: cq.Solid,
    ply_path: str,
    out_image_path: str = "render.png",
    image_size=IMAGE_SIZE,
    bg_color=BG_COLOR,
    colors=DEFAULT_COLORS,
    camera_dist_factor: float = 2.2,
    ambient_intensity: float = AMBIENT_INTENSITY,
    directional_intensity: float = DIRECTIONAL_INTENSITY,
    scale_factor: float = SCALE_FACTOR,
):
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    tm_a = _cq_solid_to_trimesh(solid_a)
    tm_b = _cq_solid_to_trimesh(solid_b)
    tm_c = _load_ply_as_trimesh(ply_path)
    
    if tm_a.vertices.shape[0] == 0 or tm_b.vertices.shape[0] == 0 or tm_c.vertices.shape[0] == 0:
        raise ValueError("One or more meshes is empty")

    if scale_factor != 1.0:
        tm_a.vertices *= scale_factor
        tm_b.vertices *= scale_factor
        tm_c.vertices *= scale_factor

    combined_temp = trimesh.util.concatenate([tm_a, tm_b, tm_c])
    original_center = combined_temp.bounding_box.centroid
    
    tm_a.vertices -= original_center
    tm_b.vertices -= original_center  
    tm_c.vertices -= original_center

    tm_a.vertices *= ADDITIONAL_SCALE
    tm_b.vertices *= ADDITIONAL_SCALE
    tm_c.vertices *= ADDITIONAL_SCALE

    combined = trimesh.util.concatenate([tm_a, tm_b, tm_c])
    extents = combined.extents
    center = np.array([0.0, 0.0, 0.0])
    radius = np.linalg.norm(extents) / 2.0
    
    cam_distance = radius * CAMERA_DISTANCE_FACTOR + CAMERA_EXTRA_DISTANCE
    
    if radius < 1e-6:
        raise ValueError(f"Geometry too small: radius = {radius}")

    scene = pyrender.Scene(bg_color=np.array(bg_color, dtype=np.uint8),
                          ambient_light=np.ones(3) * ambient_intensity)

    material_a = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=colors[0], 
        metallicFactor=METALLIC_FACTOR, 
        roughnessFactor=ROUGHNESS_FACTOR,
        alphaMode='OPAQUE' if colors[0][3] >= 1.0 else 'BLEND'
    )
    material_b = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=colors[1], 
        metallicFactor=METALLIC_FACTOR, 
        roughnessFactor=ROUGHNESS_FACTOR,
        alphaMode='OPAQUE' if colors[1][3] >= 1.0 else 'BLEND'
    )
    material_c = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=colors[2], 
        metallicFactor=METALLIC_FACTOR, 
        roughnessFactor=ROUGHNESS_FACTOR,
        alphaMode='OPAQUE' if colors[2][3] >= 1.0 else 'BLEND'
    )
    
    mesh_a = pyrender.Mesh.from_trimesh(tm_a, material=material_a)
    mesh_b = pyrender.Mesh.from_trimesh(tm_b, material=material_b)
    mesh_c = pyrender.Mesh.from_trimesh(tm_c, material=material_c)

    scene.add(mesh_a)
    scene.add(mesh_b) 
    scene.add(mesh_c)

    if SHOW_OUTER_NBZ_EDGES:
        edges_b = tm_b.edges_unique
        edge_vertices_b = tm_b.vertices[edges_b]
        
        # Create individual cylinder meshes for each edge
        for edge in edge_vertices_b:
            start_point = edge[0]
            end_point = edge[1]
            
            # Calculate edge vector and length
            edge_vector = end_point - start_point
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 1e-6:  # Only create edge if it has meaningful length
                # Create a thin cylinder along the edge
                cylinder = trimesh.creation.cylinder(
                    radius=0.002,  # Very thin radius
                    height=edge_length,
                    sections=8  # Low polygon count for performance
                )
                
                # Orient and position the cylinder along the edge
                if edge_length > 0:
                    # Calculate rotation to align cylinder with edge
                    z_axis = np.array([0, 0, 1])
                    edge_unit = edge_vector / edge_length
                    
                    if not np.allclose(edge_unit, z_axis) and not np.allclose(edge_unit, -z_axis):
                        rotation_axis = np.cross(z_axis, edge_unit)
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        rotation_angle = np.arccos(np.clip(np.dot(z_axis, edge_unit), -1, 1))
                        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
                    elif np.allclose(edge_unit, -z_axis):
                        # Special case: edge points in -z direction
                        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
                    else:
                        # Edge already aligned with z-axis
                        rotation_matrix = np.eye(4)
                    
                    # Position at edge midpoint
                    midpoint = (start_point + end_point) / 2
                    rotation_matrix[:3, 3] = midpoint
                    
                    cylinder.apply_transform(rotation_matrix)
                
                # Create pyrender mesh and add to scene
                edge_mesh = pyrender.Mesh.from_trimesh(cylinder, material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=EDGE_COLOR, 
                    metallicFactor=0.0, 
                    roughnessFactor=1.0
                ))
                scene.add(edge_mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(CAMERA_FOV_DEGREES),
                                       znear=max(0.001, cam_distance * 0.01),
                                       zfar=cam_distance * 20.0)
    
    eye_x = cam_distance * CAMERA_X_FACTOR
    eye_y = cam_distance * CAMERA_Y_FACTOR
    eye_z = cam_distance * CAMERA_Z_FACTOR
    
    azimuth_rad = np.deg2rad(CAMERA_AZIMUTH_DEGREES)
    cos_azimuth = np.cos(azimuth_rad)
    sin_azimuth = np.sin(azimuth_rad)
    
    eye_x_rotated = eye_x * cos_azimuth - eye_y * sin_azimuth
    eye_y_rotated = eye_x * sin_azimuth + eye_y * cos_azimuth
    
    tilt_angle_rad = np.deg2rad(-CAMERA_TILT_DOWN_DEGREES)
    cos_tilt = np.cos(tilt_angle_rad)
    sin_tilt = np.sin(tilt_angle_rad)
    
    eye_x_final = eye_x_rotated * cos_tilt - eye_z * sin_tilt
    eye_z_final = eye_x_rotated * sin_tilt + eye_z * cos_tilt
    
    eye = np.array([eye_x_final, eye_y_rotated, eye_z_final])
    target = np.array([0.0, 0.0, 0.0])

    def look_at(eye, target, up=np.array([0, 0, 1], dtype=float)):
        f = target - eye
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-12:
            raise ValueError("Eye and target are too close")
        f = f / f_norm
        
        u = up / np.linalg.norm(up)
        s = np.cross(f, u)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-12:
            up = np.array([1, 0, 0], dtype=float) if abs(f[0]) < 0.9 else np.array([0, 1, 0], dtype=float)
            u = up / np.linalg.norm(up)
            s = np.cross(f, u)
            s_norm = np.linalg.norm(s)
        s = s / s_norm
        u = np.cross(s, f)
        
        m = np.eye(4)
        m[0, :3], m[1, :3], m[2, :3] = s, u, -f
        m[:3, 3] = eye
        return m

    cam_pose = look_at(eye, target)
    scene.add(camera, pose=cam_pose)

    key_light = pyrender.DirectionalLight(intensity=KEY_LIGHT_INTENSITY)
    scene.add(key_light, pose=cam_pose)
    
    for direction in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = center + np.array(direction) * cam_distance * 0.5
        scene.add(pyrender.DirectionalLight(intensity=FILL_LIGHT_INTENSITY), pose=fill_pose)

    width, height = image_size
    renderer = None
    color = None
    
    for attempt in range(3):
        try:
            if attempt == 0:
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                renderer = pyrender.OffscreenRenderer(width, height)
            elif attempt == 1:
                os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                renderer = pyrender.OffscreenRenderer(width, height)
            else:
                os.environ.pop('PYOPENGL_PLATFORM', None)
                renderer = pyrender.OffscreenRenderer(width, height)
            
            color, _ = renderer.render(scene)
            break
            
        except Exception as e:
            if renderer:
                try:
                    renderer.delete()
                except:
                    pass
                renderer = None
            
            if attempt == 2:
                raise RuntimeError(f"All rendering attempts failed. Last error: {e}")
    
    if renderer:
        renderer.delete()

    if color is None:
        raise RuntimeError("Failed to render scene")

    imageio.imwrite(out_image_path, color)
    return out_image_path

if __name__ == "__main__":
    out = render_cq_solids_and_ply(
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        ply_path=PLY_FILE_PATH,
        out_image_path=OUTPUT_IMAGE_PATH,
        colors=DEFAULT_COLORS,
    )