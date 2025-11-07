# render_cq_and_ply.py
import io
import os
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio

import cadquery as cq
from cadquery import exporters
from wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ

# =============================================================================
# CAMERA CONFIGURATION - Modify these variables to adjust camera position
# =============================================================================
CAMERA_EXTRA_DISTANCE = 0.0  # Extra distance in meters beyond auto-calculated distance
CAMERA_TILT_DOWN_DEGREES = -50.0  # Downward tilt angle in degrees (positive = tilt down)
CAMERA_AZIMUTH_DEGREES = 0.0  # Rotation around Z-axis in degrees (0 = default position)
CAMERA_ELEVATION_FACTOR = 0.4  # Height factor (0.0 = ground level, 1.0 = very high)
CAMERA_DISTANCE_FACTOR = 1.5  # Base distance multiplier from object radius
CAMERA_FOV_DEGREES = 65.0  # Field of view in degrees (larger = wider view)

# Additional camera positioning factors (for fine-tuning)
CAMERA_X_FACTOR = 1 #0.6  # X position factor (side-to-side)
CAMERA_Y_FACTOR = 1 #0.4  # Y position factor (front-to-back) 
CAMERA_Z_FACTOR = 1# 0.6  # Z position factor (up-down, before tilt is applied)
# =============================================================================

# Configure for headless rendering on HPC systems
def setup_headless_rendering():
    """Setup environment for headless rendering on HPC clusters."""
    # Set environment variables for headless operation
    os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Use EGL for headless rendering
    os.environ['DISPLAY'] = ':99'  # Virtual display
    
    # Try to start a virtual framebuffer if needed
    try:
        import subprocess
        # Check if Xvfb is available and not already running
        result = subprocess.run(['pgrep', 'Xvfb'], capture_output=True)
        if result.returncode != 0:  # Xvfb not running
            # Try to start Xvfb
            subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            import time
            time.sleep(1)  # Give Xvfb time to start
    except:
        pass  # If Xvfb fails, continue with EGL
    
    # Configure OpenGL for headless rendering
    try:
        import OpenGL
        OpenGL.ERROR_CHECKING = False
        OpenGL.ERROR_LOGGING = False
    except:
        pass

# Setup headless rendering before importing pyrender
setup_headless_rendering()


def _cq_solid_to_trimesh(solid: cq.Solid, tolerance: float = 0.1) -> trimesh.Trimesh:
    """Convert a CadQuery Solid to a trimesh.Trimesh via in-memory STL export."""
    import tempfile
    import os
    
    # Use temporary file approach to avoid bytes/string issues
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Method 1: Try using the newer CadQuery export API
        try:
            # Use Workplane export which is more reliable
            wp = cq.Workplane().add(solid)
            cq.exporters.export(wp, temp_path, tolerance=tolerance, angularTolerance=0.1)
        except (AttributeError, TypeError):
            # Method 2: Fallback to direct shape export to file
            try:
                with open(temp_path, 'w') as f:
                    stl_str = exporters.exportShape(solid, "STL", tolerance=tolerance, angularTolerance=0.1)
                    f.write(stl_str)
            except (TypeError, AttributeError):
                # Method 3: Use shape's built-in export if available
                try:
                    solid.exportStl(temp_path, tolerance=tolerance, angularTolerance=0.1)
                except (AttributeError, TypeError):
                    # Method 4: Last resort - use the wrapped shape
                    try:
                        with open(temp_path, 'w') as f:
                            stl_str = exporters.exportShape(solid.wrapped, "STL", tolerance=tolerance, angularTolerance=0.1)
                            f.write(stl_str)
                    except Exception as e:
                        raise RuntimeError(f"Failed to export CadQuery solid to STL: {e}")
        
        # Load the STL file using trimesh
        tm = trimesh.load(temp_path, file_type="stl")
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
        
        return tm
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _load_ply_as_trimesh(ply_path: str) -> trimesh.Trimesh:
    """Load a PLY file as a trimesh.Trimesh."""
    tm = trimesh.load(ply_path, file_type="ply")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
    return tm


def render_cq_solids_and_ply(
    solid_a: cq.Solid,
    solid_b: cq.Solid,
    ply_path: str,
    out_image_path: str = "render.png",
    image_size=(1200, 900),
    bg_color=(255, 255, 255, 0),
    colors=((1.0, 0.2, 0.2, 0.5), (0.2, 0.6, 1.0, 0.5), (0.3, 1.0, 0.4, 0.5)),
    camera_dist_factor: float = 2.2,
    ambient_intensity: float = 0.3,
    directional_intensity: float = 3.5,
    scale_factor: float = 0.001,  # Convert mm to m for better rendering
):
    """
    Render two CadQuery solids and one PLY mesh together with semi-transparency.

    Args:
        solid_a, solid_b: CadQuery solids (assumed to be in mm)
        ply_path: path to a .ply mesh file
        out_image_path: output image filename (e.g. render.png)
        colors: 3 RGBA tuples for solid A, solid B, and the PLY mesh
        scale_factor: Scale factor to apply to geometries (0.001 converts mm to m)
    """
    print(f"Starting render with PLY file: {ply_path}")
    
    # Check if PLY file exists
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    # Convert solids and load PLY with debugging
    print("Converting CadQuery solids to trimesh...")
    tm_a = _cq_solid_to_trimesh(solid_a)
    tm_b = _cq_solid_to_trimesh(solid_b)
    print(f"Solid A: {tm_a.vertices.shape[0]} vertices, {tm_a.faces.shape[0]} faces")
    print(f"Solid B: {tm_b.vertices.shape[0]} vertices, {tm_b.faces.shape[0]} faces")
    
    print("Loading PLY mesh...")
    tm_c = _load_ply_as_trimesh(ply_path)
    print(f"PLY mesh: {tm_c.vertices.shape[0]} vertices, {tm_c.faces.shape[0]} faces")
    
    # Check if meshes are valid
    if tm_a.vertices.shape[0] == 0 or tm_b.vertices.shape[0] == 0 or tm_c.vertices.shape[0] == 0:
        raise ValueError("One or more meshes is empty")
    
    # Print original bounding boxes for debugging
    print(f"Original Solid A bounds: {tm_a.bounds}")
    print(f"Original Solid B bounds: {tm_b.bounds}")
    print(f"Original PLY mesh bounds: {tm_c.bounds}")

    # Apply scale factor to convert mm to more reasonable units
    if scale_factor != 1.0:
        print(f"Applying scale factor: {scale_factor}")
        tm_a.vertices *= scale_factor
        tm_b.vertices *= scale_factor
        tm_c.vertices *= scale_factor
        
        print(f"Scaled Solid A bounds: {tm_a.bounds}")
        print(f"Scaled Solid B bounds: {tm_b.bounds}")
        print(f"Scaled PLY mesh bounds: {tm_c.bounds}")

    # Center all meshes at origin for better camera positioning
    print("Centering meshes...")
    combined_temp = trimesh.util.concatenate([tm_a, tm_b, tm_c])
    original_center = combined_temp.bounding_box.centroid
    print(f"Original combined center: {original_center}")
    
    # Translate all meshes to center them at origin
    tm_a.vertices -= original_center
    tm_b.vertices -= original_center  
    tm_c.vertices -= original_center
    
    print(f"Centered Solid A bounds: {tm_a.bounds}")
    print(f"Centered Solid B bounds: {tm_b.bounds}")
    print(f"Centered PLY mesh bounds: {tm_c.bounds}")

    # Apply additional uniform scaling to make objects larger in the view
    additional_scale = 2.0  # Make objects 2x larger
    print(f"Applying additional scale factor: {additional_scale}")
    tm_a.vertices *= additional_scale
    tm_b.vertices *= additional_scale
    tm_c.vertices *= additional_scale

    # Compute combined bounds for camera framing (should be centered at origin now)
    combined = trimesh.util.concatenate([tm_a, tm_b, tm_c])
    extents = combined.extents
    center = np.array([0.0, 0.0, 0.0])  # Should be at origin now
    radius = np.linalg.norm(extents) / 2.0  # Use half diagonal as radius
    
    # Use configurable camera distance and positioning
    cam_distance = radius * CAMERA_DISTANCE_FACTOR + CAMERA_EXTRA_DISTANCE
    
    print(f"Final combined bounds: {combined.bounds}")
    print(f"Center (should be ~[0,0,0]): {combined.bounding_box.centroid}")
    print(f"Extents: {extents}")
    print(f"Radius: {radius}")
    print(f"Camera distance: {cam_distance} (base: {radius * CAMERA_DISTANCE_FACTOR:.2f}m + extra: {CAMERA_EXTRA_DISTANCE}m)")

    # Validate geometry is reasonable
    if radius < 1e-6:
        raise ValueError(f"Geometry too small: radius = {radius}")

    # Create the pyrender scene
    scene = pyrender.Scene(bg_color=np.array(bg_color, dtype=np.uint8),
                          ambient_light=np.ones(3) * ambient_intensity)

    # Add meshes to the scene with proper transparency settings
    # For transparent materials, we need to use alphaMode='BLEND'
    material_a = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=colors[0], 
        metallicFactor=0.0, 
        roughnessFactor=0.5,
        alphaMode='OPAQUE' if colors[0][3] >= 1.0 else 'BLEND'
    )
    material_b = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=colors[1], 
        metallicFactor=0.0, 
        roughnessFactor=0.5,
        alphaMode='OPAQUE' if colors[1][3] >= 1.0 else 'BLEND'
    )
    material_c = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=colors[2], 
        metallicFactor=0.0, 
        roughnessFactor=0.5,
        alphaMode='OPAQUE' if colors[2][3] >= 1.0 else 'BLEND'
    )
    
    mesh_a = pyrender.Mesh.from_trimesh(tm_a, material=material_a)
    mesh_b = pyrender.Mesh.from_trimesh(tm_b, material=material_b)
    mesh_c = pyrender.Mesh.from_trimesh(tm_c, material=material_c)

    scene.add(mesh_a)
    scene.add(mesh_b) 
    scene.add(mesh_c)
    
    print(f"Material transparency settings:")
    print(f"  - Red shape (innerNBZ): alpha={colors[0][3]}, mode={'OPAQUE' if colors[0][3] >= 1.0 else 'BLEND'}")
    print(f"  - Blue shape (outerNBZ): alpha={colors[1][3]}, mode={'OPAQUE' if colors[1][3] >= 1.0 else 'BLEND'}")
    print(f"  - Green shape (PLY): alpha={colors[2][3]}, mode={'OPAQUE' if colors[2][3] >= 1.0 else 'BLEND'}")

    # Camera setup using configurable field of view
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(CAMERA_FOV_DEGREES),
                                       znear=max(0.001, cam_distance * 0.01),
                                       zfar=cam_distance * 20.0)
    
    # Calculate base camera position using configurable factors
    eye_x = cam_distance * CAMERA_X_FACTOR
    eye_y = cam_distance * CAMERA_Y_FACTOR
    eye_z = cam_distance * CAMERA_Z_FACTOR
    
    # Apply azimuth rotation (rotation around Z-axis)
    azimuth_rad = np.deg2rad(CAMERA_AZIMUTH_DEGREES)
    cos_azimuth = np.cos(azimuth_rad)
    sin_azimuth = np.sin(azimuth_rad)
    
    # Rotate X and Y coordinates for azimuth
    eye_x_rotated = eye_x * cos_azimuth - eye_y * sin_azimuth
    eye_y_rotated = eye_x * sin_azimuth + eye_y * cos_azimuth
    
    # Apply tilt (rotation around Y-axis for pitch/elevation)
    tilt_angle_rad = np.deg2rad(-CAMERA_TILT_DOWN_DEGREES)  # Negative for downward tilt
    cos_tilt = np.cos(tilt_angle_rad)
    sin_tilt = np.sin(tilt_angle_rad)
    
    # Apply tilt rotation to the rotated X and Z coordinates
    eye_x_final = eye_x_rotated * cos_tilt - eye_z * sin_tilt
    eye_z_final = eye_x_rotated * sin_tilt + eye_z * cos_tilt
    
    eye = np.array([eye_x_final, eye_y_rotated, eye_z_final])
    target = np.array([0.0, 0.0, 0.0])  # Look at origin
    
    print(f"Camera configuration:")
    print(f"  - Extra distance: {CAMERA_EXTRA_DISTANCE}m")
    print(f"  - Tilt down: {CAMERA_TILT_DOWN_DEGREES}°")
    print(f"  - Azimuth: {CAMERA_AZIMUTH_DEGREES}°")
    print(f"  - Field of view: {CAMERA_FOV_DEGREES}°")
    print(f"Camera eye position: {eye}")
    print(f"Camera target: {target}")
    print(f"Camera distance from target: {np.linalg.norm(eye - target):.2f}m")

    def look_at(eye, target, up=np.array([0, 0, 1], dtype=float)):  # Z-up for technical drawings
        f = target - eye
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-12:
            raise ValueError("Eye and target are too close")
        f = f / f_norm
        
        u = up / np.linalg.norm(up)
        s = np.cross(f, u)
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-12:
            # If up is parallel to f, choose a different up vector
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
    
    print(f"Camera eye position: {eye}")
    print(f"Camera target: {center}")

    # Simplified but bright lighting
    # Main directional light from camera direction
    key_light = pyrender.DirectionalLight(intensity=5.0)  # Brighter light
    scene.add(key_light, pose=cam_pose)
    
    # Fill lights from different angles
    for direction in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
        fill_pose = np.eye(4)
        fill_pose[:3, 3] = center + np.array(direction) * cam_distance * 0.5
        scene.add(pyrender.DirectionalLight(intensity=2.0), pose=fill_pose)

    # Render offscreen with fallback options
    width, height = image_size
    renderer = None
    color = None
    
    # Try multiple rendering backends
    for attempt in range(3):
        try:
            if attempt == 0:
                # First try: EGL platform
                os.environ['PYOPENGL_PLATFORM'] = 'egl'
                renderer = pyrender.OffscreenRenderer(width, height)
            elif attempt == 1:
                # Second try: OSMesa platform (software rendering)
                os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
                renderer = pyrender.OffscreenRenderer(width, height)
            else:
                # Third try: Default platform with virtual display
                os.environ.pop('PYOPENGL_PLATFORM', None)
                renderer = pyrender.OffscreenRenderer(width, height)
            
            color, _ = renderer.render(scene)
            break  # Success, exit the loop
            
        except Exception as e:
            print(f"Rendering attempt {attempt + 1} failed: {e}")
            if renderer:
                try:
                    renderer.delete()
                except:
                    pass
                renderer = None
            
            if attempt == 2:  # Last attempt failed
                raise RuntimeError(f"All rendering attempts failed. Last error: {e}")
    
    if renderer:
        renderer.delete()

    if color is None:
        raise RuntimeError("Failed to render scene")

    # Save
    imageio.imwrite(out_image_path, color)
    return out_image_path


# Example usage
if __name__ == "__main__":

    # Replace with your actual PLY file path
    ply_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh.ply"

    out = render_cq_solids_and_ply(
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        ply_path=ply_file,
        out_image_path="/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply.png",
        colors=((1.0, 0.0, 0.0, 1.0),      # Inner NBZ: Solid red (fully opaque)
                (1.0, 1.0, 0.0, 0.2),      # Outer NBZ: Yellow with 80% transparency (20% opaque)
                (0.0, 0.0, 1.0, 0.4)),     # PLY mesh: Blue with 60% transparency (40% opaque)
    )
    print(f"Saved render to: {out}")
