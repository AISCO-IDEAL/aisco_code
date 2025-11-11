# render_cq_and_ply.py

import io
import os
import numpy as np
import trimesh
import pyrender
from pyrender import RenderFlags
import imageio.v2 as imageio
from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import cadquery as cq
from cadquery import exporters
from wp2_assembler.version1.components.nbz import innerNBZ, outerNBZ

class CADVisualizer:
    """
    A class for visualizing CadQuery solids and PLY meshes with flexible rendering options.
    """
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (1200, 900),
                 bg_color: Tuple[int, int, int, int] = (255, 255, 255, 0),
                 scale_factor: float = 0.001,
                 additional_scale: float = 2.0,
                 mesh_tolerance: float = 0.1,
                 angular_tolerance: float = 0.1):
        """
        Initialize the CAD Visualizer.
        
        Args:
            image_size: Output image dimensions (width, height)
            bg_color: Background color as RGBA tuple
            scale_factor: Scale factor for converting units (e.g., mm to m)
            additional_scale: Additional scaling for better visualization
            mesh_tolerance: Tolerance for mesh generation
            angular_tolerance: Angular tolerance for mesh generation
        """
        # Rendering parameters
        self.image_size = image_size
        self.bg_color = bg_color
        self.scale_factor = scale_factor
        self.additional_scale = additional_scale
        self.mesh_tolerance = mesh_tolerance
        self.angular_tolerance = angular_tolerance
        
        # Camera parameters (defaults)
        self.camera_distance_factor = 1.5
        self.camera_fov_degrees = 65.0
        self.camera_tilt_down_degrees = -50.0
        self.camera_azimuth_degrees = 0.0
        self.camera_position_factors = (1, 1, 1)  # x, y, z factors
        
        # Lighting parameters
        self.ambient_intensity = 0.3
        self.key_light_intensity = 5.0
        self.fill_light_intensity = 2.0
        
        # Material parameters
        self.metallic_factor = 0.0
        self.roughness_factor = 0.5
        
        # Edge rendering parameters
        self.edge_color = (0.0, 0.0, 0.0, 1.0)
        self.edge_radius = 0.002
        
        # Display settings
        self.display_port = ':99'
        self.xvfb_screen_size = '1024x768x24'
        
        # Initialize headless rendering
        self._setup_headless_rendering()
    
    def _setup_headless_rendering(self):
        """Setup environment for headless rendering on HPC clusters."""
        # Set comprehensive OpenGL environment variables for headless rendering
        os.environ['PYOPENGL_PLATFORM'] = 'osmesa'  # Start with osmesa as primary
        os.environ['DISPLAY'] = self.display_port
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
        os.environ['MESA_LOADER_DRIVER_OVERRIDE'] = 'llvmpipe'
        
        # Additional environment variables for robustness
        os.environ['EGL_PLATFORM'] = 'surfaceless'
        os.environ['MESA_GLAPI_NO_REMAP'] = '1'
        
        # Force deterministic behavior
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)  # Ensure deterministic mesh processing
        
        try:
            import subprocess
            # Kill any existing Xvfb processes to start fresh
            try:
                subprocess.run(['pkill', '-f', 'Xvfb'], capture_output=True)
                import time
                time.sleep(1)
            except:
                pass
            
            # Start Xvfb with comprehensive options
            xvfb_cmd = [
                'Xvfb', self.display_port, 
                '-screen', '0', self.xvfb_screen_size,
                '-ac', '+extension', 'GLX', '+render', '-noreset',
                '-nolisten', 'tcp', '-dpi', '96'
            ]
            subprocess.Popen(xvfb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)  # Give more time for Xvfb to start
        except:
            pass
        
        try:
            import OpenGL
            OpenGL.ERROR_CHECKING = False
            OpenGL.ERROR_LOGGING = False
            OpenGL.FULL_LOGGING = False
        except:
            pass

        # Pre-configure PyRender for headless operation
        try:
            # Import pyrender and force platform detection
            import pyrender
            # Set the platform preference before any rendering operations
            os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Start with egl as it's most reliable on HPC
            
            # Try to initialize a dummy renderer to force platform detection
            platforms_to_try = ['egl', 'osmesa', 'glx']
            successful_platform = None
            
            for platform in platforms_to_try:
                try:
                    os.environ['PYOPENGL_PLATFORM'] = platform
                    test_renderer = pyrender.OffscreenRenderer(1, 1)
                    test_renderer.delete()
                    successful_platform = platform
                    print(f"Successfully initialized PyRender with {platform} platform")
                    break
                except Exception as platform_error:
                    # Only show detailed errors for debugging if needed
                    if "OSMesa" in str(platform_error):
                        # This is the common OSMesa attribute error, handle silently
                        continue
                    # For other errors, you can uncomment the next line for debugging
                    # print(f"Platform {platform} failed: {platform_error}")
                    continue
            
            if not successful_platform:
                print("Warning: No PyRender platform could be initialized during setup, will attempt during rendering")
                    
        except Exception as e:
            print(f"Warning: Could not pre-configure PyRender platforms: {e}")
    
    def set_camera_params(self, 
                         distance_factor: float = None,
                         fov_degrees: float = None,
                         tilt_down_degrees: float = None,
                         azimuth_degrees: float = None,
                         position_factors: Tuple[float, float, float] = None):
        """
        Set camera parameters for different viewing angles.
        
        Args:
            distance_factor: Multiplier for camera distance from object
            fov_degrees: Field of view in degrees
            tilt_down_degrees: Downward tilt angle in degrees
            azimuth_degrees: Rotation around Z-axis in degrees
            position_factors: (x, y, z) position factors
        """
        if distance_factor is not None:
            self.camera_distance_factor = distance_factor
        if fov_degrees is not None:
            self.camera_fov_degrees = fov_degrees
        if tilt_down_degrees is not None:
            self.camera_tilt_down_degrees = tilt_down_degrees
        if azimuth_degrees is not None:
            self.camera_azimuth_degrees = azimuth_degrees
        if position_factors is not None:
            self.camera_position_factors = position_factors
    
    def set_lighting(self, 
                    ambient_intensity: float = None,
                    key_light_intensity: float = None,
                    fill_light_intensity: float = None):
        """Set lighting parameters."""
        if ambient_intensity is not None:
            self.ambient_intensity = ambient_intensity
        if key_light_intensity is not None:
            self.key_light_intensity = key_light_intensity
        if fill_light_intensity is not None:
            self.fill_light_intensity = fill_light_intensity
    
    def _cq_solid_to_trimesh(self, solid: cq.Solid) -> trimesh.Trimesh:
        """Convert a CadQuery Solid to a trimesh.Trimesh via STL export."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        try:
            try:
                wp = cq.Workplane().add(solid)
                cq.exporters.export(wp, temp_path, tolerance=self.mesh_tolerance, angularTolerance=self.angular_tolerance)
            except (AttributeError, TypeError):
                try:
                    with open(temp_path, 'w') as f:
                        stl_str = exporters.exportShape(solid, "STL", tolerance=self.mesh_tolerance, angularTolerance=self.angular_tolerance)
                        f.write(stl_str)
                except (TypeError, AttributeError):
                    try:
                        solid.exportStl(temp_path, tolerance=self.mesh_tolerance, angularTolerance=self.angular_tolerance)
                    except (AttributeError, TypeError):
                        try:
                            with open(temp_path, 'w') as f:
                                stl_str = exporters.exportShape(solid.wrapped, "STL", tolerance=self.mesh_tolerance, angularTolerance=self.angular_tolerance)
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
    
    def _load_ply_as_trimesh(self, ply_path: str) -> trimesh.Trimesh:
        """Load PLY file with robust error handling and retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tm = trimesh.load(ply_path, file_type="ply")
                if isinstance(tm, trimesh.Scene):
                    tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))

                # Validate mesh has data
                if tm.vertices.shape[0] == 0:
                    raise ValueError(f"PLY mesh is empty: {ply_path}")

                # Robustness fixes with error handling
                try:
                    tm.remove_infinite_values()
                    tm.remove_unreferenced_vertices()
                    
                    # Only apply repairs if mesh has faces
                    if len(tm.faces) > 0:
                        trimesh.repair.fix_inversion(tm)
                        trimesh.repair.fix_normals(tm)
                        try:
                            trimesh.repair.fill_holes(tm)
                        except:
                            pass  # Some meshes can't have holes filled
                        
                        # Validate and process mesh
                        tm.process(validate=True)
                        
                        # Ensure mesh has proper normals
                        if tm.face_normals is None or len(tm.face_normals) == 0:
                            tm.fix_normals()
                        
                        # For problematic meshes, ensure double-sided rendering will work
                        if not tm.is_watertight:
                            # Check if normals are consistently oriented
                            if hasattr(tm, 'face_normals') and tm.face_normals is not None:
                                # Don't invert unless we're sure it's needed
                                pass
                    
                    return tm
                    
                except Exception as repair_error:
                    # If repair fails, return the basic mesh
                    if tm.vertices.shape[0] > 0:
                        return tm
                    else:
                        raise repair_error
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to load PLY file {ply_path} after {max_retries} attempts: {e}")
                # Wait before retry
                import time
                time.sleep(0.1)
        
        raise RuntimeError(f"Failed to load PLY file: {ply_path}")

    def _add_edges_to_scene(self, scene: pyrender.Scene, mesh: trimesh.Trimesh):
        """Add wireframe edges to the scene for a given mesh."""
        edges = mesh.edges_unique
        edge_vertices = mesh.vertices[edges]
        
        for edge in edge_vertices:
            start_point = edge[0]
            end_point = edge[1]
            
            edge_vector = end_point - start_point
            edge_length = np.linalg.norm(edge_vector)
            
            if edge_length > 1e-6:
                cylinder = trimesh.creation.cylinder(
                    radius=self.edge_radius,
                    height=edge_length,
                    sections=8
                )
                
                if edge_length > 0:
                    z_axis = np.array([0, 0, 1])
                    edge_unit = edge_vector / edge_length
                    
                    if not np.allclose(edge_unit, z_axis) and not np.allclose(edge_unit, -z_axis):
                        rotation_axis = np.cross(z_axis, edge_unit)
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        rotation_angle = np.arccos(np.clip(np.dot(z_axis, edge_unit), -1, 1))
                        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, rotation_axis)
                    elif np.allclose(edge_unit, -z_axis):
                        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
                    else:
                        rotation_matrix = np.eye(4)
                    
                    midpoint = (start_point + end_point) / 2
                    rotation_matrix[:3, 3] = midpoint
                    cylinder.apply_transform(rotation_matrix)
                
                edge_mesh = pyrender.Mesh.from_trimesh(cylinder, material=pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=self.edge_color, 
                    metallicFactor=0.0, 
                    roughnessFactor=1.0
                ))
                scene.add(edge_mesh)
    
    def _setup_camera_and_lighting(self, scene: pyrender.Scene, radius: float):
        """Setup camera and lighting for the scene."""
        cam_distance = radius * self.camera_distance_factor
        
        camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(self.camera_fov_degrees),
                                           znear=max(0.001, cam_distance * 0.01),
                                           zfar=cam_distance * 20.0)
        
        # Calculate camera position
        eye_x = cam_distance * self.camera_position_factors[0]
        eye_y = cam_distance * self.camera_position_factors[1]
        eye_z = cam_distance * self.camera_position_factors[2]
        
        # Apply azimuth rotation
        azimuth_rad = np.deg2rad(self.camera_azimuth_degrees)
        cos_azimuth = np.cos(azimuth_rad)
        sin_azimuth = np.sin(azimuth_rad)
        
        eye_x_rotated = eye_x * cos_azimuth - eye_y * sin_azimuth
        eye_y_rotated = eye_x * sin_azimuth + eye_y * cos_azimuth
        
        # Apply tilt
        tilt_angle_rad = np.deg2rad(-self.camera_tilt_down_degrees)
        cos_tilt = np.cos(tilt_angle_rad)
        sin_tilt = np.sin(tilt_angle_rad)
        
        eye_x_final = eye_x_rotated * cos_tilt - eye_z * sin_tilt
        eye_z_final = eye_x_rotated * sin_tilt + eye_z * cos_tilt
        
        eye = np.array([eye_x_final, eye_y_rotated, eye_z_final])
        target = np.array([0.0, 0.0, 0.0])
        
        # Look-at matrix
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
        
        # Add lighting
        key_light = pyrender.DirectionalLight(intensity=self.key_light_intensity)
        scene.add(key_light, pose=cam_pose)
        
        # Fill lights
        center = np.array([0.0, 0.0, 0.0])
        for direction in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
            fill_pose = np.eye(4)
            fill_pose[:3, 3] = center + np.array(direction) * cam_distance * 0.5
            scene.add(pyrender.DirectionalLight(intensity=self.fill_light_intensity), pose=fill_pose)
    
    def render(self, 
               solid_a: Optional[cq.Solid] = None,
               solid_b: Optional[cq.Solid] = None,
               ply_path: Optional[str] = None,
               colors: List[Tuple[float, float, float, float]] = None,
               show_edges: List[bool] = None,
               out_image_path: str = "render.png") -> Tuple[str, bool]:
        """
        Render the specified objects with given colors and edge settings.
        
        Returns:
            Tuple[str, bool]: (image_path, success_flag) where success_flag indicates 
            if rendering was completed with the preferred pyrender engine (True) or 
            fallback matplotlib engine (False)
        """
        # Default colors and edge settings
        if colors is None:
            colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)]
        if show_edges is None:
            show_edges = [False, True, False]

        # Load and convert meshes with deterministic ordering
        meshes = []
        mesh_types = []  # Track mesh types for deterministic handling
        
        if solid_a is not None:
            meshes.append(self._cq_solid_to_trimesh(solid_a))
            mesh_types.append('solid_a')
        if solid_b is not None:
            meshes.append(self._cq_solid_to_trimesh(solid_b))
            mesh_types.append('solid_b')
        if ply_path is not None and os.path.exists(ply_path):
            meshes.append(self._load_ply_as_trimesh(ply_path))
            mesh_types.append('ply')
        
        if not meshes:
            raise ValueError("No valid objects to render")
        
        # Validate all meshes
        for i, mesh in enumerate(meshes):
            if mesh.vertices.shape[0] == 0:
                raise ValueError(f"Mesh {i} ({mesh_types[i]}) is empty")
        
        # Apply scaling consistently
        for mesh in meshes:
            if self.scale_factor != 1.0:
                mesh.vertices *= self.scale_factor
        
        # Center all meshes deterministically
        if len(meshes) > 1:
            # Use a stable method to calculate center
            all_vertices = np.vstack([mesh.vertices for mesh in meshes])
            original_center = np.mean(all_vertices, axis=0)  # More stable than bounding box centroid
            
            for mesh in meshes:
                mesh.vertices -= original_center
        
        # Apply additional scaling
        for mesh in meshes:
            mesh.vertices *= self.additional_scale
        
        # Calculate scene bounds
        combined = trimesh.util.concatenate(meshes)
        extents = combined.extents
        radius = np.linalg.norm(extents) / 2.0
        
        if radius < 1e-6:
            raise ValueError(f"Geometry too small: radius = {radius}")
        
        # Sort meshes by alpha value for proper transparency rendering
        # Opaque objects first, then transparent objects back-to-front
        mesh_data = list(zip(meshes, colors, show_edges, mesh_types))
        
        # Separate opaque and transparent objects
        opaque_meshes = [(m, c, e, t) for m, c, e, t in mesh_data if c[3] >= 0.99]
        transparent_meshes = [(m, c, e, t) for m, c, e, t in mesh_data if c[3] < 0.99]
        
        # Sort transparent meshes by distance from camera (approximate)
        camera_pos = np.array([0, 0, radius * self.camera_distance_factor])
        def mesh_distance(mesh_data):
            mesh, _, _, _ = mesh_data
            center = np.mean(mesh.vertices, axis=0)
            return np.linalg.norm(center - camera_pos)
        
        transparent_meshes.sort(key=mesh_distance, reverse=True)  # Back to front
        
        # Add meshes to scene in proper order
        all_ordered_meshes = opaque_meshes + transparent_meshes
        
        # Enhanced rendering with immediate fallback to matplotlib on GL context failure
        width, height = self.image_size
        
        # Try pyrender with multiple configurations
        pyrender_success = False
        color = None
        
        # Try multiple rendering approaches
        render_configs = [
            ('osmesa', RenderFlags.RGBA),
            ('egl', RenderFlags.RGBA),
            (None, RenderFlags.RGBA | RenderFlags.FLAT),
            (None, RenderFlags.RGBA)
        ]
        
        for attempt, (platform, flags) in enumerate(render_configs):
            try:
                # Set platform environment
                if platform:
                    os.environ['PYOPENGL_PLATFORM'] = platform
                elif 'PYOPENGL_PLATFORM' in os.environ:
                    os.environ.pop('PYOPENGL_PLATFORM')
                
                # Create scene with enhanced settings for transparency
                scene = pyrender.Scene(
                    bg_color=np.array(self.bg_color, dtype=np.uint8),
                    ambient_light=np.ones(3) * self.ambient_intensity
                )
                
                for mesh, color_val, show_edge, mesh_type in all_ordered_meshes:
                    # Enhanced material settings for better transparency
                    material = pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=color_val, 
                        metallicFactor=self.metallic_factor, 
                        roughnessFactor=self.roughness_factor,
                        alphaMode='OPAQUE' if color_val[3] >= 0.99 else 'BLEND',
                        doubleSided=True,  # Critical for transparent meshes
                        alphaCutoff=0.01   # Lower threshold for better transparency
                    )
                    
                    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                    scene.add(pyrender_mesh)
                    
                    # Add edges if requested
                    if show_edge:
                        self._add_edges_to_scene(scene, mesh)
                
                # Setup camera and lighting
                self._setup_camera_and_lighting(scene, radius)
                
                # Try to create renderer and render
                renderer = pyrender.OffscreenRenderer(width, height)
                color, _ = renderer.render(scene, flags=flags)
                renderer.delete()
                pyrender_success = True
                break
                
            except Exception as e:
                # Clean up any partial renderer
                try:
                    if 'renderer' in locals():
                        renderer.delete()
                except:
                    pass
                
                # Log the specific error for debugging
                print(f"Pyrender attempt {attempt + 1} failed with platform '{platform}': {e}")
                
                # If this is the last attempt, we'll fall back to matplotlib
                if attempt == len(render_configs) - 1:
                    print("All pyrender attempts failed, falling back to matplotlib rendering...")
                    break
        
        # If pyrender succeeded, save and return
        if pyrender_success and color is not None:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
            imageio.imwrite(out_image_path, color)
            return out_image_path, True
        
        # Fallback to matplotlib-based rendering
        print("Using matplotlib fallback rendering...")
        try:
            fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            # Set background color
            fig.patch.set_facecolor([c/255.0 for c in self.bg_color[:3]])
            ax.set_facecolor([c/255.0 for c in self.bg_color[:3]])
            
            # Render each mesh
            for mesh, color_val, show_edge, mesh_type in all_ordered_meshes:
                if len(mesh.faces) > 0:
                    # Create 3D polygon collection
                    poly3d = [[mesh.vertices[vert_idx] for vert_idx in face] for face in mesh.faces]
                    collection = Poly3DCollection(
                        poly3d, 
                        alpha=color_val[3], 
                        facecolor=color_val[:3],
                        edgecolor='black' if show_edge else color_val[:3],
                        linewidth=0.5 if show_edge else 0
                    )
                    ax.add_collection3d(collection)
                else:
                    # For point clouds or meshes without faces, use scatter plot
                    ax.scatter(
                        mesh.vertices[:, 0], 
                        mesh.vertices[:, 1], 
                        mesh.vertices[:, 2],
                        c=[color_val[:3]], 
                        alpha=color_val[3],
                        s=1
                    )
            
            # Set axis properties
            ax.set_xlim(-radius, radius)
            ax.set_ylim(-radius, radius)
            ax.set_zlim(-radius, radius)
            ax.set_box_aspect([1, 1, 1])
            
            # Set camera angle similar to pyrender
            ax.view_init(elev=self.camera_tilt_down_degrees, azim=self.camera_azimuth_degrees)
            
            # Remove axes for cleaner look
            ax.set_axis_off()
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
            
            # Save with high DPI and tight layout
            plt.savefig(
                out_image_path, 
                dpi=100, 
                bbox_inches='tight', 
                pad_inches=0,
                facecolor=fig.get_facecolor(),
                edgecolor='none'
            )
            plt.close(fig)
            
            return out_image_path, False
            
        except Exception as fallback_error:
            plt.close('all')  # Clean up any open figures
            raise RuntimeError(f"Both pyrender and matplotlib rendering failed. Matplotlib error: {fallback_error}")

    def has_blue_color(self, image_path: str, blue_threshold: float = 0.1, 
                      min_blue_ratio: float = 0.001) -> bool:
        """
        Check if the rendered image contains any significant blue color.
        
        Args:
            image_path: Path to the image file to analyze
            blue_threshold: Minimum blue channel value (0-1) to consider as "blue"
            min_blue_ratio: Minimum ratio of blue pixels required to return True
            
        Returns:
            bool: True if image contains significant blue color, False otherwise
        """
        try:
            # Load the image
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Read image using imageio (consistent with the render method)
            image = imageio.imread(image_path)
            
            # Convert to float and normalize to 0-1 range if needed
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            
            # Handle different image formats
            if len(image.shape) == 2:
                # Grayscale image - no blue channel
                return False
            elif len(image.shape) == 3:
                if image.shape[2] == 1:
                    # Single channel image
                    return False
                elif image.shape[2] >= 3:
                    # RGB or RGBA image - blue is channel 2
                    blue_channel = image[:, :, 2]
                    red_channel = image[:, :, 0]
                    green_channel = image[:, :, 1]
                    
                    # Find pixels where blue is dominant and above threshold
                    blue_dominant = (blue_channel > blue_threshold) & \
                                    (blue_channel > (red_channel+green_channel)/2) 
                    
                    # Calculate the ratio of blue pixels
                    total_pixels = blue_channel.size
                    blue_pixels = np.sum(blue_dominant)
                    blue_ratio = blue_pixels / total_pixels
                    
                    return blue_ratio >= min_blue_ratio
                else:
                    return False
            else:
                return False
                
        except Exception as e:
            print(f"Error analyzing image for blue color: {e}")
            return False

    def analyze_color_distribution(self, image_path: str) -> dict:
        """
        Analyze the color distribution in the rendered image.
        
        Args:
            image_path: Path to the image file to analyze
            
        Returns:
            dict: Dictionary containing color analysis results with keys:
                - 'has_red': bool
                - 'has_green': bool  
                - 'has_blue': bool
                - 'dominant_color': str ('red', 'green', 'blue', 'other')
                - 'color_ratios': dict with RGB ratios
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and normalize image
            image = imageio.imread(image_path)
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            
            # Initialize results
            results = {
                'has_red': False,
                'has_green': False,
                'has_blue': False,
                'dominant_color': 'other',
                'color_ratios': {'red': 0.0, 'green': 0.0, 'blue': 0.0}
            }
            
            if len(image.shape) != 3 or image.shape[2] < 3:
                return results
            
            # Extract color channels
            red_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            blue_channel = image[:, :, 2]
            
            # Define threshold for significant color presence
            threshold = 0.1
            min_ratio = 0.001
            
            # Find dominant pixels for each color
            red_dominant = (red_channel > threshold) & \
                          (red_channel > green_channel) & \
                          (red_channel > blue_channel)
            
            green_dominant = (green_channel > threshold) & \
                            (green_channel > red_channel) & \
                            (green_channel > blue_channel)
            
            blue_dominant = (blue_channel > threshold) & \
                           (blue_channel > (red_channel+green_channel)/2) 
            
            # Calculate ratios
            total_pixels = red_channel.size
            red_ratio = np.sum(red_dominant) / total_pixels
            green_ratio = np.sum(green_dominant) / total_pixels
            blue_ratio = np.sum(blue_dominant) / total_pixels
            
            # Update results
            results['has_red'] = red_ratio >= min_ratio
            results['has_green'] = green_ratio >= min_ratio
            results['has_blue'] = blue_ratio >= min_ratio
            
            results['color_ratios'] = {
                'red': red_ratio,
                'green': green_ratio,
                'blue': blue_ratio
            }
            
            # Determine dominant color
            max_ratio = max(red_ratio, green_ratio, blue_ratio)
            if max_ratio >= min_ratio:
                if max_ratio == red_ratio:
                    results['dominant_color'] = 'red'
                elif max_ratio == green_ratio:
                    results['dominant_color'] = 'green'
                else:
                    results['dominant_color'] = 'blue'
            
            return results
            
        except Exception as e:
            print(f"Error analyzing color distribution: {e}")
            return results

    def render_until_blue_detected(self, 
                                   solid_a: Optional[cq.Solid] = None,
                                   solid_b: Optional[cq.Solid] = None,
                                   ply_path: Optional[str] = None,
                                   colors: List[Tuple[float, float, float, float]] = None,
                                   show_edges: List[bool] = None,
                                   out_image_path: str = "render.png",
                                   max_attempts: int = 100,
                                   randomize_camera: bool = False,
                                   randomize_lighting: bool = False,
                                   verbose: bool = True) -> Tuple[str, bool, int]:
        """
        Render objects repeatedly until blue color is detected in the image.
        
        This function will keep rendering with different parameters (camera angles, lighting)
        until blue color is found in the resulting image, or until max_attempts is reached.
        
        Args:
            solid_a: First CadQuery solid to render
            solid_b: Second CadQuery solid to render  
            ply_path: Path to PLY mesh file to render
            colors: List of RGBA color tuples for objects
            show_edges: List of booleans for edge display
            out_image_path: Output image path
            max_attempts: Maximum number of rendering attempts
            blue_threshold: Minimum blue intensity to consider as blue (0-1)
            min_blue_ratio: Minimum ratio of blue pixels required
            randomize_camera: Whether to randomize camera parameters between attempts
            randomize_lighting: Whether to randomize lighting between attempts
            verbose: Whether to print progress information
            
        Returns:
            Tuple[str, bool, int]: (final_image_path, render_success, attempts_made)
        """
        import random
        import time
        
        # Store original parameters to restore later
        original_camera_distance = self.camera_distance_factor
        original_camera_fov = self.camera_fov_degrees
        original_camera_tilt = self.camera_tilt_down_degrees
        original_camera_azimuth = self.camera_azimuth_degrees
        original_camera_position = self.camera_position_factors
        original_ambient = self.ambient_intensity
        original_key_light = self.key_light_intensity
        original_fill_light = self.fill_light_intensity
        
        attempts = 0
        last_render_success = False
        
        try:
            while attempts < max_attempts:
                attempts += 1
                print(f"Rendering attempt {attempts}/{max_attempts}...")
                
                if verbose:
                    print(f"Rendering attempt {attempts}/{max_attempts}...")
                
                # Randomize parameters if requested
                if randomize_camera and attempts > 1:
                    # Randomize camera parameters
                    self.camera_distance_factor = random.uniform(1.2, 2.5)
                    self.camera_fov_degrees = random.uniform(45.0, 85.0)
                    self.camera_tilt_down_degrees = random.uniform(-70.0, -20.0)
                    self.camera_azimuth_degrees = random.uniform(0.0, 360.0)
                    self.camera_position_factors = (
                        random.uniform(0.8, 1.2),
                        random.uniform(0.8, 1.2), 
                        random.uniform(0.8, 1.2)
                    )
                
                if randomize_lighting and attempts > 1:
                    # Randomize lighting parameters
                    self.ambient_intensity = random.uniform(0.1, 0.5)
                    self.key_light_intensity = random.uniform(3.0, 8.0)
                    self.fill_light_intensity = random.uniform(1.0, 4.0)
                
                # Create unique filename for this attempt
                base_path, ext = os.path.splitext(out_image_path)
                current_image_path = f"{base_path}_attempt_{attempts}{ext}"
                
                try:
                    # Perform the rendering
                    image_path, render_success = self.render(
                        solid_a=solid_a,
                        solid_b=solid_b,
                        ply_path=ply_path,
                        colors=colors,
                        show_edges=show_edges,
                        out_image_path=current_image_path
                    )
                    
                    last_render_success = render_success
                    
                    # Check if the image contains blue
                    has_blue = self.has_blue_color(
                        image_path=current_image_path
                    )
                    
                    if verbose:
                        color_analysis = self.analyze_color_distribution(image_path)
                        print(f"  Render method: {'pyrender' if render_success else 'matplotlib'}")
                        print(f"  Blue detected: {has_blue}")
                        print(f"  Color ratios: R={color_analysis['color_ratios']['red']:.3f}, "
                              f"G={color_analysis['color_ratios']['green']:.3f}, "
                              f"B={color_analysis['color_ratios']['blue']:.3f}")
                    
                    if has_blue:
                        # Success! Copy to final output path if different
                        if current_image_path != out_image_path:
                            import shutil
                            shutil.copy2(current_image_path, out_image_path)
                            # Clean up temporary file
                            try:
                                os.remove(current_image_path)
                            except:
                                pass
                        
                        if verbose:
                            print(f"SUCCESS: Blue color detected after {attempts} attempts!")
                            print(f"Final image saved to: {out_image_path}")
                        
                        return out_image_path, render_success, attempts
                    
                    else:
                        if verbose:
                            print(f"  No significant blue detected, retrying...")
                        
                        # Keep the last attempt image for comparison if this is the final attempt
                        if attempts < max_attempts:
                            try:
                                os.remove(current_image_path)
                            except:
                                pass
                        else:
                            # This was the last attempt, keep the image
                            if current_image_path != out_image_path:
                                import shutil
                                shutil.copy2(current_image_path, out_image_path)
                    
                except Exception as render_error:
                    if verbose:
                        print(f"  Render attempt {attempts} failed: {render_error}")
                    continue
                
                # Small delay between attempts to avoid overwhelming the system
                if attempts < max_attempts:
                    time.sleep(0.1)
            
            # If we get here, we've exhausted all attempts without finding blue
            if verbose:
                print(f"FAILED: No blue color detected after {max_attempts} attempts.")
                print(f"Last rendered image saved to: {out_image_path}")
            
            return out_image_path, last_render_success, attempts
            
        finally:
            # Always restore original parameters
            self.camera_distance_factor = original_camera_distance
            self.camera_fov_degrees = original_camera_fov
            self.camera_tilt_down_degrees = original_camera_tilt
            self.camera_azimuth_degrees = original_camera_azimuth
            self.camera_position_factors = original_camera_position
            self.ambient_intensity = original_ambient
            self.key_light_intensity = original_key_light
            self.fill_light_intensity = original_fill_light

if __name__ == "__main__":
    # Example usage of the new class-based approach
    visualizer = CADVisualizer()
    
    # Render with default settings
    ply_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh.ply"
    output_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply1.png"
    
    colors = [(1.0, 0.0, 0.0, 1.0),      # Red opaque
              (0.0, 1.0, 0.0, 0.3),      # Green transparent
              (0.0, 0.0, 1.0, 0.6)]      # Blue transparent
    show_edges = [False, True, False]    # Only show edges for outerNBZ
    
    out = visualizer.render_until_blue_detected(
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        ply_path=ply_file,
        colors=colors,
        show_edges=show_edges,
        out_image_path=output_file
    )
        # Example usage of the new class-based approach
    visualizer = CADVisualizer()
    print(out)
    # Render with default settings
    ply_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh1.ply"
    output_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply2.png"
    
    colors = [(1.0, 0.0, 0.0, 1.0),      # Red opaque
              (0.0, 1.0, 0.0, 0.3),      # Green transparent
              (0.0, 0.0, 1.0, 0.5)]      # Blue transparent
    show_edges = [False, True, False]    # Only show edges for outerNBZ
    
    out = visualizer.render_until_blue_detected(
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        ply_path=ply_file,
        colors=colors,
        show_edges=show_edges,
        out_image_path=output_file
    )
        # Example usage of the new class-based approach
    visualizer = CADVisualizer()
    print(out)
    # Render with default settings
    ply_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh_g.ply"
    output_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply3.png"
    
    colors = [(1.0, 0.0, 0.0, 1.0),      # Red opaque
              (0.0, 1.0, 0.0, 0.3),      # Green transparent
              (0.0, 0.0, 1.0, 0.5)]      # Blue transparent
    show_edges = [False, True, False]    # Only show edges for outerNBZ
    
    out = visualizer.render_until_blue_detected(
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        ply_path=ply_file,
        colors=colors,
        show_edges=show_edges,
        out_image_path=output_file
    )
    print(out)