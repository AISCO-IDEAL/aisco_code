# render_cq_and_ply.py

import io
import os
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio
from typing import Tuple, Optional, List, Union

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
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        os.environ['DISPLAY'] = self.display_port
        try:
            import subprocess
            result = subprocess.run(['pgrep', 'Xvfb'], capture_output=True)
            if result.returncode != 0:
                subprocess.Popen(['Xvfb', self.display_port, '-screen', '0', self.xvfb_screen_size], 
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
        """Load a PLY file as a trimesh.Trimesh."""
        tm = trimesh.load(ply_path, file_type="ply")
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(tuple(g for g in tm.geometry.values()))
        return tm
    
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
               out_image_path: str = "render.png") -> str:
        """
        Render the specified objects with given colors and edge settings.
        
        Args:
            solid_a: First CadQuery solid (optional)
            solid_b: Second CadQuery solid (optional)
            ply_path: Path to PLY file (optional)
            colors: List of RGBA colors for each object
            show_edges: List of booleans indicating whether to show edges for each object
            out_image_path: Output image file path
            
        Returns:
            Path to the rendered image
        """
        # Default colors and edge settings
        if colors is None:
            colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.5), (0.0, 0.0, 1.0, 0.5)]
        if show_edges is None:
            show_edges = [False, True, False]  # Only show edges for middle object by default
        
        # Load and convert meshes
        meshes = []
        if solid_a is not None:
            meshes.append(self._cq_solid_to_trimesh(solid_a))
        if solid_b is not None:
            meshes.append(self._cq_solid_to_trimesh(solid_b))
        if ply_path is not None and os.path.exists(ply_path):
            meshes.append(self._load_ply_as_trimesh(ply_path))
        
        if not meshes:
            raise ValueError("No valid objects to render")
        
        # Check if meshes are valid
        for i, mesh in enumerate(meshes):
            if mesh.vertices.shape[0] == 0:
                raise ValueError(f"Mesh {i} is empty")
        
        # Apply scaling
        for mesh in meshes:
            if self.scale_factor != 1.0:
                mesh.vertices *= self.scale_factor
        
        # Center all meshes
        if len(meshes) > 1:
            combined_temp = trimesh.util.concatenate(meshes)
            original_center = combined_temp.bounding_box.centroid
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
        
        # Create scene
        scene = pyrender.Scene(bg_color=np.array(self.bg_color, dtype=np.uint8),
                              ambient_light=np.ones(3) * self.ambient_intensity)
        
        # Add meshes to scene
        for i, mesh in enumerate(meshes):
            if i < len(colors):
                color = colors[i]
                material = pyrender.MetallicRoughnessMaterial(
                    baseColorFactor=color, 
                    metallicFactor=self.metallic_factor, 
                    roughnessFactor=self.roughness_factor,
                    alphaMode='OPAQUE' if color[3] >= 1.0 else 'BLEND'
                )
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
                scene.add(pyrender_mesh)
                
                # Add edges if requested
                if i < len(show_edges) and show_edges[i]:
                    self._add_edges_to_scene(scene, mesh)
        
        # Setup camera and lighting
        self._setup_camera_and_lighting(scene, radius)
        
        # Render
        width, height = self.image_size
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
    # Example usage of the new class-based approach
    visualizer = CADVisualizer()
    
    # Render with default settings
    ply_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh.ply"
    output_file = "/cluster/home/clhuber/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply.png"
    
    colors = [(1.0, 0.0, 0.0, 1.0),      # Red opaque
              (0.0, 1.0, 0.0, 0.3),      # Green transparent
              (0.0, 0.0, 1.0, 0.5)]      # Blue transparent
    show_edges = [False, True, False]    # Only show edges for outerNBZ
    
    out = visualizer.render(
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        ply_path=ply_file,
        colors=colors,
        show_edges=show_edges,
        out_image_path=output_file
    )