import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import OpenGL
import pyrender
import trimesh
from pyrender import RenderFlags

# Constants for numerical thresholds
_MIN_EDGE_LENGTH = 1e-6
_MIN_VECTOR_NORM = 1e-12
_CAMERA_ALIGNMENT_THRESHOLD = 0.9
_OPACITY_THRESHOLD = 0.99


class NoPLYMeshesError(Exception):
    def __init__(self) -> None:
        super().__init__("Exactly two PLY files are required and both paths must exist.")


class CADVisualizer:
    def __init__(
        self,
        ply_path_a: str | None = None,
        ply_path_b: str | None = None,
        out_image_path: str = "render.png",
        max_attempts: int = 100,
    ) -> None:
        self.__image_size = (1200, 900)
        self.__scale_factor = 0.001

        self.__camera_distance_factor = 1
        self.__camera_fov_degrees = 65.0
        self.__camera_tilt_down_degrees = -50.0
        self.__camera_azimuth_degrees = 0.0
        self.__camera_position_factors = (1, 1, 1)

        self.__edge_radius = 0.002

        self.__blue_threshold = 0.1
        self.__min_blue_ratio = 0.001

        self.__ply_a = ply_path_a
        self.__ply_b = ply_path_b

        # Use colors: mesh A (full/original) in opaque blue, mesh B (solar deck) in opaque white
        self.__colors = [
            (0.0, 0.0, 1.0, 1.0),  # mesh A: opaque blue
            (1.0, 1.0, 1.0, 1.0),  # mesh B: opaque white (solar deck)
        ]
        self.__show_edges = [False, False]
        self.__out_image_path = out_image_path
        self.__max_attempts = max_attempts

        # Track Xvfb process for cleanup (Linux only)
        self.__xvfb_process: subprocess.Popen | None = None

    def __setup_rendering_environment(self) -> None:
        """Setup rendering environment - called fresh for each render"""
        # Cleanup previous Xvfb if exists
        if self.__xvfb_process:
            try:
                self.__xvfb_process.terminate()
                self.__xvfb_process.wait(timeout=1)
            except:  # noqa: E722
                pass
            finally:
                self.__xvfb_process = None

        if sys.platform.startswith("linux"):
            # Kill any existing Xvfb processes
            subprocess.run(["pkill", "-f", "Xvfb"], check=False, capture_output=True)
            time.sleep(0.1)

            # Cluster-optimized settings for EGL; safe on many headless Linux hosts
            os.environ.update(
                {
                    "PYOPENGL_PLATFORM": "egl",
                    "DISPLAY": ":99",
                    "MESA_GL_VERSION_OVERRIDE": "4.6",
                    "MESA_GLSL_VERSION_OVERRIDE": "460",
                    "EGL_PLATFORM": "surfaceless",
                    "__GL_SYNC_TO_VBLANK": "0",
                    "__GL_YIELD": "NOTHING",
                    "PYTHONHASHSEED": "0",
                }
            )

            # Start fresh Xvfb for headless rendering
            try:
                self.__xvfb_process = subprocess.Popen(
                    [
                        "Xvfb",
                        ":99",
                        "-screen",
                        "0",
                        "1920x1080x24",
                        "-ac",
                        "+extension",
                        "GLX",
                        "+render",
                        "-noreset",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Give Xvfb time to start
                time.sleep(0.5)
            except FileNotFoundError:
                # Xvfb not available; continue without it
                self.__xvfb_process = None
        else:
            # On non-Linux platforms, avoid forcing EGL/Display settings
            for var in ("DISPLAY",):
                os.environ.pop(var, None)

        OpenGL.ERROR_CHECKING = OpenGL.ERROR_LOGGING = OpenGL.FULL_LOGGING = False

    def __del__(self) -> None:
        """Cleanup Xvfb process on object destruction"""
        try:
            if self.__xvfb_process:
                try:
                    self.__xvfb_process.terminate()
                    self.__xvfb_process.wait(timeout=2)
                except:  # noqa: E722
                    pass
                finally:
                    self.__xvfb_process = None

            # Final cleanup of any lingering Xvfb processes (Linux only)
            if sys.platform.startswith("linux"):
                try:  # noqa: SIM105
                    subprocess.run(["pkill", "-f", "Xvfb"], check=False, capture_output=True, timeout=2)
                except:  # noqa: E722
                    pass
        except:  # noqa: E722
            pass

    def __load_mesh(self, path, file_type) -> trimesh.Trimesh | None:
        loaded = trimesh.load(path, file_type=file_type, process=False)
        if isinstance(loaded, trimesh.Scene):
            geoms = tuple(loaded.geometry.values())
            tm: trimesh.Trimesh = trimesh.util.concatenate(geoms)
        elif isinstance(loaded, trimesh.Trimesh):
            tm = loaded
        else:
            return None

        # Minimal processing for speed
        tm.remove_infinite_values()
        tm.remove_unreferenced_vertices()
        if len(tm.faces) > 0:
            trimesh.repair.fix_normals(tm, multibody=False)

        return tm

    def __add_edges_to_scene(self, scene, mesh) -> None:
        # Batch edge creation for better performance
        edges_unique = mesh.edges_unique
        vertices = mesh.vertices

        # Pre-allocate arrays
        edge_vectors = vertices[edges_unique[:, 1]] - vertices[edges_unique[:, 0]]
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)
        valid_edges = edge_lengths > _MIN_EDGE_LENGTH

        for idx in np.where(valid_edges)[0]:
            a, b = vertices[edges_unique[idx]]
            v = edge_vectors[idx]
            length = edge_lengths[idx]

            cyl = trimesh.creation.cylinder(radius=self.__edge_radius, height=length, sections=6)
            z, u = np.array([0, 0, 1]), v / length

            if np.allclose(u, -z):
                r = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            elif not np.allclose(u, z):
                axis = np.cross(z, u)
                axis /= np.linalg.norm(axis)
                ang = np.arccos(np.clip(np.dot(z, u), -1, 1))
                r = trimesh.transformations.rotation_matrix(ang, axis)
            else:
                r = np.eye(4)

            r[:3, 3] = (a + b) / 2
            cyl.apply_transform(r)
            scene.add(
                pyrender.Mesh.from_trimesh(
                    cyl,
                    material=pyrender.MetallicRoughnessMaterial(
                        baseColorFactor=(0.0, 0.0, 0.0, 1.0),
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                )
            )

    def __setup_camera_and_lighting(self, scene, radius) -> None:
        # Ensure radius is valid and set minimum distance
        radius = max(radius, 0.1)  # Prevent zero or very small radius
        d = radius * self.__camera_distance_factor

        # Ensure znear and zfar are valid
        znear = max(0.001, d * 0.01)
        zfar = max(znear * 2, d * 20.0)  # Ensure zfar > znear

        cam = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(self.__camera_fov_degrees),
            znear=znear,
            zfar=zfar,
        )

        # Apply azimuth and tilt transformations
        ex, ey, ez = (d * f for f in self.__camera_position_factors)
        az = np.deg2rad(self.__camera_azimuth_degrees)
        c, s = np.cos(az), np.sin(az)
        exr, eyr = ex * c - ey * s, ex * s + ey * c
        tilt = np.deg2rad(-self.__camera_tilt_down_degrees)
        ct, st = np.cos(tilt), np.sin(tilt)
        eye = np.array([exr * ct - ez * st, eyr, exr * st + ez * ct])

        # Look-at matrix
        f = -eye / np.linalg.norm(eye)
        up = np.array([0, 0, 1.0])
        s = np.cross(f, up)
        sn = np.linalg.norm(s)
        if sn < _MIN_VECTOR_NORM:
            up = np.array([1, 0, 0]) if abs(f[0]) < _CAMERA_ALIGNMENT_THRESHOLD else np.array([0, 1, 0])
            s = np.cross(f, up)
            sn = np.linalg.norm(s)
        s /= sn
        u = np.cross(s, f)
        pose = np.eye(4)
        pose[0, :3], pose[1, :3], pose[2, :3] = s, u, -f
        pose[:3, 3] = eye

        scene.add(cam, pose=pose)
        scene.add(pyrender.DirectionalLight(intensity=5.0), pose=pose)

        # Add fill lights
        for dvec in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
            p = np.eye(4)
            p[:3, 3] = np.array(dvec) * d * 0.5
            scene.add(pyrender.DirectionalLight(intensity=2.0), pose=p)

    def __render(self, out_image_path: str) -> tuple[str, bool]:
        # Setup fresh rendering environment for each render attempt
        self.__setup_rendering_environment()

        # Require exactly two valid PLY paths
        if not (self.__ply_a and self.__ply_b and Path(self.__ply_a).exists() and Path(self.__ply_b).exists()):
            raise NoPLYMeshesError()

        meshes: list[trimesh.Trimesh] = []

        loaded_mesh_a = self.__load_mesh(self.__ply_a, "ply")
        if loaded_mesh_a:
            meshes.append(loaded_mesh_a)
        loaded_mesh_b = self.__load_mesh(self.__ply_b, "ply")
        if loaded_mesh_b:
            meshes.append(loaded_mesh_b)

        # Vectorized scaling and centering
        for m in meshes:
            m.vertices *= self.__scale_factor
        all_vertices = np.vstack([m.vertices for m in meshes])
        center = np.mean(all_vertices, axis=0)
        for m in meshes:
            m.vertices -= center

        radius = np.linalg.norm(trimesh.util.concatenate(meshes).extents) / 2.0

        # Sort meshes: opaque first, then transparent by distance
        data = list(zip(meshes, self.__colors[: len(meshes)], self.__show_edges[: len(meshes)], strict=False))
        opaque = [(m, c, e) for m, c, e in data if c[3] >= _OPACITY_THRESHOLD]
        transp = [(m, c, e) for m, c, e in data if c[3] < _OPACITY_THRESHOLD]
        cam_pos = np.array([0, 0, radius * self.__camera_distance_factor])
        transp.sort(key=lambda x: float(-np.linalg.norm(np.mean(x[0].vertices, axis=0) - cam_pos)))

        # Build scene
        scene = pyrender.Scene(
            bg_color=np.array((255, 255, 255, 0), dtype=np.uint8),
            ambient_light=np.ones(3) * 0.3,
        )

        for m, col, edge in opaque + transp:
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=col,
                metallicFactor=0.0,
                roughnessFactor=0.5,
                alphaMode="OPAQUE" if col[3] >= _OPACITY_THRESHOLD else "BLEND",
                doubleSided=True,
                alphaCutoff=0.01,
            )
            scene.add(pyrender.Mesh.from_trimesh(m, material=mat))
            if edge:
                self.__add_edges_to_scene(scene, m)

        self.__setup_camera_and_lighting(scene, radius)

        # Create fresh renderer for each render to avoid EGL context issues
        w, h = self.__image_size
        renderer = pyrender.OffscreenRenderer(w, h)
        color, _ = renderer.render(scene, flags=RenderFlags.RGBA)
        renderer.delete()

        Path(out_image_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_image_path, color)
        return out_image_path, True

    def __has_blue_color(self, image_path) -> bool:
        img = imageio.imread(image_path).astype(np.float32) / 255.0
        # Vectorized operations for faster blue detection
        b, r, g = img[:, :, 2], img[:, :, 0], img[:, :, 1]
        blue_dom = (b > self.__blue_threshold) & (b > (r + g) * 0.5)
        return bool(np.mean(blue_dom) >= self.__min_blue_ratio)

    def render_until_blue_detected(self) -> tuple[str, bool, int]:
        for attempts in range(1, self.__max_attempts + 1):
            p = Path(self.__out_image_path)
            cur = f"{p.parent / p.stem}_attempt_{attempts}{p.suffix}"
            _, ok = self.__render(out_image_path=cur)

            if self.__has_blue_color(cur):
                shutil.copy2(cur, self.__out_image_path)
                Path(cur).unlink()
                return self.__out_image_path, ok, attempts
            Path(cur).unlink()

        return self.__out_image_path, ok, self.__max_attempts
