import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cadquery as cq
import imageio.v2 as imageio
import numpy as np
import OpenGL
import pyrender
import trimesh
from pyrender import RenderFlags

from subprojects.wp2_assembler.archive.version1.components.nbz import innerNBZ, outerNBZ

# Constants for numerical thresholds
_MIN_EDGE_LENGTH = 1e-6
_MIN_VECTOR_NORM = 1e-12
_CAMERA_ALIGNMENT_THRESHOLD = 0.9
_OPACITY_THRESHOLD = 0.99


class CADVisualizer:
    def __init__(
        self,
        ply_path=None,
        stl_path=None,
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        out_image_path="render.png",
        max_attempts=100,
    ) -> None:
        self.__image_size = (1200, 900)
        self.__bg_color = (255, 255, 255, 0)
        self.__scale_factor = 0.001
        self.__additional_scale = 1
        self.__mesh_tolerance = 0.1
        self.__angular_tolerance = 1

        self.__camera_distance_factor = 1
        self.__camera_fov_degrees = 65.0
        self.__camera_tilt_down_degrees = -50.0
        self.__camera_azimuth_degrees = 0.0
        self.__camera_position_factors = (1, 1, 1)

        self.__ambient_intensity = 0.3
        self.__key_light_intensity = 5.0
        self.__fill_light_intensity = 2.0

        self.__metallic_factor = 0.0
        self.__roughness_factor = 0.5

        self.__edge_color = (0.0, 0.0, 0.0, 1.0)
        self.__edge_radius = 0.002

        self.__display_port = ":99"
        self.__xvfb_screen_size = "1024x768x24"

        self.__blue_threshold = 0.1
        self.__min_blue_ratio = 0.001

        self.__solid_a = solid_a
        self.__solid_b = solid_b
        self.__ply_path = ply_path
        self.__stl_path = stl_path
        self.__colors = [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 0.3),
            (0.0, 0.0, 1.0, 0.6),
        ]
        self.__show_edges = [False, True, False]
        self.__out_image_path = out_image_path
        self.__max_attempts = max_attempts

        self.__setup_headless_rendering()

    def __setup_headless_rendering(self) -> None:
        env = {
            "PYOPENGL_PLATFORM": "osmesa",
            "DISPLAY": self.__display_port,
            "MESA_GL_VERSION_OVERRIDE": "3.3",
            "MESA_GLSL_VERSION_OVERRIDE": "330",
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "GALLIUM_DRIVER": "llvmpipe",
            "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
            "EGL_PLATFORM": "surfaceless",
            "MESA_GLAPI_NO_REMAP": "1",
            "PYTHONHASHSEED": "0",
        }
        os.environ.update(env)

        subprocess.run(["pkill", "-f", "Xvfb"], check=False, capture_output=True)
        subprocess.Popen(
            [
                "Xvfb",
                self.__display_port,
                "-screen",
                "0",
                self.__xvfb_screen_size,
                "-ac",
                "+extension",
                "GLX",
                "+render",
                "-noreset",
                "-nolisten",
                "tcp",
                "-dpi",
                "96",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        OpenGL.ERROR_CHECKING = False
        OpenGL.ERROR_LOGGING = False
        OpenGL.FULL_LOGGING = False

        os.environ["PYOPENGL_PLATFORM"] = "egl"
        r = pyrender.OffscreenRenderer(1, 1)
        r.delete()

    def __cq_solid_to_trimesh(self, solid) -> trimesh.Trimesh:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            path = tmp.name
        cq.exporters.export(
            cq.Workplane().add(solid),
            path,
            tolerance=self.__mesh_tolerance,
            angularTolerance=self.__angular_tolerance,
        )
        mesh = self.__load_mesh(path, "stl")
        assert mesh is not None, "Failed to load mesh from CQ solid"
        return mesh

    def __load_mesh(self, path, file_type) -> trimesh.Trimesh | None:
        for _ in range(3):
            loaded = trimesh.load(path, file_type=file_type)
            if isinstance(loaded, trimesh.Scene):
                geoms = tuple(loaded.geometry.values())
                tm: trimesh.Trimesh = trimesh.util.concatenate(geoms)
            else:
                tm = loaded

            tm.remove_infinite_values()
            tm.remove_unreferenced_vertices()
            if len(tm.faces) > 0:
                trimesh.repair.fix_inversion(tm)
                trimesh.repair.fix_normals(tm)
                tm.process(validate=True)
                if not hasattr(tm, "face_normals") or len(tm.face_normals) == 0:
                    tm.fix_normals()
            return tm
        return None

    def __add_edges_to_scene(self, scene, mesh) -> None:
        for e in mesh.vertices[mesh.edges_unique]:
            a, b = e
            v = b - a
            length = np.linalg.norm(v)
            if length <= _MIN_EDGE_LENGTH:
                continue

            cyl = trimesh.creation.cylinder(radius=self.__edge_radius, height=length, sections=8)
            z, u = np.array([0, 0, 1]), v / length

            if np.allclose(u, -z):
                r = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
            elif not (np.allclose(u, z)):
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
                        baseColorFactor=self.__edge_color,
                        metallicFactor=0.0,
                        roughnessFactor=1.0,
                    ),
                )
            )

    def __setup_camera_and_lighting(self, scene, radius) -> None:
        d = radius * self.__camera_distance_factor
        cam = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(self.__camera_fov_degrees),
            znear=max(0.001, d * 0.01),
            zfar=d * 20.0,
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
        scene.add(pyrender.DirectionalLight(intensity=self.__key_light_intensity), pose=pose)

        # Add fill lights
        for dvec in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
            p = np.eye(4)
            p[:3, 3] = np.array(dvec) * d * 0.5
            scene.add(pyrender.DirectionalLight(intensity=self.__fill_light_intensity), pose=p)

    def __render(self, out_image_path: str) -> tuple[str, bool]:
        meshes = [
            self.__cq_solid_to_trimesh(self.__solid_a),
            self.__cq_solid_to_trimesh(self.__solid_b),
        ]
        types = ["solid_a", "solid_b"]

        if self.__ply_path and Path(self.__ply_path).exists():
            loaded_mesh = self.__load_mesh(self.__ply_path, "ply")
            if loaded_mesh:
                meshes.append(loaded_mesh)
                types.append("ply")
        if self.__stl_path and Path(self.__stl_path).exists():
            loaded_mesh = self.__load_mesh(self.__stl_path, "stl")
            if loaded_mesh:
                meshes.append(loaded_mesh)
                types.append("stl")

        # Scale and center meshes
        for m in meshes:
            m.vertices *= self.__scale_factor
        center = np.mean(np.vstack([m.vertices for m in meshes]), axis=0)
        for m in meshes:
            m.vertices = (m.vertices - center) * self.__additional_scale

        radius = np.linalg.norm(trimesh.util.concatenate(meshes).extents) / 2.0  # type: ignore[arg-type]

        # Sort meshes: opaque first, then transparent by distance
        data = list(zip(meshes, self.__colors, self.__show_edges, types, strict=False))
        opaque = [(m, c, e, t) for m, c, e, t in data if c[3] >= _OPACITY_THRESHOLD]
        transp = [(m, c, e, t) for m, c, e, t in data if c[3] < _OPACITY_THRESHOLD]
        cam_pos = np.array([0, 0, radius * self.__camera_distance_factor])
        transp.sort(key=lambda x: float(-np.linalg.norm(np.mean(x[0].vertices, axis=0) - cam_pos)))

        # Build scene
        scene = pyrender.Scene(
            bg_color=np.array(self.__bg_color, dtype=np.uint8),
            ambient_light=np.ones(3) * self.__ambient_intensity,
        )

        for m, col, edge, _ in opaque + transp:
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=col,
                metallicFactor=self.__metallic_factor,
                roughnessFactor=self.__roughness_factor,
                alphaMode="OPAQUE" if col[3] >= _OPACITY_THRESHOLD else "BLEND",
                doubleSided=True,
                alphaCutoff=0.01,
            )
            scene.add(pyrender.Mesh.from_trimesh(m, material=mat))
            if edge:
                self.__add_edges_to_scene(scene, m)

        self.__setup_camera_and_lighting(scene, radius)

        # Render
        w, h = self.__image_size
        renderer = pyrender.OffscreenRenderer(w, h)
        color, _ = renderer.render(scene, flags=RenderFlags.RGBA)
        renderer.delete()

        Path(out_image_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(out_image_path, color)
        return out_image_path, True

    def __has_blue_color(self, image_path) -> bool:
        img = imageio.imread(image_path).astype(np.float32) / 255.0
        b, r, g = img[:, :, 2], img[:, :, 0], img[:, :, 1]
        blue_dom = (b > self.__blue_threshold) & (b > (r + g) / 2)
        return blue_dom.sum() / b.size >= self.__min_blue_ratio

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


if __name__ == "__main__":
    username = "clhuber"
    base_path = f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data"

    for mesh_name, render_name in [
        ("mesh1.ply", "render_ply1.png"),
        ("mesh.ply", "render_ply.png"),
        ("mesh_g.ply", "render_ply2.png"),
    ]:
        visualizer = CADVisualizer(
            ply_path=f"{base_path}/Meshes/{mesh_name}",
            out_image_path=f"{base_path}/Renders/{render_name}",
        )
        visualizer.render_until_blue_detected()
