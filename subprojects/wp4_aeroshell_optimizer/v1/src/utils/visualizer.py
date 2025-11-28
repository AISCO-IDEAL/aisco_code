
import os
import shutil
import subprocess
import tempfile
import time

import cadquery as cq
import imageio.v2 as imageio
import numpy as np
import OpenGL
import pyrender
import trimesh
from cadquery import exporters
from pyrender import RenderFlags

from subprojects.wp2_assembler.archive.version1.components.nbz import innerNBZ, outerNBZ

username = "clhuber"


class CADVisualizer:
    def __init__(
        self,
        ply_path=None,
        stl_path=None,
        solid_a=innerNBZ(),
        solid_b=outerNBZ(),
        out_image_path="render.png",
        max_attempts=100,
    ):
        self.__image_size = (1200, 900)
        self.__bg_color = (255, 255, 255, 0)
        self.__scale_factor = 0.001
        self.__additional_scale = 2.0
        self.__mesh_tolerance = 0.1
        self.__angular_tolerance = 1

        self.__camera_distance_factor = 1.5
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
        self.__show_edges = None
        self.__out_image_path = out_image_path
        self.__max_attempts = max_attempts

        self.__setup_headless_rendering()

    def __setup_headless_rendering(self):
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
        for k, v in env.items():
            os.environ[k] = v

        np.random.seed(42)

        subprocess.run(["pkill", "-f", "Xvfb"], check=False, capture_output=True)
        time.sleep(1)
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

    def __set_camera_params(
        self,
        distance_factor=None,
        fov_degrees=None,
        tilt_down_degrees=None,
        azimuth_degrees=None,
        position_factors=None,
    ):
        for name, val in [
            ("__camera_distance_factor", distance_factor),
            ("__camera_fov_degrees", fov_degrees),
            ("__camera_tilt_down_degrees", tilt_down_degrees),
            ("__camera_azimuth_degrees", azimuth_degrees),
            ("__camera_position_factors", position_factors),
        ]:
            if val is not None:
                setattr(self, name, val)

    def __set_lighting(
        self,
        ambient_intensity=None,
        key_light_intensity=None,
        fill_light_intensity=None,
    ):
        for name, val in [
            ("__ambient_intensity", ambient_intensity),
            ("__key_light_intensity", key_light_intensity),
            ("__fill_light_intensity", fill_light_intensity),
        ]:
            if val is not None:
                setattr(self, name, val)

    def __cq_solid_to_trimesh(self, solid):
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            path = tmp.name
        cq.exporters.export(
            cq.Workplane().add(solid),
            path,
            tolerance=self.__mesh_tolerance,
            angularTolerance=self.__angular_tolerance,
        )

        tm = trimesh.load(path, file_type="stl")
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
        return tm

    def __load_mesh(self, path, file_type):
        max_retries = 3
        for attempt in range(max_retries):
            tm = trimesh.load(path, file_type=file_type)
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
            tm.remove_infinite_values()
            tm.remove_unreferenced_vertices()
            if len(tm.faces) > 0:
                trimesh.repair.fix_inversion(tm)
                trimesh.repair.fix_normals(tm)
                tm.process(validate=True)
                if (
                    not getattr(tm, "face_normals", None) is not None
                    or len(tm.face_normals) == 0
                ):
                    tm.fix_normals()
            return tm

    def __add_edges_to_scene(self, scene, mesh):
        edges = mesh.edges_unique
        vs = mesh.vertices[edges]
        for e in vs:
            a, b = e
            v = b - a
            l = np.linalg.norm(v)  # noqa: E741
            if l <= 1e-6:  # noqa: PLR2004
                continue
            cyl = trimesh.creation.cylinder(
                radius=self.__edge_radius, height=l, sections=8
            )
            z = np.array([0, 0, 1])
            u = v / l
            if not (np.allclose(u, z) or np.allclose(u, -z)):
                axis = np.cross(z, u)
                axis /= np.linalg.norm(axis)
                ang = np.arccos(np.clip(np.dot(z, u), -1, 1))
                r = trimesh.transformations.rotation_matrix(ang, axis)
            elif np.allclose(u, -z):
                r = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
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

    def __setup_camera_and_lighting(self, scene, radius):
        d = radius * self.__camera_distance_factor
        cam = pyrender.PerspectiveCamera(
            yfov=np.deg2rad(self.__camera_fov_degrees),
            znear=max(0.001, d * 0.01),
            zfar=d * 20.0,
        )
        ex, ey, ez = (
            d * self.__camera_position_factors[0],
            d * self.__camera_position_factors[1],
            d * self.__camera_position_factors[2],
        )

        az = np.deg2rad(self.__camera_azimuth_degrees)
        c, s = np.cos(az), np.sin(az)
        exr, eyr = ex * c - ey * s, ex * s + ey * c

        tilt = np.deg2rad(-self.__camera_tilt_down_degrees)
        ct, st = np.cos(tilt), np.sin(tilt)
        exf = exr * ct - ez * st
        ezf = exr * st + ez * ct

        eye = np.array([exf, eyr, ezf])
        tgt = np.array([0.0, 0.0, 0.0])

        def look_at(eye, target, up=np.array([0, 0, 1.0])):
            f = target - eye
            fn = np.linalg.norm(f)
            f /= fn
            u = up / np.linalg.norm(up)
            s = np.cross(f, u)
            sn = np.linalg.norm(s)
            if sn < 1e-12:  # noqa: PLR2004
                up = np.array([1, 0, 0]) if abs(f[0]) < 0.9 else np.array([0, 1, 0])  # noqa: PLR2004
                u = up / np.linalg.norm(up)
                s = np.cross(f, u)
                sn = np.linalg.norm(s)
            s /= sn
            u = np.cross(s, f)
            m = np.eye(4)
            m[0, :3], m[1, :3], m[2, :3] = s, u, -f
            m[:3, 3] = eye
            return m

        pose = look_at(eye, tgt)
        scene.add(cam, pose=pose)

        key = pyrender.DirectionalLight(intensity=self.__key_light_intensity)
        scene.add(key, pose=pose)

        center = np.array([0.0, 0.0, 0.0])
        for dvec in [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]:
            p = np.eye(4)
            p[:3, 3] = center + np.array(dvec) * d * 0.5
            scene.add(
                pyrender.DirectionalLight(intensity=self.__fill_light_intensity),
                pose=p,
            )

    def __render(self, out_image_path: str):
        self.__show_edges = [False, True, False]

        meshes, types = (
            [
                self.__cq_solid_to_trimesh(self.__solid_a),
                self.__cq_solid_to_trimesh(self.__solid_b),
            ],
            ["solid_a", "solid_b"],
        )

        if self.__ply_path and os.path.exists(self.__ply_path):
            meshes.append(self.__load_mesh(self.__ply_path, "ply"))
            types.append("ply")
        if self.__stl_path and os.path.exists(self.__stl_path):
            meshes.append(self.__load_mesh(self.__stl_path, "stl"))
            types.append("stl")

        for m in meshes:
            m.vertices *= self.__scale_factor

        allv = np.vstack([m.vertices for m in meshes])
        center = np.mean(allv, axis=0)
        for m in meshes:
            m.vertices -= center
            m.vertices *= self.__additional_scale

        combo = trimesh.util.concatenate(meshes)
        ext = combo.extents
        radius = np.linalg.norm(ext) / 2.0

        data = list(zip(meshes, self.__colors, self.__show_edges, types, strict=False))

        opaque = [(m, c, e, t) for m, c, e, t in data if c[3] >= 0.99]  # noqa: PLR2004
        transp = [(m, c, e, t) for m, c, e, t in data if c[3] < 0.99]  # noqa: PLR2004

        cam_pos = np.array([0, 0, radius * self.__camera_distance_factor])

        def dist(x):
            m, _, _, _ = x
            return np.linalg.norm(np.mean(m.vertices, axis=0) - cam_pos)

        transp.sort(key=dist, reverse=True)
        ordered = opaque + transp

        w, h = self.__image_size
        pyrender_success = False
        color = None

        os.environ["PYOPENGL_PLATFORM"] = "egl"

        scene = pyrender.Scene(
            bg_color=np.array(self.__bg_color, dtype=np.uint8),
            ambient_light=np.ones(3) * self.__ambient_intensity,
        )

        for m, col, edge, _ in ordered:
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=col,
                metallicFactor=self.__metallic_factor,
                roughnessFactor=self.__roughness_factor,
                alphaMode="OPAQUE" if col[3] >= 0.99 else "BLEND",  # noqa: PLR2004
                doubleSided=True,
                alphaCutoff=0.01,
            )
            scene.add(pyrender.Mesh.from_trimesh(m, material=mat))
            if edge:
                self.__add_edges_to_scene(scene, m)

        self.__setup_camera_and_lighting(scene, radius)

        renderer = pyrender.OffscreenRenderer(w, h)
        color, _ = renderer.render(scene, flags=RenderFlags.RGBA)
        renderer.delete()
        pyrender_success = True

        if pyrender_success and color is not None:
            os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
            imageio.imwrite(out_image_path, color)
            return out_image_path, True
        return False, False

    def __has_blue_color(self, image_path):
        img = imageio.imread(image_path)
        img = img.astype(np.float32) / 255.0
        b, r, g = img[:, :, 2], img[:, :, 0], img[:, :, 1]
        blue_dom = (b > self.__blue_threshold) & (b > (r + g) / 2)
        ratio = blue_dom.sum() / b.size
        return ratio >= self.__min_blue_ratio

    def render_until_blue_detected(self):
        attempts = 0
        last_ok = False
        while attempts < self.__max_attempts:
            attempts += 1
            base, ext = os.path.splitext(self.__out_image_path)  # noqa: PTH122
            cur = f"{base}_attempt_{attempts}{ext}"
            _, ok = self.__render(out_image_path=cur)
            last_ok = ok
            has_blue = self.__has_blue_color(cur)
            if has_blue:
                shutil.copy2(cur, self.__out_image_path)
                os.remove(cur)  # noqa: PTH107
                return self.__out_image_path, ok, attempts
            else:
                os.remove(cur)  # noqa: PTH107
        return self.__out_image_path, last_ok, attempts


if __name__ == "__main__":
    visualizer = CADVisualizer(
        ply_path=f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh1.ply",
        out_image_path=f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply1.png",
    )
    out = visualizer.render_until_blue_detected()

    visualizer = CADVisualizer(
        ply_path=f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh.ply",
        out_image_path=f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply.png",
    )
    out = visualizer.render_until_blue_detected()

    visualizer = CADVisualizer(
        ply_path=f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Meshes/mesh_g.ply",
        out_image_path=f"/cluster/home/{username}/AISCO_code/subprojects/wp4_aeroshell_optimizer/data/Renders/render_ply2.png",
    )
    out = visualizer.render_until_blue_detected()
