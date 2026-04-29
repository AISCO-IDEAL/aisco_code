import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import cadquery as cq
import imageio.v2 as imageio
import numpy as np
import OpenGL
import pyrender
import torch
import trimesh
from pyrender import RenderFlags

import wandb
from subprojects.wp2_assembler.version4.components.nbz import call_assembly
from subprojects.wp2_assembler.version4.utils.reg_imp import get_outer_prism_bounds_mm
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.config_utils import load_config
from subprojects.wp5.unified_mesh_extraction import MeshSDFExtractor, load_mesh_extraction_config

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
        out_image_path="render.png",
        max_attempts=100,
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

        # Track Xvfb process for cleanup
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

        # Kill any existing Xvfb processes
        subprocess.run(["pkill", "-f", "Xvfb"], check=False, capture_output=True)
        time.sleep(0.1)

        # Cluster-optimized settings for RTX 4090 with Xvfb
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
        self.__xvfb_process = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1920x1080x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Give Xvfb time to start
        time.sleep(0.5)

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

            # Final cleanup of any lingering Xvfb processes
            try:  # noqa: SIM105
                subprocess.run(["pkill", "-f", "Xvfb"], check=False, capture_output=True, timeout=2)
            except:  # noqa: E722
                pass
        except:  # noqa: E722
            pass

    def __cq_solid_to_trimesh(self, solid_or_list) -> trimesh.Trimesh:
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            path = tmp.name
        wp = cq.Workplane()
        if isinstance(solid_or_list, (list | tuple)):
            for s in solid_or_list:
                if s is None:
                    continue
                wp = wp.add(s)
        else:
            wp = wp.add(solid_or_list)
        cq.exporters.export(
            wp,
            path,
            tolerance=0.1,
            angularTolerance=1,
        )
        mesh = self.__load_mesh(path, "stl")
        Path(path).unlink()  # Clean up temp file immediately
        assert mesh is not None, "Failed to load mesh from CQ solid(s)"

        return mesh

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
        self.__general_config = load_config()

        meshes = []
        types = ["solid_a", "solid_b"]

        inner_box, outer_box = self.inner_bb, self.outer_bb
        meshes = [
            self.__cq_solid_to_trimesh(inner_box),
            self.__cq_solid_to_trimesh(outer_box),
        ]

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

        # Vectorized scaling and centering
        for m in meshes:
            m.vertices *= self.__scale_factor
        all_vertices = np.vstack([m.vertices for m in meshes])
        center = np.mean(all_vertices, axis=0)
        for m in meshes:
            m.vertices -= center

        radius = np.linalg.norm(trimesh.util.concatenate(meshes).extents) / 2.0

        # Sort meshes: opaque first, then transparent by distance
        data = list(zip(meshes, self.__colors, self.__show_edges, types, strict=False))
        opaque = [(m, c, e, t) for m, c, e, t in data if c[3] >= _OPACITY_THRESHOLD]
        transp = [(m, c, e, t) for m, c, e, t in data if c[3] < _OPACITY_THRESHOLD]
        cam_pos = np.array([0, 0, radius * self.__camera_distance_factor])
        transp.sort(key=lambda x: float(-np.linalg.norm(np.mean(x[0].vertices, axis=0) - cam_pos)))

        # Build scene
        scene = pyrender.Scene(
            bg_color=np.array((255, 255, 255, 0), dtype=np.uint8),
            ambient_light=np.ones(3) * 0.3,
        )

        for m, col, edge, _ in opaque + transp:
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

    def render_until_blue_detected(self, outer_bb: cq.Solid | None = None, inner_bb: list | None = None) -> tuple[str, bool, int]:
        if outer_bb is None:
            config = load_config(
                v=self.__general_config["general_info"]["version"],
                cfg_name=self.__general_config["active_configs"]["losses"],
            )
            bb_config = config.get("bb_config")

            if bb_config:
                assembley_version = bb_config.get("assembly_version")
                assembly_id = bb_config["assembly_id"]

            if assembley_version == 3:
                self.inner_bb, self.outer_b, _ = call_assembly(
                    assembly_id
                )  # getting assembly for inner_nbz and outer_nbz as solid

        else:
            self.outer_bb = outer_bb
            self.inner_bb = inner_bb

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


class AutoVisToWandB:
    def __init__(self, inner_bb, outer_bb, sdf_model, epoch_name, device):
        try:
            self.device = device
            ply_file = self.extract_sdf_to_temp_ply(
                sdf_model=sdf_model,
                resolution=128,
                bbox=[-5e3, 5e3, -5e3, 5e3, -5e3, 5e3],
                isovalue=0.0,
                device=device,
            )
            self.device = device
            tmp_dir = tempfile.mkdtemp(prefix="nice")
            out_image_path = os.path.join(tmp_dir, "nice.png")
            visualiser = CADVisualizer(ply_path=ply_file, out_image_path=out_image_path)
            path, _, _ = visualiser.render_until_blue_detected(outer_bb=outer_bb, inner_bb=inner_bb)
            wandb.log(
                {
                    "Bounding Boxes:": wandb.Image(
                        path,
                    )
                },
                step=epoch_name,
            )
        except Exception as _:
            return

    def write_ply(self, vertices: torch.Tensor, faces: torch.Tensor, out_path: str) -> None:
        """
        Minimal ASCII PLY writer.

        Args:
            vertices: (V, 3) float tensor
            faces: (F, 3) long tensor
            out_path: output .ply file path
        """
        v = vertices.detach().cpu().numpy()
        f = faces.detach().cpu().numpy()

        num_vertices = v.shape[0]
        num_faces = f.shape[0]

        with open(out_path, "w") as f_out:
            # Header
            f_out.write("ply\n")
            f_out.write("format ascii 1.0\n")
            f_out.write(f"element vertex {num_vertices}\n")
            f_out.write("property float x\n")
            f_out.write("property float y\n")
            f_out.write("property float z\n")
            f_out.write(f"element face {num_faces}\n")
            f_out.write("property list uchar int vertex_indices\n")
            f_out.write("end_header\n")

            # Vertices
            for i in range(num_vertices):
                x, y, z = v[i]
                f_out.write(f"{x} {y} {z}\n")

            # Faces
            for i in range(num_faces):
                i0, i1, i2 = f[i]
                f_out.write(f"3 {int(i0)} {int(i1)} {int(i2)}\n")

    def extract_sdf_to_temp_ply(
        self,
        sdf_model: torch.nn.Module,
        resolution: int | None = None,
        bbox: list | None = None,
        isovalue: float | None = None,
        device: str | None = None,
    ) -> str:
        """
        Calls the WP5 mesh extractor and writes a PLY file to a temporary directory.
        """
        mesh_config = load_mesh_extraction_config()
        if resolution is not None:
            mesh_config = replace(mesh_config, base_resolution=resolution)
        if bbox is not None:
            if len(bbox) == 6 and not isinstance(bbox[0], list):
                bbox = [[bbox[0], bbox[1]], [bbox[2], bbox[3]], [bbox[4], bbox[5]]]
            mesh_config = replace(mesh_config, bbox=bbox)
        if isovalue is not None:
            mesh_config = replace(mesh_config, isovalue=isovalue)
        if device is not None:
            mesh_config = replace(mesh_config, device=device)

        sdf_model = sdf_model.to(mesh_config.device)
        sdf_model.eval()

        extractor = MeshSDFExtractor(config=mesh_config)
        _bbox = mesh_config.bbox
        if _bbox is None:
            _prism = get_outer_prism_bounds_mm()
            _bbox = [[_prism["x"][0], _prism["x"][1]], [_prism["y"][0], _prism["y"][1]], [_prism["z"][0], _prism["z"][1]]]
        extractor.set_grid(bbox=_bbox, resolution=mesh_config.base_resolution)
        vertices, faces = extractor.extract_mesh(sdf_model)

        # Create temporary directory and file path
        tmp_dir = tempfile.mkdtemp(prefix="mesh_sdf_")
        ply_path = os.path.join(tmp_dir, "mesh_sdf_output.ply")

        # Write PLY
        self.write_ply(vertices, faces, ply_path)

        return ply_path
