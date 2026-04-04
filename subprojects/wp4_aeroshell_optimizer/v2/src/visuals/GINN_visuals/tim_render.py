"""
Mesh Visualization using PyVista

Provides functions to render triangle meshes in an interactive 3D viewer.
Supports both single-mesh and multi-mesh rendering.
"""


from pathlib import Path
from typing import Any

import numpy as np
import torch
from subprojects.wp4_aeroshell_optimizer.v2.src.utils.pyvista_headless import (
    ensure_pyvista_headless,
    is_headless_runtime,
)
from torch import Tensor

pv = ensure_pyvista_headless()

# Default color palette for multi-body rendering
DEFAULT_COLORS = [
    "lightgrey",
    "lightcoral",
    "lightgreen",
    "lightyellow",
    "lightpink",
    "lightsalmon",
    "lightcyan",
    "plum",
]


def _add_origin_reference(
    plotter: Any,
    bounds: tuple[float, float, float, float, float, float],
    plane_opacity: float = 0.15,
    plane_scale: float = 0.25,
    edge_width: float = 4,
) -> None:
    """
    Add origin axes and reference planes to the plotter.

    Args:
        plotter: PyVista plotter object
        bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        plane_opacity: Opacity for reference planes (default: 0.15)
        plane_scale: Scale factor for plane size relative to bounds (default: 0.25)
        edge_width: Line width for plane edges (default: 4)
    """
    

    # Calculate plane size from bounds
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    plane_size = max(x_range, y_range, z_range) * plane_scale

    # Create XY plane (Z=0, blue)
    xy_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(
        xy_plane,
        color="blue",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="blue",
        line_width=edge_width,
    )

    # Create XZ plane (Y=0, green)
    xz_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 1, 0),
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(
        xz_plane,
        color="green",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="green",
        line_width=edge_width,
    )

    # Create YZ plane (X=0, red)
    yz_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(1, 0, 0),
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(
        yz_plane,
        color="red",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="red",
        line_width=edge_width,
    )


def _add_bounding_box(
    plotter: Any,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    color: str = "black",
    line_width: float = 0.5,
    opacity: float = 1.0,
) -> None:
    """
    Add wireframe bounding box to plotter.

    Draws 12 edges of the box defined by xyz_min and xyz_max.

    Args:
        plotter: PyVista plotter object
        xyz_min: Minimum corner of bounding box (x, y, z)
        xyz_max: Maximum corner of bounding box (x, y, z)
        color: Line color (default: 'black')
        line_width: Line width in pixels (default: 0.5)
        opacity: Line opacity (default: 1.0)
    """
    

    # 8 corners of the box
    corners = [
        [xyz_min[0], xyz_min[1], xyz_min[2]],  # 0
        [xyz_max[0], xyz_min[1], xyz_min[2]],  # 1
        [xyz_max[0], xyz_max[1], xyz_min[2]],  # 2
        [xyz_min[0], xyz_max[1], xyz_min[2]],  # 3
        [xyz_min[0], xyz_min[1], xyz_max[2]],  # 4
        [xyz_max[0], xyz_min[1], xyz_max[2]],  # 5
        [xyz_max[0], xyz_max[1], xyz_max[2]],  # 6
        [xyz_min[0], xyz_max[1], xyz_max[2]],  # 7
    ]

    # 12 edges (pairs of corner indices)
    edges = [
        # Bottom face (z=min)
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # Top face (z=max)
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # Vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Create line segments for each edge
    for i, j in edges:
        line = pv.Line(corners[i], corners[j])
        plotter.add_mesh(
            line,
            color=color,
            line_width=line_width,
            opacity=opacity,
        )


def render_mesh(
    verts: Tensor,
    faces: Tensor,
    show: bool = True,
    color: str = "lightgrey",
    show_edges: bool = False,
    origin: bool = True,
):
    """
    Render a triangle mesh using PyVista.

    Args:
        verts: [V, 3] vertex positions
        faces: [F, 3] face indices (0-indexed)
        show: If True, display interactive window. If False, return plotter.
        color: Mesh color (default: 'lightgrey')
        show_edges: If True, show wireframe edges (default: False)
        origin: If True, show origin reference planes (default: True)

    Returns:
        PyVista plotter object (can be used for further customization or saving)

    Example:
        >>> verts = torch.rand(100, 3)
        >>> faces = torch.randint(0, 100, (50, 3))
        >>> render_mesh(verts, faces)  # Opens interactive window
    """

    # Convert to numpy
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Compute bounds for origin reference
    bounds = (
        verts_np[:, 0].min(),
        verts_np[:, 0].max(),  # x_min, x_max
        verts_np[:, 1].min(),
        verts_np[:, 1].max(),  # y_min, y_max
        verts_np[:, 2].min(),
        verts_np[:, 2].max(),  # z_min, z_max
    )

    # PyVista expects faces in VTK format: [n_verts, v0, v1, v2, ...]
    # For triangles: [3, v0, v1, v2, 3, v3, v4, v5, ...]
    n_faces = faces_np.shape[0]
    pv_faces = np.hstack(
        [
            np.full((n_faces, 1), 3, dtype=np.int64),  # 3 verts per face
            faces_np,
        ]
    ).flatten()

    # Create PyVista mesh and compute normals for smooth shading
    mesh = pv.PolyData(verts_np, pv_faces)
    mesh.compute_normals(inplace=True)

    # Set up plotter with nice lighting
    headless = is_headless_runtime()
    plotter = pv.Plotter(off_screen=(not show) or headless)
    plotter.add_mesh(
        mesh,
        color=color,
        show_edges=show_edges,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    plotter.add_axes()
    if origin:
        _add_origin_reference(plotter, bounds)

    if show and not headless:
        plotter.show()

    return plotter


def render_mesh_result(
    mesh_result,
    show: bool = True,
    color: str = "lightgrey",
    show_edges: bool = False,
    origin: bool = True,
    show_bounds: bool = True,
    curves_data: list[tuple] | None = None,
    workplanes_data: list[tuple] | None = None,
    opacity: float = 1.0,
):
    """
    Render a MeshResult object using PyVista.

    Args:
        mesh_result: MeshResult object
        show: If True, display interactive window (default: True)
        color: Mesh color (default: 'lightgrey')
        show_edges: If True, show wireframe edges (default: False)
        origin: If True, show origin reference planes (default: True)
        show_bounds: If True, show bounding box wireframe (default: True)

    Returns:
        PyVista plotter object

    Example:
        >>> from torchcad import CADProgram
        >>> prog = CADProgram.load_from_yaml("sphere.yaml")
        >>> mesh = prog.mesh(backend='skimage')
        >>> render_mesh_result(mesh)
    """

    # MeshResult always contains lists (batch_size >= 1)
    # For visualization, we only render the first batch element
    if mesh_result.batch_size > 1:
        print(
            f"Warning: MeshResult has {mesh_result.batch_size} batches. Rendering only the first batch."
        )

    # Get vertices and faces from first batch
    verts = mesh_result.vertices[0]  # [V, 3]

    if mesh_result.faces is None:
        raise ValueError("MeshResult has no faces. Cannot render.")
    faces = mesh_result.faces[0]  # [F, 3]

    # Convert to numpy
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Compute bounds for origin reference
    bounds = (
        verts_np[:, 0].min(),
        verts_np[:, 0].max(),  # x_min, x_max
        verts_np[:, 1].min(),
        verts_np[:, 1].max(),  # y_min, y_max
        verts_np[:, 2].min(),
        verts_np[:, 2].max(),  # z_min, z_max
    )

    # PyVista expects faces in VTK format
    n_faces = faces_np.shape[0]
    pv_faces = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces_np]).flatten()

    # Create PyVista mesh and compute normals
    mesh = pv.PolyData(verts_np, pv_faces)
    mesh.compute_normals(inplace=True)

    # Set up plotter with nice lighting
    headless = is_headless_runtime()
    plotter = pv.Plotter(off_screen=(not show) or headless)
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    plotter.add_axes()

    if origin:
        _add_origin_reference(plotter, bounds)

    if show_bounds:
        _add_bounding_box(
            plotter,
            mesh_result.xyz_min,
            mesh_result.xyz_max,
            color="black",
            line_width=0.5,
        )

    if curves_data:
        _add_curves_to_plotter(plotter, curves_data)

    if workplanes_data:
        # Use YAML bounding box (not mesh vertex bounds) for workplane sizing
        wp_bounds = (
            mesh_result.xyz_min[0],
            mesh_result.xyz_max[0],
            mesh_result.xyz_min[1],
            mesh_result.xyz_max[1],
            mesh_result.xyz_min[2],
            mesh_result.xyz_max[2],
        )
        _add_workplanes_to_plotter(plotter, workplanes_data, wp_bounds)

    if show and not headless:
        plotter.show()

    return plotter



def render_meshes(
    meshes: dict[str, tuple[Tensor, Tensor]],
    show: bool = True,
    colors: dict[str, str] | None = None,
    show_edges: bool = False,
    origin: bool = True,
    show_bounds: bool = True,
    curves_data: list[tuple] | None = None,
    workplanes_data: list[tuple] | None = None,
    opacity: float = 1.0,
):
    """
    Render multiple meshes in the same viewer with different colors.

    Args:
        meshes: Dict mapping body name to MeshResult
        show: If True, display interactive window. If False, return plotter.
        colors: Optional dict mapping body name to color. If not provided,
                colors are assigned automatically from DEFAULT_COLORS.
        show_edges: If True, show wireframe edges (default: False)
        origin: If True, show origin reference planes (default: True)
        show_bounds: If True, show bounding box wireframe (default: True)

    Returns:
        PyVista plotter object

    Example:
        >>> meshes = {"bracket": surf_mesh, "bolt": surf_mesh2}
        >>> render_meshes(meshes)
    """

    headless = is_headless_runtime()
    plotter = pv.Plotter(off_screen=(not show) or headless)

    # Initialize bound tracking for origin reference
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    z_min, z_max = float("inf"), float("-inf")

    # Track bounds from first mesh (all should have same bounds from YAML)
    bbox_min = None
    bbox_max = None

    for idx, (name, mesh_result) in enumerate(meshes.items()):
        # NEW: MeshResult always contains lists (batch_size >= 1)
        # For visualization, we only render the first batch element
        if mesh_result.batch_size > 1:
            print(
                f"Warning: Body '{name}' has {mesh_result.batch_size} batches. Rendering only the first batch."
            )

        # Get vertices and faces from first batch
        verts = mesh_result.vertices[0]  # [V, 3]

        if mesh_result.faces is None:
            raise ValueError(f"Body '{name}' has no faces. Cannot render.")
        faces = mesh_result.faces[0]  # [F, 3]

        # Get color for this body
        if colors and name in colors:
            color = colors[name]
        else:
            color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        # Convert to numpy
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        # Update bounds for origin reference
        x_min = min(x_min, verts_np[:, 0].min())
        x_max = max(x_max, verts_np[:, 0].max())
        y_min = min(y_min, verts_np[:, 1].min())
        y_max = max(y_max, verts_np[:, 1].max())
        z_min = min(z_min, verts_np[:, 2].min())
        z_max = max(z_max, verts_np[:, 2].max())

        # Store bounding box from first mesh
        if bbox_min is None:
            bbox_min = mesh_result.xyz_min
            bbox_max = mesh_result.xyz_max

        # PyVista format
        n_faces = faces_np.shape[0]
        pv_faces = np.hstack(
            [np.full((n_faces, 1), 3, dtype=np.int64), faces_np]
        ).flatten()

        mesh = pv.PolyData(verts_np, pv_faces)
        mesh.compute_normals(inplace=True)

        plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
            label=name,
        )

    plotter.add_axes()
    if origin:
        bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
        _add_origin_reference(plotter, bounds)

    if show_bounds and bbox_min is not None:
        _add_bounding_box(
            plotter,
            bbox_min,
            bbox_max,
            color="black",
            line_width=0.5,
        )

    if curves_data:
        _add_curves_to_plotter(plotter, curves_data)

    if workplanes_data and bbox_min is not None:
        # Use YAML bounding box (not mesh vertex bounds) for workplane sizing
        wp_bounds = (
            bbox_min[0],
            bbox_max[0],
            bbox_min[1],
            bbox_max[1],
            bbox_min[2],
            bbox_max[2],
        )
        _add_workplanes_to_plotter(plotter, workplanes_data, wp_bounds)

    plotter.add_legend()

    if show and not headless:
        plotter.show()

    return plotter


def _add_curves_to_plotter(
    plotter,
    curves_data: list[tuple],
) -> None:
    """
    Add smooth curve splines to an existing plotter.

    Args:
        plotter: PyVista plotter object to add curves to
        curves_data: List of (polyline_np, cp_np, color, name) tuples.
                     polyline_np is (S, 3) numpy array of curve sample points.
                     cp_np is (N_cp, 3) numpy array of control points.
    """

    for polyline_np, cp_np, color, name in curves_data:
        spline = pv.Spline(polyline_np, n_points=200)
        plotter.add_mesh(
            spline,
            color=color,
            line_width=3,
            label=name,
        )

        # Add control point markers
        cp_poly = pv.PolyData(cp_np)
        plotter.add_mesh(
            cp_poly,
            color=color,
            point_size=12,
            render_points_as_spheres=False,
            style="points",
        )


def _add_workplanes_to_plotter(
    plotter,
    workplanes_data: list[tuple],
    bounds: tuple,
) -> None:
    """
    Add semi-transparent workplane quads with normal/axis arrows.

    Args:
        plotter: PyVista plotter object
        workplanes_data: List of (origin_np, normal_np, u_np, v_np, color, name)
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    """

    for origin_np, normal_np, u_np, v_np, color, name in workplanes_data:
        # Auto-size: project bounding box ranges onto u and v directions
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]

        ranges = np.array([x_range, y_range, z_range])
        i_size = abs(np.dot(ranges, np.abs(u_np))) * 0.5
        j_size = abs(np.dot(ranges, np.abs(v_np))) * 0.5

        # Fallback: use max range * 0.5 if projection gives 0
        if i_size < 1e-6:
            i_size = max(ranges) * 0.5
        if j_size < 1e-6:
            j_size = max(ranges) * 0.5

        # Create plane quad centered at origin
        plane = pv.Plane(
            center=origin_np,
            direction=normal_np,
            i_size=i_size,
            j_size=j_size,
            i_resolution=1,
            j_resolution=1,
        )
        plotter.add_mesh(
            plane,
            color=color,
            opacity=0.2,
            show_edges=True,
            edge_color=color,
            line_width=3,
            label=f"wp:{name}",
        )

        # Normal arrow (from center, along normal)
        arrow_scale = min(i_size, j_size) * 0.2
        plotter.add_arrows(
            np.array([origin_np]),
            np.array([normal_np]) * arrow_scale,
            color="white",
        )

        # u-axis arrow (from center, along u)
        u_arrow_scale = i_size * 0.15
        plotter.add_arrows(
            np.array([origin_np]),
            np.array([u_np]) * u_arrow_scale,
            color="red",
        )

        # v-axis arrow (from center, along v)
        v_arrow_scale = j_size * 0.15
        plotter.add_arrows(
            np.array([origin_np]),
            np.array([v_np]) * v_arrow_scale,
            color="green",
        )


def _add_sdf_slice_widget(
    plotter,
    sdf_fn,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    plane: str = "xy",
    slice_resolution: int = 256,
    batch_idx: int = 0,
) -> None:
    """
    Add an interactive SDF slice plane widget to an existing plotter.

    The user drags an axis-aligned plane through the 3D bounding box.
    The callback evaluates the SDF on a 2D grid at the plane offset and
    displays filled colors (RdBu_r), regular isolines, and a bold zero contour.

    Args:
        plotter: PyVista plotter object
        sdf_fn: Callable that takes [N, 3] Tensor and returns [N] or [B, N] Tensor
        xyz_min: Bounding box minimum corner (x, y, z)
        xyz_max: Bounding box maximum corner (x, y, z)
        plane: Slice plane - 'xy', 'xz', or 'yz'
        slice_resolution: Number of grid points per axis (default: 256)
        batch_idx: Which batch element to use for SDF evaluation (default: 0)
    """

    PLANE_CFG = {
        "xy": {"grid_axes": (0, 1), "normal_axis": 2, "normal_vec": (0, 0, 1)},
        "xz": {"grid_axes": (0, 2), "normal_axis": 1, "normal_vec": (0, 1, 0)},
        "yz": {"grid_axes": (1, 2), "normal_axis": 0, "normal_vec": (1, 0, 0)},
    }

    if plane not in PLANE_CFG:
        raise ValueError(
            f"Invalid slice plane '{plane}'. Must be one of: 'xy', 'xz', 'yz'"
        )

    cfg = PLANE_CFG[plane]
    nax = cfg["normal_axis"]
    ax0, ax1 = cfg["grid_axes"]

    # Initial plane position: center of bounding box along normal axis
    center = [(xyz_min[i] + xyz_max[i]) / 2.0 for i in range(3)]

    # Pre-compute the fixed 2D grid (u, v never change — only offset moves)
    u = np.linspace(xyz_min[ax0], xyz_max[ax0], slice_resolution)
    v = np.linspace(xyz_min[ax1], xyz_max[ax1], slice_resolution)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    n_pts = slice_resolution * slice_resolution

    def on_plane_move(normal, origin):
        # Clamp offset to meshing bounds
        offset = float(np.clip(origin[nax], xyz_min[nax], xyz_max[nax]))

        # Build [N, 3] query points (F-order to match StructuredGrid point ordering)
        points = torch.zeros(n_pts, 3)
        points[:, ax0] = torch.from_numpy(uu.ravel(order="F")).float()
        points[:, ax1] = torch.from_numpy(vv.ravel(order="F")).float()
        points[:, nax] = offset

        # Evaluate SDF
        with torch.no_grad():
            sdf_vals = sdf_fn(points)
            if sdf_vals.dim() == 2:
                sdf_vals = sdf_vals[batch_idx]

        # Build StructuredGrid for display
        coords = [None, None, None]
        coords[ax0] = uu
        coords[ax1] = vv
        coords[nax] = np.full_like(uu, float(offset))
        grid = pv.StructuredGrid(coords[0], coords[1], coords[2])
        sdf_np = sdf_vals.cpu().numpy()
        grid.point_data["sdf"] = sdf_np

        lo, hi = float(sdf_np.min()), float(sdf_np.max())

        # Symmetric color range so SDF=0 is always at the center (white)
        abs_max = max(abs(lo), abs(hi))
        plotter.add_mesh(
            grid,
            name="sdf_slice",
            scalars="sdf",
            cmap="RdBu_r",
            clim=[-abs_max, abs_max],
            opacity=0.8,
            show_scalar_bar=False,
        )

        # --- Isolines colored by local SDF value ---
        iso_spacing = 0.125
        levels = np.arange(np.ceil(lo / iso_spacing) * iso_spacing, hi, iso_spacing)
        levels = levels[levels != 0.0]  # exclude zero (drawn separately)

        if len(levels) > 0:
            isocontours = grid.contour(isosurfaces=levels.tolist(), scalars="sdf")
            if isocontours.n_points > 0:
                plotter.add_mesh(
                    isocontours,
                    name="sdf_isolines",
                    scalars="sdf",
                    cmap="RdBu_r",
                    clim=[-abs_max, abs_max],
                    line_width=1.5,
                    opacity=1.0,
                    show_scalar_bar=False,
                )
            else:
                plotter.remove_actor("sdf_isolines", render=False)
        else:
            plotter.remove_actor("sdf_isolines", render=False)

        # Zero-level isoline: bold black, semi-transparent
        if lo < 0.0 < hi:
            zero_contour = grid.contour(isosurfaces=[0.0], scalars="sdf")
            if zero_contour.n_points > 0:
                plotter.add_mesh(
                    zero_contour,
                    name="sdf_zero",
                    color="black",
                    line_width=4.0,
                    opacity=1.0,
                )
            else:
                plotter.remove_actor("sdf_zero", render=False)
        else:
            plotter.remove_actor("sdf_zero", render=False)

    plotter.add_plane_widget(
        on_plane_move,
        normal=cfg["normal_vec"],
        origin=center,
        bounds=(xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2]),
        factor=1.0,
        normal_rotation=False,
        outline_translation=False,
        origin_translation=True,
        implicit=True,
        tubing=False,
        outline_opacity=0.0,
    )

    # Trigger initial render at center position
    on_plane_move(cfg["normal_vec"], center)


def save_mesh_ply(verts: Tensor, faces: Tensor, filename: str):
    """
    Save a mesh to PLY file (Polygon File Format).
    
    PLY format is superior to STL for complex meshes as it supports
    per-vertex attributes and better preserves geometry data.

    Args:
        verts: [V, 3] vertex positions
        faces: [F, 3] face indices
        filename: Output file path (should end in .ply)

    Example:
        >>> save_mesh_ply(verts, faces, "output.ply")
    """

    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    n_faces = faces_np.shape[0]
    pv_faces = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces_np]).flatten()

    mesh = pv.PolyData(verts_np, pv_faces)
    mesh.save(filename)


def save_mesh_obj(verts: Tensor, faces: Tensor, filename: str):
    """
    Save a mesh to OBJ file.

    Args:
        verts: [V, 3] vertex positions
        faces: [F, 3] face indices (0-indexed, will be converted to 1-indexed)
        filename: Output file path (should end in .obj)

    Example:
        >>> save_mesh_obj(verts, faces, "output.obj")
    """
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    with open(filename, "w") as f:
        # Write vertices
        f.writelines(f"v {v[0]} {v[1]} {v[2]}\n" for v in verts_np)

        # Write faces (OBJ uses 1-indexed vertices)
        f.writelines(
            f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n" for face in faces_np
        )


def save_meshes_ply(meshes: dict[str, tuple[Tensor, Tensor]], filename: str):
    """
    Save multiple meshes to a single PLY file.

    All meshes are concatenated into one file. Each body becomes part 
    of the combined mesh. PLY format preserves geometry better than STL.

    Args:
        meshes: Dict mapping body name to (verts, faces) tuple
        filename: Output file path (should end in .ply)

    Example:
        >>> meshes = {"bracket": (verts1, faces1), "bolt": (verts2, faces2)}
        >>> save_meshes_ply(meshes, "assembly.ply")
    """

    combined_verts = []
    combined_faces = []
    vertex_offset = 0

    for name, (verts, faces) in meshes.items():
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        combined_verts.append(verts_np)
        combined_faces.append(faces_np + vertex_offset)
        vertex_offset += len(verts_np)

    all_verts = np.vstack(combined_verts)
    all_faces = np.vstack(combined_faces)

    n_faces = all_faces.shape[0]
    pv_faces = np.hstack(
        [np.full((n_faces, 1), 3, dtype=np.int64), all_faces]
    ).flatten()

    mesh = pv.PolyData(all_verts, pv_faces)
    mesh.save(filename)


def save_meshes_obj(meshes: dict[str, tuple[Tensor, Tensor]], filename: str):
    """
    Save multiple meshes to a single OBJ file.

    Each body is written as a separate group (g) in the OBJ file,
    allowing them to be identified when imported into other software.

    Args:
        meshes: Dict mapping body name to (verts, faces) tuple
        filename: Output file path (should end in .obj)

    Example:
        >>> meshes = {"bracket": (verts1, faces1), "bolt": (verts2, faces2)}
        >>> save_meshes_obj(meshes, "assembly.obj")
    """
    vertex_offset = 0

    with open(filename, "w") as f:
        f.write("# TorchCAD multi-body export\n\n")

        # First pass: write all vertices
        for name, (verts, faces) in meshes.items():
            verts_np = verts.detach().cpu().numpy()
            f.write(f"# Vertices for {name}\n")
            f.writelines(f"v {v[0]} {v[1]} {v[2]}\n" for v in verts_np)
            f.write("\n")

        # Second pass: write faces with groups
        vertex_offset = 0
        for name, (verts, faces) in meshes.items():
            verts_np = verts.detach().cpu().numpy()
            faces_np = faces.detach().cpu().numpy()

            f.write(f"g {name}\n")
            for face in faces_np:
                # OBJ is 1-indexed, plus offset for previous bodies
                f.write(
                    f"f {face[0] + 1 + vertex_offset} {face[1] + 1 + vertex_offset} {face[2] + 1 + vertex_offset}\n"
                )
            f.write("\n")

            vertex_offset += len(verts_np)


# Map view names to PyVista camera position strings
VIEW_CAMERAS = {
    "iso": "iso",
    "front": "yz",  # looking along -X
    "back": "-yz",  # looking along +X
    "left": "xz",  # looking along -Y
    "right": "-xz",  # looking along +Y
    "top": "xy",  # looking along -Z
    "bottom": "-xy",  # looking along +Z
}


_MULTI_BODY_PALETTE = [
    "lightgrey",
    "steelblue",
    "coral",
    "mediumseagreen",
    "orchid",
    "goldenrod",
    "slategrey",
    "tomato",
]


def _mesh_result_to_pv(mesh_result, np_module):
    """Convert a single MeshResult (batch 0) to PyVista PolyData + bounds."""
    if mesh_result.batch_size > 1:
        print(
            f"Warning: MeshResult has {mesh_result.batch_size} batches. "
            "Rendering only the first batch."
        )

    verts = mesh_result.vertices[0]
    if mesh_result.faces is None:
        raise ValueError("MeshResult has no faces. Cannot render.")
    faces = mesh_result.faces[0]

    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    n_faces = faces_np.shape[0]
    if n_faces == 0:
        return None, None

    pv_faces = np_module.hstack(
        [np_module.full((n_faces, 1), 3, dtype=np_module.int64), faces_np]
    ).flatten()
    pv_mesh = pv.PolyData(verts_np, pv_faces)
    pv_mesh.compute_normals(inplace=True)

    bounds = (
        verts_np[:, 0].min(),
        verts_np[:, 0].max(),
        verts_np[:, 1].min(),
        verts_np[:, 1].max(),
        verts_np[:, 2].min(),
        verts_np[:, 2].max(),
    )
    return pv_mesh, bounds


def save_mesh_image(
    mesh_result,
    path: str,
    view: str = "iso",
    format: str = "png",
    color: str = "lightgrey",
    colors: dict[str, str] | None = None,
    opacities: dict[str, float] | None = None,
    show_edges: bool = False,
    origin: bool = True,
    show_bounds: bool = True,
    window_size: tuple[int, int] = (800, 600),
) -> None:
    """
    Save mesh visualization to image file.

    Args:
        mesh_result: MeshResult object, or dict[str, MeshResult] for multi-body
        path: Output file path (extension added if not present)
        view: Camera view ('iso', 'front', 'back', 'left', 'right', 'top', 'bottom')
        format: Output format ('png', 'jpg', 'pdf')
        color: Mesh color for single-body (default: 'lightgrey')
        colors: Color dict for multi-body (default: auto-assign from palette)
        opacities: Opacity dict for multi-body (default: 1.0 for all). Values in [0, 1].
        show_edges: Show wireframe edges (default: False)
        origin: Show origin reference planes (default: True)
        show_bounds: Show bounding box wireframe (default: True)
        window_size: Image dimensions (width, height) in pixels

    Example:
        >>> mesh = prog.mesh(backend='skimage')
        >>> save_mesh_image(mesh, 'output.png', view='iso')
        >>> # Multi-body:
        >>> meshes = prog.mesh(backend='skimage')  # dict
        >>> save_mesh_image(meshes, 'assembly.png', colors={'wing': 'steelblue'})
    """

    if view not in VIEW_CAMERAS:
        raise ValueError(
            f"Unknown view '{view}'. Available views: {list(VIEW_CAMERAS.keys())}"
        )

    # Ensure path has correct extension
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    # Normalize to list of (name, MeshResult) pairs
    if isinstance(mesh_result, dict):
        bodies = list(mesh_result.items())
    else:
        bodies = [(None, mesh_result)]

    # Convert all bodies to PyVista meshes
    pv_meshes: list[tuple[str | None, Any]] = []
    all_bounds = []
    for name, mr in bodies:
        pv_mesh, body_bounds = _mesh_result_to_pv(mr, np)
        if pv_mesh is not None:
            pv_meshes.append((name, pv_mesh))
            all_bounds.append(body_bounds)

    if not pv_meshes:
        raise ValueError("No non-empty meshes to render.")

    # Unified bounds across all bodies
    bounds = (
        min(b[0] for b in all_bounds),
        max(b[1] for b in all_bounds),
        min(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
        min(b[4] for b in all_bounds),
        max(b[5] for b in all_bounds),
    )

    # Set up off-screen plotter
    plotter = pv.Plotter(off_screen=True, window_size=window_size)

    for i, (name, pv_mesh) in enumerate(pv_meshes):
        if name is not None and colors and name in colors:
            body_color = colors[name]
        elif name is None:
            body_color = color
        else:
            body_color = _MULTI_BODY_PALETTE[i % len(_MULTI_BODY_PALETTE)]
        body_opacity = 1.0
        if name is not None and opacities and name in opacities:
            body_opacity = opacities[name]
        plotter.add_mesh(
            pv_mesh,
            color=body_color,
            opacity=body_opacity,
            show_edges=show_edges,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
            label=name,
        )

    plotter.add_axes()

    if origin:
        _add_origin_reference(plotter, bounds)

    if show_bounds:
        # Use unified bounds for bounding box
        if isinstance(mesh_result, dict):
            # Compute unified xyz_min/max from all bodies
            first_mr = next(iter(mesh_result.values()))
            xyz_min = list(first_mr.xyz_min)
            xyz_max = list(first_mr.xyz_max)
            for mr in mesh_result.values():
                for j in range(3):
                    xyz_min[j] = min(xyz_min[j], mr.xyz_min[j])
                    xyz_max[j] = max(xyz_max[j], mr.xyz_max[j])
            _add_bounding_box(
                plotter,
                tuple(xyz_min),
                tuple(xyz_max),
                color="black",
                line_width=0.5,
            )
        else:
            _add_bounding_box(
                plotter,
                mesh_result.xyz_min,
                mesh_result.xyz_max,
                color="black",
                line_width=0.5,
            )

    # Set camera view (PyVista handles zoom automatically)
    plotter.camera_position = VIEW_CAMERAS[view]

    # Save screenshot
    plotter.screenshot(str(path))