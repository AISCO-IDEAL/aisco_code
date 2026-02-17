"""
Mesh property computation utilities for geometric analysis.

Provides functions to compute mesh surface area, normals, and other
geometric properties needed for benchmarking.
"""

import torch


def compute_mesh_surface_area(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute total surface area of a triangular mesh.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        Scalar tensor with total surface area
    """
    # Get vertices for each face
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)

    # Compute edges
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)

    # Cross product gives 2x triangle area
    cross = torch.cross(e1, e2, dim=1)  # (F, 3)

    # Triangle areas
    areas = 0.5 * torch.norm(cross, dim=1)  # (F,)

    # Total surface area
    total_area = torch.sum(areas)

    return total_area


def compute_triangle_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute individual triangle areas for each face.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        (F,) tensor of triangle areas
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    cross = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(cross, dim=1)

    return areas


def compute_face_normals(vertices: torch.Tensor, faces: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Compute face normals for each triangle.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices
        normalize: If True, return unit normals

    Returns:
        (F, 3) tensor of face normals
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    e1 = v1 - v0
    e2 = v2 - v0

    normals = torch.cross(e1, e2, dim=1)

    if normalize:
        normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)

    return normals


def compute_vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute vertex normals by averaging adjacent face normals (area-weighted).

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        (V, 3) tensor of vertex normals
    """
    if vertices.shape[0] == 0:
        print("Warning: No vertices in mesh, return empty Tensor when computing vertex normals.")
        return torch.empty((0, 3), device=vertices.device)
    # Compute face normals (unnormalized for area weighting)
    face_normals = compute_face_normals(vertices, faces, normalize=False)

    # Accumulate face normals to vertices
    vertex_normals = torch.zeros_like(vertices)

    for i in range(3):
        vertex_normals.index_add_(0, faces[:, i], face_normals)

    # Normalize
    vertex_normals = vertex_normals / (torch.norm(vertex_normals, dim=1, keepdim=True) + 1e-8)

    return vertex_normals


def compute_mesh_volume(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute signed volume of a closed triangular mesh using divergence theorem.

    Volume is computed as: V = (1/6) * sum(dot(v0, cross(v1, v2)))
    for each triangle with vertices v0, v1, v2.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        Scalar tensor with signed volume (positive if faces are oriented outward)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute cross product v1 x v2
    cross_prod = torch.cross(v1, v2, dim=1)

    # Dot product with v0 and sum
    volume = torch.sum(v0 * cross_prod) / 6.0

    return volume


def compute_edge_lengths(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute edge lengths for each triangle.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        Three (F,) tensors with edge lengths: (edge01, edge12, edge20)
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge01 = torch.norm(v1 - v0, dim=1)
    edge12 = torch.norm(v2 - v1, dim=1)
    edge20 = torch.norm(v0 - v2, dim=1)

    return edge01, edge12, edge20


def compute_mesh_bounds(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute axis-aligned bounding box of mesh.

    Args:
        vertices: (V, 3) tensor of vertex coordinates

    Returns:
        (min_coords, max_coords): Each is a (3,) tensor with (x, y, z) bounds
    """
    min_coords = torch.min(vertices, dim=0)[0]
    max_coords = torch.max(vertices, dim=0)[0]

    return min_coords, max_coords


def compute_mean_edge_length(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute mean edge length across all triangles (useful for mesh quality metrics).

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        Scalar tensor with mean edge length
    """
    edge01, edge12, edge20 = compute_edge_lengths(vertices, faces)

    all_edges = torch.cat([edge01, edge12, edge20])
    mean_length = torch.mean(all_edges)

    return mean_length


def compute_mesh_center(vertices: torch.Tensor, method: str = 'centroid') -> torch.Tensor:
    """
    Compute mesh center.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        method: 'centroid' (mean of vertices) or 'bbox' (bounding box center)

    Returns:
        (3,) tensor with center coordinates
    """
    if method == 'centroid':
        return torch.mean(vertices, dim=0)
    elif method == 'bbox':
        min_coords, max_coords = compute_mesh_bounds(vertices)
        return (min_coords + max_coords) / 2.0
    else:
        raise ValueError(f"Unknown method: {method}. Use 'centroid' or 'bbox'.")


def compute_gaussian_curvature_approx(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """
    Approximate Gaussian curvature at vertices using angle defect method.

    For a vertex, Gaussian curvature K ≈ (2π - sum of angles) / area
    where angles are the angles of incident triangles at that vertex.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices
        vertex_normals: Optional precomputed vertex normals

    Returns:
        (V,) tensor of approximate Gaussian curvature at each vertex
    """
    num_vertices = vertices.shape[0]

    # Initialize angle defect and vertex areas
    angle_sum = torch.zeros(num_vertices, device=vertices.device)
    vertex_area = torch.zeros(num_vertices, device=vertices.device)

    # Process each face
    for face_idx in range(faces.shape[0]):
        # Get vertex indices
        v_indices = faces[face_idx]  # (3,)

        # Get vertex positions
        v0, v1, v2 = vertices[v_indices[0]], vertices[v_indices[1]], vertices[v_indices[2]]

        # Compute edges
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # Compute angles at each vertex
        angles = []
        edges_list = [(e0, -e2), (e1, -e0), (e2, -e1)]  # (outgoing, incoming) for each vertex

        for e_out, e_in in edges_list:
            cos_angle = torch.dot(e_out, e_in) / (torch.norm(e_out) * torch.norm(e_in) + 1e-8)
            cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
            angle = torch.acos(cos_angle)
            angles.append(angle)

        # Compute face area
        cross = torch.cross(e0, -e2, dim=0)
        face_area = 0.5 * torch.norm(cross)

        # Distribute face area to vertices (1/3 each)
        vertex_area_contrib = face_area / 3.0

        # Accumulate
        for i in range(3):
            angle_sum[v_indices[i]] += angles[i]
            vertex_area[v_indices[i]] += vertex_area_contrib

    # Compute Gaussian curvature using angle defect
    angle_defect = 2.0 * torch.pi - angle_sum
    gaussian_curvature = angle_defect / (vertex_area + 1e-8)

    return gaussian_curvature


def compute_mesh_statistics(vertices: torch.Tensor, faces: torch.Tensor) -> dict:
    """
    Compute comprehensive mesh statistics for quality assessment.

    Args:
        vertices: (V, 3) tensor of vertex coordinates
        faces: (F, 3) tensor of face vertex indices

    Returns:
        Dictionary with mesh statistics
    """
    stats = {}

    # Basic counts
    stats['num_vertices'] = vertices.shape[0]
    stats['num_faces'] = faces.shape[0]

    # Surface area
    stats['surface_area'] = compute_mesh_surface_area(vertices, faces).item()

    # Volume
    stats['volume'] = compute_mesh_volume(vertices, faces).item()

    # Bounding box
    min_coords, max_coords = compute_mesh_bounds(vertices)
    stats['bbox_min'] = min_coords.cpu().numpy().tolist()
    stats['bbox_max'] = max_coords.cpu().numpy().tolist()
    stats['bbox_diagonal'] = torch.norm(max_coords - min_coords).item()

    # Edge lengths
    edge01, edge12, edge20 = compute_edge_lengths(vertices, faces)
    all_edges = torch.cat([edge01, edge12, edge20])
    stats['mean_edge_length'] = torch.mean(all_edges).item()
    stats['min_edge_length'] = torch.min(all_edges).item()
    stats['max_edge_length'] = torch.max(all_edges).item()
    stats['std_edge_length'] = torch.std(all_edges).item()

    # Triangle areas
    triangle_areas = compute_triangle_areas(vertices, faces)
    stats['mean_triangle_area'] = torch.mean(triangle_areas).item()
    stats['min_triangle_area'] = torch.min(triangle_areas).item()
    stats['max_triangle_area'] = torch.max(triangle_areas).item()

    return stats
