import numpy as np
import trimesh


def facenormal_angle(
    mesh: trimesh.Trimesh, sun_direction=np.array([0, 0, 1])
) -> np.ndarray:
    # nserck
    """
    Compute the incidence angle between each face normal and the sun vector.

    Angles are clipped to [0°, 90°] so faces pointing away from the sun saturate at
    90°. This avoids negative cosines downstream.

    Args:
        mesh (trimesh.Trimesh): Mesh whose face normals should be compared.
        sun_direction (np.ndarray): Vector pointing from the panel towards the sun.

    Returns:
        np.ndarray: Angle in degrees for every face in `mesh`.
    """

    # 1. Normalize the sun direction
    sun_direction_normalized = sun_direction / np.linalg.norm(sun_direction)

    # 2. Get the face normals
    face_normals = mesh.face_normals

    # 3. Calculate the cosine of the angle (dot product)
    cos_angle = np.dot(face_normals, sun_direction_normalized)

    # 4. CLAMP the cosine value to be between 0.0 and 1.0
    # This forces all angles > 90 degrees (negative cos_angle) to 90 degrees (cos_angle = 0.0)
    # The clamp to 1.0 is kept for numerical stability.
    cos_angle_clamped = np.clip(cos_angle, 0.0, 1.0)

    # 5. Calculate the angle in radians (now guaranteed to be between 0 and pi/2)
    angle_rad = np.arccos(cos_angle_clamped)

    # 6. Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def powerratio_iam(theta_grad, b0=0.05):
    # nserck
    """
    Convert incidence angles to a relative power weight via the ASHRAE IAM model.

    Args:
        theta_grad (float | np.ndarray): Incidence angle(s) in degrees.
        b0 (float): IAM coefficient; larger values penalise oblique angles more.

    Returns:
        np.ndarray: Weight mask between 0.0 and 1.0 (shadowing is not considered).
    """

    # Ensure all inputs are treated as arrays for vectorized operation
    theta_grad = np.asarray(theta_grad)

    # 2. Umrechnung des Winkels in Radiant (für numpy.cos)
    theta_rad = np.radians(theta_grad)
    cos_theta = np.cos(theta_rad)

    #

    # For the IAM formula, we must prevent division by zero (when cos_theta is 0, i.e., theta=90°).
    # clip cos_theta to a minimum value slightly above 0 to avoid division by zero
    cos_theta_safe = np.clip(cos_theta, a_min=1e-6, a_max=1.0)

    # 3. Berechnung des Incidence Angle Modifier (IAM) nach ASHRAE
    # Formel K(theta) = 1 - b0 * (1/cos(theta) - 1)
    iam_factor = 1.0 - b0 * ((1.0 / cos_theta_safe) - 1.0)

    # 4. Berechnung des Gesamtverhältnisses
    ratio = iam_factor  ##  * cos_theta fehlerhaft?

    # 5. ANGLE CHECK (Vectorized Fix): Set ratio to 0.0 where angle >= 90°
    # np.where checks the condition element-wise:
    # - If True (angle >= 90): returns 0.0
    # - If False (angle < 90): returns the calculated 'verhaeltnis'
    ratio_final = np.where(theta_grad >= 90.0, 0.0, ratio)
    ratio_final = np.clip(ratio_final, a_min=0.0, a_max=None)

    return ratio_final


def sky_face(mesh, threshold=0.01) -> trimesh.Trimesh:
    # nserck
    """
    Filter a mesh down to faces whose normals have a positive Z component.

    Args:
        mesh (trimesh.Trimesh): Source mesh in the same units as downstream code.
        threshold (float): Minimum Z-component of the face normal to keep the face.

    Returns:
        trimesh.Trimesh: Mesh containing only faces that exceed `threshold`.
    """
    threshold = 0.01
    face_normals = mesh.face_normals
    z_components = face_normals[:, 2]
    upward_facing_mask = z_components > threshold
    top_face_indices = np.where(upward_facing_mask)[0]
    top_faces = mesh.faces[top_face_indices]
    top_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=top_faces)
    return top_mesh


def shadowing_factor(
    mesh: trimesh.Trimesh,
    sun_direction=np.array([0, 0, -1]),
) -> np.ndarray:
    # nserck
    """
    Estimate whether each face centre is self-shadowed via ray casting.

    Args:
        mesh (trimesh.Trimesh): Full geometry (not just sky-facing subsets).
        sun_direction (np.ndarray): Direction the sun rays travel towards (usually
            the negative of the sun vector, e.g. `[0, 0, -1]`).

    Returns:
        np.ndarray: Boolean mask where `True` indicates the ray hit geometry,
        meaning the face is shadowed.
    """
    # TODO: adapt Function so it can evaluate shadowing given the entire mesh,
    # but then only outputs the mask for the relevant area (eg. Sky facing area, intended solar area..)

    intersector = mesh.ray

    sun_direction = sun_direction / np.linalg.norm(sun_direction)

    # 3. Bestimmen Sie die Startpunkte der Strahlen
    # Wir verwenden die Zentroiden (Mittelpunkte) jeder Facette
    epsilon = 1e-4
    face_normals = mesh.face_normals

    # Verschieben Sie den Startpunkt jeder Facette leicht entlang ihrer Normalen
    ray_origins_offset = mesh.triangles_center + epsilon * face_normals

    # 4. Definieren Sie die Richtungen der Strahlen
    # Die Strahlen zeigen in die entgegengesetzte Sonnenrichtung.
    # Jeder Strahl hat dieselbe Richtung.
    ray_directions = np.tile(-sun_direction, (len(ray_origins_offset), 1))

    # 5. Raycasting durchführen
    # Wir nutzen intersects_any, was sehr schnell ist:
    # Es prüft, ob JEDER Strahl ÜBERHAUPT eine Oberfläche trifft.
    # Wenn ein Strahl trifft, ist die Facette (deren Zentrum) beschattet.
    # hit_any ist ein Boolean-Array, so groß wie die Anzahl der Facetten.
    hit_any = intersector.intersects_any(
        ray_origins=ray_origins_offset, ray_directions=ray_directions
    )

    # hit_any = True bedeutet, dass der Strahl die Oberfläche getroffen hat -> Facette ist beschattet
    # hit_any = False bedeutet, dass der Strahl die Oberfläche nicht getroffen hat -> Facette ist unbeschattet

    # 6. Ergebnisse interpretieren
    # shadowed_faces_indices = np.where(hit_any)[0]
    # illuminated_faces_indices = np.where(~hit_any)[0]
    return hit_any
