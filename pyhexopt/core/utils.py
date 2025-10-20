from collections import defaultdict
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

_EPS = 1e-12


def _hex_face_keys_from_cell(cell: np.ndarray) -> list[tuple[int, ...]]:
    """
    Given one hexahedron cell (8 node indices), return its 6 faces
    as canonical (sorted tuple) keys of Python ints.
    Assumes the cell array lists the first 4 nodes as bottom and next 4 as top.
    """
    n = [int(x) for x in cell]
    faces = [
        frozenset((n[0], n[1], n[2], n[3])),  # bottom
        frozenset((n[4], n[5], n[6], n[7])),  # top
        frozenset((n[0], n[1], n[5], n[4])),
        frozenset((n[1], n[2], n[6], n[5])),
        frozenset((n[2], n[3], n[7], n[6])),
        frozenset((n[3], n[0], n[4], n[7])),
    ]
    return faces


def _hex_faces_ordered_from_cell(cell: np.ndarray) -> list[tuple[int, ...]]:
    """
    Return the 6 faces of a hexahedron cell as ordered tuples of node indices.
    Ordering follows the same convention you used in _hex_face_keys_from_cell,
    but preserves the vertex ordering needed to compute face normals.
    Assumes cell is length 8 and arranged [0..3]=bottom, [4..7]=top.
    """
    n = [int(x) for x in cell]
    faces = [
        (n[0], n[1], n[2], n[3]),  # bottom (quad)
        (n[4], n[5], n[6], n[7]),  # top (quad)
        (n[0], n[1], n[5], n[4]),
        (n[1], n[2], n[6], n[5]),
        (n[2], n[3], n[7], n[6]),
        (n[3], n[0], n[4], n[7]),
    ]
    return faces


def get_boundary_nodes(cells) -> np.ndarray:
    """
    Return sorted numpy array of node indices that belong to the boundary.
    Works robustly by using canonical integer keys for faces.
    """
    face_count = defaultdict(lambda: 0)

    for cell in cells:
        for face_key in _hex_face_keys_from_cell(cell):
            face_count[face_key] += 1

    # faces that appear only once are boundary faces
    boundary_nodes = set()
    for face_key, count in face_count.items():
        if count == 1:
            boundary_nodes.update(face_key)

    return np.array(sorted(boundary_nodes), dtype=int)


def get_boundary_faces(cells) -> list[tuple[int, ...]]:
    """
    Return ordered boundary faces (as tuples of node indices).
    A face is considered boundary if it appears exactly once across all cells.
    Ordering is preserved from the parent cell's ordering so normals can be computed.
    """
    face_count = {}
    for cell in cells:
        for face in _hex_faces_ordered_from_cell(cell):
            key = frozenset(face)
            if key in face_count:
                face_count[key][0] += 1
            else:
                face_count[key] = [1, tuple(face)]

    # keep only faces occurring once
    boundary_faces = [ordered for count, ordered in face_count.values() if count == 1]
    return boundary_faces


def face_normal(points: np.ndarray, face: tuple[int, ...]) -> np.ndarray:
    """
    Compute a unit normal for a face given vertex coordinates.
    Works for tri (len==3) and quad (len==4); returns zero vector for degenerate faces.
    """
    coords = points[np.array(face, dtype=int)]
    if coords.shape[0] == 3:  # noqa: PLR2004
        v0 = coords[1] - coords[0]
        v1 = coords[2] - coords[0]
        n = np.cross(v0, v1)
    elif coords.shape[0] == 4:  # noqa: PLR2004
        # split quad into two triangles (0,1,2) and (0,2,3)
        v01 = coords[1] - coords[0]
        v02 = coords[2] - coords[0]
        n1 = np.cross(v01, v02)
        v02b = coords[2] - coords[0]
        v03 = coords[3] - coords[0]
        n2 = np.cross(v02b, v03)
        n = n1 + n2
    else:
        # general polygon fallback: best-fit normal via SVD
        pts_centered = coords - coords.mean(axis=0)
        try:
            _, s, vh = np.linalg.svd(pts_centered, full_matrices=False)
            n = vh[-1] * s[-1]
        except np.linalg.LinAlgError:
            n = np.zeros(3)
    norm = np.linalg.norm(n)
    if norm < _EPS:
        return np.zeros(3)
    return n / norm


def compute_face_normals(points: np.ndarray, faces: list[tuple[int, ...]]) -> np.ndarray:
    """
    Vectorized-ish loop to compute normals for a list of ordered faces.
    Returns an array shape (F,3).
    """
    normals = [face_normal(points, f) for f in faces]
    return np.vstack(normals) if len(normals) > 0 else np.zeros((0, 3))


def build_vertex_to_face_adjacency(n_points: int, faces: list[tuple[int, ...]]) -> list[list[int]]:
    """
    Build a simple adjacency: for each vertex index 0..n_points-1 return a Python list
    of face indices that touch that vertex.
    (This is simple to test; later you can produce CSR arrays for JAX.)
    """
    vert_to_faces = [[] for _ in range(n_points)]
    for fi, face in enumerate(faces):
        for v in face:
            vert_to_faces[int(v)].append(fi)
    return vert_to_faces


def detect_edge_mask_from_face_normals(
    n_points: int, faces: list[tuple[int, ...]], face_normals: np.ndarray, angle_deg: float = 30.0
) -> np.ndarray:
    """
    Given precomputed face_normals (F,3) and faces (ordered vertex tuples),
    mark vertices as edge_nodes when the max angle between any two adjacent
    boundary-face normals touching the vertex exceeds threshold.

    Returns boolean mask shape (n_points,) True => vertex is a free-edge.
    """
    cos_thresh = np.cos(np.deg2rad(angle_deg))
    vert_to_faces = build_vertex_to_face_adjacency(n_points, faces)
    edge_mask = np.zeros((n_points,), dtype=bool)

    for v_idx, adj in enumerate(vert_to_faces):
        if len(adj) == 0:
            continue  # not on boundary
        if len(adj) == 1:
            # single boundary face touching vertex -> it's an edge/corner
            edge_mask[v_idx] = True
            continue

        normals = face_normals[adj]
        # filter degenerate normals
        norms = np.linalg.norm(normals, axis=1)
        valid = norms > _EPS
        if np.sum(valid) < 2:  # noqa: PLR2004
            continue
        normals = normals[valid]

        # small adjacency sizes; compute pairwise dot and detect sharp angle
        m = normals.shape[0]
        is_edge = False
        for i in range(m):
            for j in range(i + 1, m):
                dot = float(np.dot(normals[i], normals[j]))
                dot = max(min(dot, 1.0), -1.0)
                angle_cos = dot
                # if angle between normals is large (dot small) -> sharp edge
                if angle_cos < cos_thresh:
                    is_edge = True
                    break
            if is_edge:
                break
        edge_mask[v_idx] = is_edge

    return edge_mask


def get_edge_nodes(points, cells, angle_deg: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Top-level function: returns (edge_nodes_array, edge_mask_bool_array).
    Composed of the smaller functions above, easy to unit-test each piece.
    """
    # 1) boundary faces only
    boundary_faces = get_boundary_faces(cells)
    if len(boundary_faces) == 0:
        N = int(np.asarray(points).shape[0])
        return np.array([], dtype=int), np.zeros((N,), dtype=bool)

    # 2) compute normals for boundary faces
    face_normals = compute_face_normals(points, boundary_faces)
    # 3) detect edge mask from face normals
    N = points.shape[0]
    edge_mask = detect_edge_mask_from_face_normals(N, boundary_faces, face_normals, angle_deg=angle_deg)
    edge_nodes = np.nonzero(edge_mask)[0]
    return np.array(sorted(int(x) for x in edge_nodes), dtype=int), edge_mask


def compute_node_normals_from_faces(
    points: np.ndarray,
    faces: list[tuple[int, ...]],
    # face_normals: np.ndarray,
) -> np.ndarray:
    """
    Compute a per-node normal as the average of adjacent boundary face normals.

    Parameters
    ----------
    points : (N,3) array
        Node coordinates (unused except for output length).
    faces : list of tuple[int]
        Boundary faces (ordered tuples of node indices).
    face_normals : (F,3) array
        Unit normals for each boundary face, consistent with `faces`.

    Returns
    -------
    node_normals : (N,3) array
        Per-node averaged unit normals. Zero vector if node not on any boundary face.

    """
    n_points = points.shape[0]
    accum = np.zeros((n_points, 3), dtype=float)
    count = np.zeros(n_points, dtype=int)

    face_normals = compute_face_normals(points, faces)

    # accumulate normals
    for fi, face in enumerate(faces):
        n = face_normals[fi]
        for v in face:
            accum[v] += n
            count[v] += 1

    # normalize
    with np.errstate(invalid="ignore", divide="ignore"):
        node_normals = np.divide(accum, count[:, None], where=count[:, None] > 0)
        norms = np.linalg.norm(node_normals, axis=1, keepdims=True)
        node_normals = np.divide(node_normals, norms, where=norms > _EPS)

    # fill non-surface nodes with zero normal
    node_normals[np.isnan(node_normals)] = 0.0
    return node_normals


def build_tangent_bases(node_normals: np.ndarray, movable_indices: np.ndarray):
    """
    Build an orthonormal tangent basis (t1, t2) for each movable node.
    - points: (N,3)
    - node_normals: (N,3) unit normals (zero for non-boundary)
    - movable_indices: (M,) indices of nodes to build bases for

    Returns:
    - T1: (M,3)
    - T2: (M,3)

    """
    M = movable_indices.shape[0]
    T1 = np.zeros((M, 3), dtype=float)
    T2 = np.zeros((M, 3), dtype=float)

    # choose reference vector a; per-vertex adapt if nearly parallel
    a_default = np.array([1.0, 0.0, 0.0])
    a_alt = np.array([0.0, 1.0, 0.0])

    for i, vi in enumerate(movable_indices):
        n = node_normals[int(vi)]
        norm_n = np.linalg.norm(n)
        if norm_n < _EPS:
            # degenerate normal -> fallback to local PCA or mark not-movable upstream
            # here we set tangents to canonical basis (will be orthogonalized below)
            t1 = np.array([1.0, 0.0, 0.0])
        else:
            # pick a reference vector not parallel to n
            a = a_default if abs(np.dot(n, a_default)) < 0.9 else a_alt  # noqa: PLR2004
            t1 = np.cross(a, n)
            t1_norm = np.linalg.norm(t1)
            if t1_norm < _EPS:
                # fallback to different a
                a = a_alt
                t1 = np.cross(a, n)
                t1_norm = np.linalg.norm(t1)
                if t1_norm < _EPS:
                    # last resort: choose arbitrary orthonormal
                    t1 = np.array([1.0, 0.0, 0.0])
                    t1_norm = 1.0
            t1 = t1 / t1_norm

        t2 = np.cross(n, t1)
        t2_norm = np.linalg.norm(t2)
        if t2_norm < _EPS:
            # if t2 degenerate, pick perp vector
            t2 = np.cross(n, t1 + 1e-6)
            t2_norm = np.linalg.norm(t2)
            if t2_norm < _EPS:
                t2 = np.array([0.0, 1.0, 0.0])
                t2_norm = 1.0
        t2 = t2 / t2_norm

        T1[i] = t1
        T2[i] = t2

    return T1, T2


def get_interior_surface_nodes(points, cells):
    boundary_nodes = get_boundary_nodes(cells)
    edge_nodes, _ = get_edge_nodes(points, cells)
    surface_nodes = np.setdiff1d(boundary_nodes, edge_nodes, assume_unique=True)
    return surface_nodes


@dataclass
class DofData:
    volumic_nodes: jax.Array  # (Nv,)
    surface_nodes: jax.Array  # (Ns,)
    edge_nodes: jax.Array  # (Ne,)
    T1: jax.Array  # (Ns, 3)
    T2: jax.Array  # (Ns, 3)
    n_tot: int
    n_volu: int
    n_surf: int
    is_free: np.ndarray
    is_surface: np.ndarray


def prepare_dof_masks_and_bases(points, cells) -> DofData:
    """
    Split nodes into fixed, surface (2 ddl), and free (3 ddl).
    Returns a structured dataclass containing indices and tangent bases.
    """
    boundary_nodes = get_boundary_nodes(cells)
    edge_nodes, _ = get_edge_nodes(points, cells)
    surface_nodes = get_interior_surface_nodes(points, cells)

    n_tot = points.shape[0]

    # interior => all - boundary
    all_nodes = np.arange(n_tot)
    volumic_nodes = np.setdiff1d(all_nodes, boundary_nodes, assume_unique=True)

    faces = get_boundary_faces(cells)
    normals = compute_node_normals_from_faces(points, faces)

    T1, T2 = build_tangent_bases(normals, surface_nodes)

    is_free = np.zeros(n_tot, dtype=bool)
    is_free[volumic_nodes] = True

    is_surface = np.zeros(n_tot, dtype=bool)
    is_surface[surface_nodes] = True

    return DofData(
        volumic_nodes=jnp.array(volumic_nodes, dtype=jnp.int32),
        surface_nodes=jnp.array(surface_nodes, dtype=jnp.int32),
        edge_nodes=jnp.array(edge_nodes, dtype=jnp.int32),
        T1=jnp.array(T1, dtype=jnp.float32),
        T2=jnp.array(T2, dtype=jnp.float32),
        n_tot=n_tot,
        n_volu=len(volumic_nodes),
        n_surf=len(surface_nodes),
        is_free=is_free,
        is_surface=is_surface,
    )
