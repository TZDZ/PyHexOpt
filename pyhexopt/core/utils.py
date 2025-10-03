import meshio
import numpy as np


def _hex_face_keys_from_cell(cell: np.ndarray) -> list[tuple[int, ...]]:
    """
    Given one hexahedron cell (8 node indices), return its 6 faces
    as canonical (sorted tuple) keys of Python ints.
    Assumes the cell array lists the first 4 nodes as bottom and next 4 as top.
    """
    # convert to plain Python ints once
    n = [int(x) for x in cell]
    faces = [
        (n[0], n[1], n[2], n[3]),  # bottom
        (n[4], n[5], n[6], n[7]),  # top
        (n[0], n[1], n[5], n[4]),
        (n[1], n[2], n[6], n[5]),
        (n[2], n[3], n[7], n[6]),
        (n[3], n[0], n[4], n[7]),
    ]
    # canonicalize each face to a sorted tuple so orientation doesn't matter
    return [tuple(sorted(face)) for face in faces]


def get_boundary_nodes(mesh: meshio.Mesh) -> np.ndarray:
    """
    Return sorted numpy array of node indices that belong to the boundary.
    Works robustly by using canonical integer keys for faces.
    """
    # handle case where mesh may have multiple cell blocks
    if "hexahedron" not in mesh.cells_dict:
        raise ValueError("Mesh does not contain hexahedra.")

    face_count = {}

    for cell in mesh.cells_dict["hexahedron"]:
        for face_key in _hex_face_keys_from_cell(cell):
            face_count[face_key] = face_count.get(face_key, 0) + 1

    # faces that appear only once are boundary faces
    boundary_nodes = set()
    for face_key, count in face_count.items():
        if count == 1:
            # face_key is a tuple of ints
            boundary_nodes.update(face_key)

    return np.array(sorted(boundary_nodes), dtype=int)
