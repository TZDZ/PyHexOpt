from collections import defaultdict

import meshio
import numpy as np


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


def get_boundary_nodes(mesh: meshio.Mesh) -> np.ndarray:
    """
    Return sorted numpy array of node indices that belong to the boundary.
    Works robustly by using canonical integer keys for faces.
    """
    # handle case where mesh may have multiple cell blocks
    if "hexahedron" not in mesh.cells_dict:
        msg = "Mesh does not contain hexahedra."
        raise ValueError(msg)

    face_count = defaultdict(lambda: 0)

    for cell in mesh.cells_dict["hexahedron"]:
        for face_key in _hex_face_keys_from_cell(cell):
            face_count[face_key] += 1

    # faces that appear only once are boundary faces
    boundary_nodes = set()
    for face_key, count in face_count.items():
        if count == 1:
            boundary_nodes.update(face_key)

    return np.array(sorted(boundary_nodes), dtype=int)
