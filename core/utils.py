import numpy as np
import meshio

def get_boundary_nodes(mesh: meshio.Mesh):
    """
    Given a meshio.Mesh object containing hexahedral cells,
    return the set of node indices that belong to the boundary.
    """
    # Each hex cell has 6 faces, defined by 4 nodes each
    hex_faces = [
        [0, 1, 2, 3],  # bottom
        [4, 5, 6, 7],  # top
        [0, 1, 5, 4],  # front
        [2, 3, 7, 6],  # back
        [0, 3, 7, 4],  # left
        [1, 2, 6, 5],  # right
    ]
    
    # Collect all faces
    face_count = {}
    
    # Extract only hexahedral cells
    if "hexahedron" not in mesh.cells_dict:
        raise ValueError("Mesh does not contain hexahedra.")
        
    for cell in mesh.cells_dict["hexahedron"]:
        for face in hex_faces:
            # Sort face nodes so that shared faces match regardless of orientation
            f = tuple(sorted(cell[face]))
            face_count[f] = face_count.get(f, 0) + 1
    
    # Boundary faces are those that appear only once
    boundary_faces = [f for f, count in face_count.items() if count == 1]
    
    # Extract boundary nodes
    boundary_nodes = set()
    for f in boundary_faces:
        boundary_nodes.update(f)
    
    return np.array(sorted(boundary_nodes))
