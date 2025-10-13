import jax
import jax.numpy as jnp
import meshio
import numpy as np


def extract_hex_node_coords(mesh: meshio.Mesh, dtype=jnp.float32, verbose=True) -> jax.Array:
    """
    Extract node coordinates for all hexahedral elements in a meshio mesh.

    Returns
    -------
      node_coords: jax.Array (E, 8, 3)
        coordinates of the 8 nodes for each hexahedral element

    """
    points = mesh.points.astype(np.float64)

    if "hexahedron" in mesh.cells_dict:
        cells = mesh.cells_dict["hexahedron"].astype(np.int64)
    else:
        found = None
        for block in mesh.cells:
            if block.type in ("hexahedron", "hex8", "hex"):
                found = block.data
                break
        if found is None:
            msg = "No hexahedron cells found in mesh."
            raise ValueError(msg)
        cells = found.astype(np.int64)

    E = cells.shape[0]
    if verbose:
        print(f"Mesh has {points.shape[0]} points, {E} hexahedral elements.")

    node_coords = points[cells]  # (E,8,3)
    return jnp.asarray(node_coords, dtype=dtype)


def extract_points_and_cells(mesh: meshio.Mesh, dtype=jnp.float32, verbose=True):
    """
    Extracts node coordinates and hexahedral connectivity from a meshio mesh.

    Returns
    -------
    points : jax.Array (N,3)
        All mesh vertex coordinates (dtype given by argument)
    cells : jax.Array (E,8)
        Hexahedral element connectivity (int32)

    """
    points = mesh.points.astype(np.float64)

    if "hexahedron" in mesh.cells_dict:
        cells = mesh.cells_dict["hexahedron"].astype(np.int64)
    else:
        found = None
        for block in mesh.cells:
            if block.type in ("hexahedron", "hex8", "hex"):
                found = block.data
                break
        if found is None:
            raise ValueError("No hexahedron cells found in mesh.")
        cells = found.astype(np.int64)

    if verbose:
        print(f"Mesh has {points.shape[0]} points, {cells.shape[0]} hexahedral elements.")

    # ensure correct dtype conversions
    points = jnp.asarray(points, dtype=dtype)
    cells = jnp.asarray(cells, dtype=jnp.int32)  # <-- FIX HERE

    return points, cells
