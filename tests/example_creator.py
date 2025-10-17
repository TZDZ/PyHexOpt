from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements, nodes_from_points


def move_modes(mesh_in, mesh_out_name):
    mesh = meshio.read(mesh_in)
    points, cells = extract_points_and_cells(mesh, dtype=jnp.float32, verbose=False)
    disp = jnp.zeros_like(points)  # shape (N,3)

    disp = disp.at[58].set(jnp.array([0.2, 0.1, 0.1], dtype=jnp.float32))
    disp = disp.at[59].set(jnp.array([0.2, 0.2, 0.1], dtype=jnp.float32))
    disp = disp.at[61].set(jnp.array([0.2, 0.1, 0.15], dtype=jnp.float32))
    disp = disp.at[60].set(jnp.array([-0.1, 0.1, 0.1], dtype=jnp.float32))
    new_points = apply_nodal_displacements(points, disp)

    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    path_out = Path(r"examples/Square_mesh/")
    if not path_out.exists():
        path_out.mkdir(parents=True)
    new_mesh.write(path_out / mesh_out_name, file_format="gmsh")


if __name__ == "__main__":
    move_modes(Path(r"examples/Square_mesh/square_rot.msh"), "square_rot_bad_3.msh")
