from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements


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


def cube_gen(out_file: Path, disc: tuple[int, int, int] = (4, 4, 4)):
    import gmsh

    gmsh.initialize()
    N1, N2, N3 = disc
    p0 = gmsh.model.geo.addPoint(0, 0, 0)
    l0 = gmsh.model.geo.extrude([(0, p0)], 1, 0, 0, [N1], [1])
    s0 = gmsh.model.geo.extrude([l0[1]], 0, 1, 0, [N2], [1], recombine=True)
    v0 = gmsh.model.geo.extrude([s0[1]], 0, 0, 1, [N3], [1], recombine=True)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [v0[1][1]])
    gmsh.model.mesh.generate(3)
    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True)
    gmsh.write(str(out_file))
    gmsh.finalize()


if __name__ == "__main__":
    cube_gen(Path("private/big_cube.msh"))
    # move_modes(Path(r"examples/Square_mesh/square_rot.msh"), "square_rot_bad_3.msh")
