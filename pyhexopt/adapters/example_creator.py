from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_displacements
from pyhexopt.core.utils import prepare_dof_masks_and_bases


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


def cube_gen_2layers(out_file: Path, disc: tuple[int, int, int] = (4, 4, 4)):
    import gmsh

    gmsh.initialize()
    N1, N2, N3 = disc
    p0 = gmsh.model.geo.addPoint(0, 0, 0)
    l0 = gmsh.model.geo.extrude([(0, p0)], 1, 0, 0, [N1], [1])
    s0 = gmsh.model.geo.extrude([l0[1]], 0, 1, 0, [N2], [1], recombine=True)
    v1 = gmsh.model.geo.extrude([(2, s0[1][1])], 0, 0, 0.8, [N3 // 2], [1], recombine=True)
    v2 = gmsh.model.geo.extrude([(2, v1[0][1])], 0, 0, 0.2, [N3 // 2], [1], recombine=True)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [v1[1][1], v2[1][1]])
    gmsh.model.mesh.generate(3)
    if not out_file.parent.exists():
        out_file.parent.mkdir(parents=True)
    gmsh.write(str(out_file))
    gmsh.finalize()


def randomize_nodes(mesh_in: Path, mesh_out: Path, delta: float = 0.1):
    mesh = meshio.read(mesh_in)
    points, cells = extract_points_and_cells(mesh, dtype=jnp.float32)
    dof = prepare_dof_masks_and_bases(points, cells)
    key = jax.random.PRNGKey(42)
    reduced_disps = jnp.concatenate(
        [
            jax.random.uniform(key, (dof.n_volu * 3,), minval=-delta, maxval=delta),
            jax.random.uniform(jax.random.split(key)[1], (dof.n_surf * 2,), minval=-delta, maxval=delta),
        ]
    )
    volu_disps = reduced_disps[: dof.n_volu * 3].reshape((dof.n_volu, 3))
    surf_disps = reduced_disps[dof.n_volu * 3 :].reshape((dof.n_surf, 2))

    all_disps = expand_displacements(
        volu_disps,
        surf_disps,
        dof.volumic_nodes,
        dof.surface_nodes,
        dof.T1,
        dof.T2,
        dof.n_tot,
    )

    moved_points = apply_nodal_displacements(points, all_disps)

    new_mesh = meshio.Mesh(points=np.array(moved_points), cells=[("hexahedron", np.array(cells))])
    if mesh_out.exists():
        mesh_out.unlink()
    new_mesh.write(str(mesh_out), file_format="gmsh")


if __name__ == "__main__":
    cube_gen(Path("private/big_cube.msh"))
    randomize_nodes(Path("private/big_cube.msh"), Path("private/big_cube_rand.msh"))
    # move_modes(Path(r"examples/Square_mesh/square_rot.msh"), "square_rot_bad_3.msh")
