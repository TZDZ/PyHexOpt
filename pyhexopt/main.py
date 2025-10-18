import os
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import meshio
import numpy as np

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_disp_from_mask, expand_displacements, objective, objective_free
from pyhexopt.core.optim import run_opt
from pyhexopt.core.utils import get_boundary_nodes, prepare_dof_masks_and_bases


def main_simple(mesh_in: str | meshio.Mesh, mesh_out: str):
    ### Lecture du maillage
    if isinstance(mesh_in, str):
        msh = meshio.read(mesh_in)
    else:
        msh = mesh_in
    ### préproc
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    boundary = get_boundary_nodes(points, cells)
    disp = jnp.zeros_like(points)  # shape (N,3)
    fixed_indices = jnp.array(boundary)  # e.g., one face is fixed
    free_mask = jnp.ones((points.shape[0],), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]

    objective = partial(
        objective_free,
        points=points,
        cells=cells,
        free_mask=free_mask,
    )

    final_params, final_state = run_opt(
        fun=objective,
        x0=free_disp0,
        method="LBFGS",
        max_iter=100,
        tol=1e-6,
    )

    disp_ = expand_disp_from_mask(final_params, free_mask)
    new_points = apply_nodal_displacements(points, disp_)
    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    if os.path.exists(mesh_out):  # noqa: PTH110
        os.remove(mesh_out)  # noqa: PTH107
    new_mesh.write(mesh_out, file_format="gmsh")


def main(mesh_in: str | meshio.Mesh, mesh_out: str):
    # --- read mesh ---
    if isinstance(mesh_in, str | Path):
        msh = meshio.read(mesh_in)
    else:
        msh = mesh_in

    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    dof = prepare_dof_masks_and_bases(points, cells)

    reduced_disps = jnp.concatenate(
        [
            jnp.zeros((dof.n_volu * 3,)),  # volumic displacements (3 per node)
            jnp.zeros((dof.n_surf * 2,)),  # surface tangential displacements (2 per node)
        ]
    )

    objective_ = partial(
        objective,
        points=points,
        cells=cells,
        n_volu=dof.n_volu,
        n_surf=dof.n_surf,
        n_tot=dof.n_tot,
        volumic_nodes=dof.volumic_nodes,
        surface_nodes=dof.surface_nodes,
        T1=dof.T1,
        T2=dof.T2,
    )

    final_params, final_state = run_opt(
        fun=objective_,
        x0=reduced_disps,
        method="LBFGS",
        max_iter=100,
        tol=1e-6,
    )

    volu_disps = final_params[: dof.n_volu * 3].reshape((dof.n_volu, 3))
    surf_disps = final_params[dof.n_volu * 3 :].reshape((dof.n_surf, 2))

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
    if os.path.exists(mesh_out):
        os.remove(mesh_out)
    new_mesh.write(mesh_out, file_format="gmsh")


if __name__ == "__main__":
    main_simple(r"private\bad_mesh_simple.msh", r"private/test_simple_mesh_out.msh")
