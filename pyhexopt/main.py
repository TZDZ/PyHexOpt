import os
from functools import partial

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_disp_from_mask, expand_displacements, objective_free, objective_mixed_dof
from pyhexopt.core.optim import run_opt
from pyhexopt.core.utils import get_boundary_nodes, get_edge_nodes, prepare_dof_masks_and_bases


def main_simple(mesh_in: str | meshio.Mesh, mesh_out: str):
    ### Lecture du maillage
    if isinstance(mesh_in, str):
        msh = meshio.read(mesh_in)
    else:
        msh = mesh_in
    ### préproc
    boundary = get_boundary_nodes(msh)
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
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
    ### Lecture du maillage
    if isinstance(mesh_in, str):
        msh = meshio.read(mesh_in)
    else:
        msh = mesh_in
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    volumic_nodes, surface_nodes, edge_nodes, T1, T2 = prepare_dof_masks_and_bases(msh)

    n_tot = points.shape[0]
    is_free = np.zeros(n_tot, dtype=bool)
    is_free[volumic_nodes] = True

    is_surface = np.zeros(n_tot, dtype=bool)
    is_surface[surface_nodes] = True

    n_volu = len(volumic_nodes)
    n_surf = len(surface_nodes)
    reduced_disps = jnp.concatenate(
        [
            jnp.zeros((n_volu * 3,)),  # flatten free 3D displacements
            jnp.zeros((len(surface_nodes) * 2,)),  # flatten surface 2D displacements
        ]
    )

    reduced_disps = reduced_disps.at[n_volu * 3 + 10].set(0.25)

    volu_disps = reduced_disps[: n_volu * 3].reshape((n_volu, 3))
    surf_disps = reduced_disps[n_volu * 3 :].reshape((n_surf, 2))

    all_disps = expand_displacements(volu_disps, surf_disps, volumic_nodes, surface_nodes, T1, T2, n_tot)

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


if __name__ == "__main__":
    main_simple(r"private\bad_mesh_simple.msh", r"private/test_simple_mesh_out.msh")
