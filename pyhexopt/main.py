import os
from functools import partial

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_disp_from_mask, objective_free, objective_mixed_dof
from pyhexopt.core.optim import run_opt
from pyhexopt.core.utils import get_boundary_nodes, get_edge_nodes, prepare_dof_masks_and_bases


def main(mesh_in: str, mesh_out: str):
    ### Lecture du maillage
    msh = meshio.read(mesh_in)
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


# def main_mixed(mesh_in: str, mesh_out: str):
#     boundary = get_boundary_nodes(mesh_in)
#     points, cells = extract_points_and_cells(mesh_in, dtype=jnp.float32)

#     # Step 1: define which boundary nodes are *fixed* (example: bottom face)
#     fixed_nodes = get_edge_nodes(mesh_in)

#     # Step 2: classify
#     free_nodes, surface_nodes, fixed_nodes, T1, T2 = prepare_dof_masks_and_bases(mesh_in, points, boundary, fixed_nodes)

#     # Step 3: build masks for indexing
#     N = points.shape[0]
#     is_free = np.zeros(N, dtype=bool)
#     is_free[free_nodes] = True

#     is_surface = np.zeros(N, dtype=bool)
#     is_surface[surface_nodes] = True

#     # Step 4: initialize displacements
#     disp0 = jnp.zeros_like(points)

#     # Step 5: optimizer parameter vector
#     # Concatenate all DOFs: [free_nodes(3D) + surface_nodes(2D)]
#     free_disp0 = jnp.concatenate(
#         [
#             jnp.zeros((len(free_nodes), 3)),
#             jnp.zeros((len(surface_nodes), 2)),
#         ],
#         axis=0,
#     )

#     objective = partial(
#         objective_mixed_dof,
#         points=points,
#         cells=cells,
#         free_nodes=free_nodes,
#         surface_nodes=surface_nodes,
#         T1=T1,
#         T2=T2,
#     )

#     final_params, final_state = run_opt(
#         fun=objective,
#         x0=free_disp0,
#         method="LBFGS",
#         max_iter=100,
#         tol=1e-6,
#     )

#     disp_ = expand_disp_from_mask(final_params, free_mask)
#     new_points = apply_nodal_displacements(points, disp_)
#     new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
#     if os.path.exists(mesh_out):  # noqa: PTH110
#         os.remove(mesh_out)  # noqa: PTH107
#     new_mesh.write(mesh_out, file_format="gmsh")


if __name__ == "__main__":
    main(r"private\bad_mesh_simple.msh", r"private/test_simple_mesh_out.msh")
