import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.jaxobian import GAUSS_POINTS, compute_scaled_jacobians_from_coords
from pyhexopt.core.move import apply_nodal_displacements, nodes_from_points
from pyhexopt.core.neighbor import get_element_neighborhood
from pyhexopt.core.obj import expand_disp_from_mask, expand_displacements, objective, objective_simple
from pyhexopt.core.optim import OptiParams, run_opt
from pyhexopt.core.utils import get_boundary_nodes, prepare_dof_masks_and_bases


def main_simple(mesh_in: str | meshio.Mesh, mesh_out: str, metaparams: OptiParams | None = None):
    ### Lecture du maillage
    if isinstance(mesh_in, str):
        msh = meshio.read(mesh_in)
    else:
        msh = mesh_in
    ### préproc
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    boundary = get_boundary_nodes(cells)
    disp = jnp.zeros_like(points)  # shape (N,3)
    fixed_indices = jnp.array(boundary)  # e.g., one face is fixed
    free_mask = jnp.ones((points.shape[0],), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]

    objective = partial(
        objective_simple,
        points=points,
        cells=cells,
        free_mask=free_mask,
    )

    final_params, final_state = run_opt(fun=objective, x0=free_disp0, metaparams=metaparams)

    disp_ = expand_disp_from_mask(final_params, free_mask)
    new_points = apply_nodal_displacements(points, disp_)
    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    if os.path.exists(mesh_out):  # noqa: PTH110
        os.remove(mesh_out)  # noqa: PTH107
    new_mesh.write(mesh_out, file_format="gmsh")


def pure_main(points: jax.Array, cells: jax.Array, metaparams: OptiParams) -> jax.Array:
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
        alpha=metaparams.alpha,
    )

    final_params, final_state = run_opt(fun=objective_, x0=reduced_disps, metaparams=metaparams)

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
    return moved_points


def main(mesh_in: str | meshio.Mesh, mesh_out: str | Path, metaparams: OptiParams | None = None):
    # --- read mesh ---
    if isinstance(mesh_in, str | Path):
        msh = meshio.read(mesh_in)
    else:
        msh = mesh_in
    if isinstance(mesh_out, str):
        mesh_out = Path(mesh_out)

    if metaparams is None:
        metaparams = OptiParams()

    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)

    moved_points = pure_main(points, cells, metaparams)

    new_mesh = meshio.Mesh(points=np.array(moved_points), cells=[("hexahedron", np.array(cells))])
    if mesh_out.exists():
        mesh_out.unlink()
    new_mesh.write(str(mesh_out), file_format="gmsh")


def recursive_opt(points, cells, node_elem_index_data, metaparams: OptiParams):
    """
    Optimize local neighborhood of `elem_index` and update global points.
    JAX-compatible: no Python dicts or int conversions.
    """
    for _ in range(20):
        node_coords = nodes_from_points(points, cells)
        jac = compute_scaled_jacobians_from_coords(node_coords, at_center=False, sample_points=GAUSS_POINTS)
        jac_min_per_elem = jnp.min(jac, axis=1)

        elem_index = jnp.argmin(jac_min_per_elem)
        neigh_nodes, neigh_elems = get_element_neighborhood(cells, node_elem_index_data, elem_index, metaparams.stencil)
        sub_cells = cells[neigh_elems]
        remap = jnp.searchsorted(neigh_nodes, sub_cells)

        sub_points = points[neigh_nodes]
        moved_sub_points = pure_main(sub_points, remap, metaparams=metaparams)

        points = points.at[neigh_nodes].set(moved_sub_points)

    return points


if __name__ == "__main__":
    main_simple(r"private\bad_mesh_simple.msh", r"private/test_simple_mesh_out.msh")
