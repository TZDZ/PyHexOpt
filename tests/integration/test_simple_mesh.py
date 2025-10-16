import os
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_disp_from_mask, expand_displacements, objective_free
from pyhexopt.core.utils import get_boundary_nodes, prepare_dof_masks_and_bases
from pyhexopt.main import main


def test_real_mesh_masked_grad(clean_square_mesh, out_path: Path):
    ### data du test
    move = np.array([0.1, 0.2, 0.3])
    moving_node = 59  # indexings starts at 0!

    ### Lecture du maillage
    msh = clean_square_mesh
    initpos = np.array(msh.points[moving_node])

    ### préproc
    boundary = get_boundary_nodes(msh)
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[moving_node].set(jnp.array(move, dtype=jnp.float32))
    fixed_indices = jnp.array(boundary)  # e.g., one face is fixed
    free_mask = jnp.ones((points.shape[0],), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]

    ### Pour l'optim : objective_free -> sortira free_disp_opt
    assert objective_free(free_disp0, points, cells, free_mask) > 0
    grad = jax.grad(objective_free)(free_disp0, points, cells, free_mask)
    np.testing.assert_allclose(grad[0], 0.0)
    assert np.any(np.abs(grad[1]) > 0.01)
    assert grad.shape[0] == len(points) - len(boundary)

    ###test : est-ce qu'on reconstruit bien le maillage déplacé ?
    disp_ = expand_disp_from_mask(free_disp0, free_mask)
    new_points = apply_nodal_displacements(points, disp_)
    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    out_file = out_path / "test_simple_mesh_out.msh"
    if out_file.exists():
        out_file.unlink()
    new_mesh.write(out_file, file_format="gmsh")
    msh = meshio.read(out_file)
    np.testing.assert_allclose(np.array(msh.points[moving_node]), move + initpos)


def test_end_to_end(clean_square_mesh, square_bad1_mesh_path, out_path):
    main(square_bad1_mesh_path, out_path / "corrected_simple_mesh.msh")
    corrected_msh = meshio.read(out_path / "corrected_simple_mesh.msh")
    np.testing.assert_allclose(clean_square_mesh.points, corrected_msh.points, atol=2e-3)


def test_move_mode_surface(clean_square_mesh, out_path):
    mesh_in = clean_square_mesh
    points, cells = extract_points_and_cells(mesh_in, dtype=jnp.float32)
    free_nodes, surface_nodes, edge_nodes, T1, T2 = prepare_dof_masks_and_bases(mesh_in)

    N = points.shape[0]
    is_free = np.zeros(N, dtype=bool)
    is_free[free_nodes] = True

    is_surface = np.zeros(N, dtype=bool)
    is_surface[surface_nodes] = True

    disp_concat = jnp.concatenate(
        [
            jnp.zeros((len(free_nodes) * 3,)),  # flatten free 3D displacements
            jnp.zeros((len(surface_nodes) * 2,)),  # flatten surface 2D displacements
        ]
    )

    n_free = len(free_nodes)
    disp_concat = disp_concat.at[: n_free * 3 + 10].set(0.25)
    free_disp_3d = disp_concat[: n_free * 3].reshape((n_free, 3))
    surface_disp_uv = disp_concat[n_free * 3 :].reshape((len(surface_nodes), 2))

    disp_full = expand_displacements(free_disp_3d, surface_disp_uv, free_nodes, surface_nodes, T1, T2, points.shape[0])

    new_points = apply_nodal_displacements(points, disp_full)

    new_mesh = meshio.Mesh(points=np.array(new_points), cells=[("hexahedron", np.array(cells))])
    mesh_out = out_path / "surf_move_clean_square_mesh.msh"
    if mesh_out.exists():
        mesh_out.unlink()
    new_mesh.write(mesh_out, file_format="gmsh")


if __name__ == "__main__":
    # test_move_mode_surface()
    pytest.main([__file__])
