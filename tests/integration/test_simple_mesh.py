import os
from pathlib import Path

import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import apply_nodal_displacements
from pyhexopt.core.obj import expand_disp_from_mask, objective_free
from pyhexopt.core.utils import get_boundary_nodes
from pyhexopt.main import main


def test_real_mesh_masked_grad():
    ### data du test
    move = np.array([0.1, 0.2, 0.3])
    moving_node = 59  # indexings starts at 0!

    ### Lecture du maillage
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
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
    if os.path.exists("tests/integration/test_simple_mesh_out.msh"):  # noqa: PTH110
        os.remove("tests/integration/test_simple_mesh_out.msh")  # noqa: PTH107
    new_mesh.write("tests/integration/test_simple_mesh_out.msh", file_format="gmsh")
    msh = meshio.read(r"tests/integration/test_simple_mesh_out.msh")
    np.testing.assert_allclose(np.array(msh.points[moving_node]), move + initpos)


def test_end_to_end():
    main(Path(r"tests/integration/bad_mesh_simple.msh"), Path(r"tests\integration\corrected_simple_mesh.msh"))
    # good_mesh = meshio.read(Path(r"tests/integration/test_simple_mesh_out.msh"))
    # corrected_msh = meshio.read(Path(r"tests\integration\corrected_simple_mesh.msh"))
    # np.testing.assert_allclose(good_mesh.points, corrected_msh.points)


if __name__ == "__main__":
    pytest.main([__file__])
