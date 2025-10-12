import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.integrated import compute_scaled_jacobians
from pyhexopt.core.jaxobian import GAUSS_POINTS
from pyhexopt.core.move import apply_nodal_displacements, nodes_from_points


def make_hex_mesh(points):
    """Helper: create a single-element hexahedral meshio.Mesh."""
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]
    return meshio.Mesh(points=points, cells=cells)


def test_scaled_jacobian_unit_cube():
    """Unit cube [0,1]^3 should give SJ = 1.0 at center."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=True)
    assert SJ.shape == (1,)
    np.testing.assert_allclose(SJ[0], 1.0, atol=1e-6)


def test_scaled_jacobian_stretched_x():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1.5, 0.8], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=True)
    assert SJ.shape == (1,)
    # The diagonal J would be [1,0.5,0.5], SJ = det(J)/(||Jx||*||Jy||*||Jz||) = 0.5
    assert SJ[0] < 1


def test_scaled_jacobian_inverted_element():
    """A geometrically inverted cube (top and bottom faces swapped) should produce negative SJ."""
    # Original unit cube
    points = np.array(
        [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ],
        dtype=float,
    )

    # Invert along z: swap top and bottom faces
    inverted_points = points.copy()
    inverted_points[[0, 1, 2, 3, 4, 5, 6, 7]] = points[[4, 5, 6, 7, 0, 1, 2, 3]]

    mesh = make_hex_mesh(inverted_points)
    SJ = compute_scaled_jacobians(mesh, at_center=True)

    assert SJ.shape == (1,)
    assert SJ[0] < 0  # inverted element has negative scaled Jacobian


def test_scaled_jacobian_unit_cube():
    """Unit cube [0,1]^3 should give SJ = 1 at all corners."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS)
    assert SJ.shape == (1, 8)
    np.testing.assert_allclose(SJ[0], 1.0, atol=1e-6)


def test_scaled_jacobian_stretched_x():
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS)
    assert SJ.shape == (1, 8)
    np.testing.assert_allclose(SJ[0], 1.0, atol=1e-6)


def test_scaled_jacobian_inverted_element():
    """Swap two nodes to invert element: SJ negative somewhere."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    # swap nodes 6 and 7 to invert
    points[[6, 7]] = points[[7, 6]]
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS)
    assert SJ.shape == (1, 8)
    assert np.any(SJ[0] < 0.0)


def test_scaled_jacobian_concave_element():
    """Concave hex: SJ < 1 at some corners."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0.5, 0.5, 0.8],
            [0, 1, 1],  # move node 6 inward to create concavity
        ],
        dtype=float,
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS)
    assert SJ.shape == (1, 8)
    assert np.any(SJ[0] < 1.0)
    assert np.any(SJ[0] < 0.0)  # still not inverted


def test_real_mesh():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    jac = compute_scaled_jacobians(msh)
    assert len(jac) == 27
    assert isinstance(jac, jnp.ndarray)


def test_real_mesh_gauss():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    jac = compute_scaled_jacobians(msh, at_center=False, sample_points=GAUSS_POINTS)
    assert jac.shape == (27, 8)
    assert isinstance(jac, jnp.ndarray)


def test_move_mode():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32, verbose=False)
    disp = jnp.zeros_like(points)  # shape (N,3)

    # Move node 27 by [0.1, 0.2, 0.3]
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    new_points = apply_nodal_displacements(points, disp)

    for a, b in zip(new_points[27], [0.7666667, 0.2, 1.3], strict=False):
        np.testing.assert_approx_equal(a, b)

    node_coords = nodes_from_points(new_points, cells)  # shape (E, 8, 3)

    # Check the effect on elements that include node 27
    # Find which elements contain node 27
    elements_with_27 = jnp.any(cells == 27, axis=1)  # shape (E,), bool array
    for a, b in zip(node_coords[elements_with_27][0][5], [0.7666667, 0.2, 1.3], strict=False):
        np.testing.assert_approx_equal(a, b)
    for a, b in zip(node_coords[elements_with_27][1][1], [0.7666667, 0.2, 1.3], strict=False):
        np.testing.assert_approx_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__])
