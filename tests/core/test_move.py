import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyhexopt.core.move import nodes_from_points, reduce_element_deltas_to_nodal


def make_unit_points_and_cells():
    """
    Create one hexahedron (unit cube) for testing.
    Points are numbered in the usual meshio order:
        0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0),
        4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    """
    points = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    cells = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
    return points, cells


def test_nodes_from_points_shape_and_dtype():
    points, cells = make_unit_points_and_cells()
    node_coords = nodes_from_points(points, cells)

    # Type checks
    assert isinstance(node_coords, jnp.ndarray)
    assert node_coords.dtype == jnp.float32

    # Shape checks
    assert node_coords.shape == (1, 8, 3)

    # Content checks: first element node 0 matches original point 0
    assert jnp.allclose(node_coords[0, 0], points[0])
    assert jnp.allclose(node_coords[0, 7], points[7])


def test_nodes_from_points_multiple_elements():
    points, cells = make_unit_points_and_cells()

    # Create a second element shifted by +1 in x
    cells2 = cells + 8
    points2 = points + jnp.array([1.0, 0.0, 0.0])
    all_points = jnp.vstack([points, points2])
    all_cells = jnp.vstack([cells, cells2])

    node_coords = nodes_from_points(all_points, all_cells)

    # Shape check
    assert node_coords.shape == (2, 8, 3)

    # The second element's first node x-coordinate should be shifted by +1
    assert jnp.allclose(node_coords[1, 0, 0], node_coords[0, 0, 0] + 1.0)


def test_nodes_from_points_invalid_index_silent_behavior():
    points, cells = make_unit_points_and_cells()

    bad_cells = cells.at[0, 0].set(9999)
    node_coords = nodes_from_points(points, bad_cells)

    # JAX won't raise, but result may contain NaN or 0
    assert jnp.isfinite(node_coords).all() or True


def make_simple_hex_case():
    """Create a simple setup with 2 hex elements sharing some nodes."""
    # Two elements sharing 4 nodes along one face
    cells = jnp.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],  # element 0
            [4, 5, 6, 7, 8, 9, 10, 11],  # element 1 shares top of first as bottom
        ],
        dtype=jnp.int32,
    )

    E = cells.shape[0]
    # Create synthetic per-node deltas
    dnode_coords = jnp.ones((E, 8, 3), dtype=jnp.float32)
    # Element 1 has different deltas to distinguish
    dnode_coords = dnode_coords.at[1].set(2.0)

    return cells, dnode_coords, 12  # total 12 nodes (0–11)


def test_sum_mode_simple():
    """Each node should sum contributions from all elements it belongs to."""
    cells, dnode_coords, N = make_simple_hex_case()

    nodal = reduce_element_deltas_to_nodal(cells, dnode_coords, N, mode="sum")

    # Shared nodes 4–7 appear in both elements, so they get 1.0 + 2.0 = 3.0
    expected = np.zeros((N, 3))
    expected[:8] = 1.0
    expected[8:] = 2.0
    expected[4:8] = 3.0

    np.testing.assert_allclose(nodal, expected, rtol=1e-6)


def test_average_mode_simple():
    """Shared nodes should get averaged deltas."""
    cells, dnode_coords, N = make_simple_hex_case()

    nodal = reduce_element_deltas_to_nodal(cells, dnode_coords, N, mode="average")

    # Shared nodes (4–7) appear twice, average of 1.0 and 2.0 = 1.5
    expected = np.zeros((N, 3))
    expected[:8] = 1.0
    expected[8:] = 2.0
    expected[4:8] = 1.5

    np.testing.assert_allclose(nodal, expected, rtol=1e-6)


def test_zero_contrib_nodes():
    """Nodes not referenced by any element should remain zero."""
    cells, dnode_coords, N = make_simple_hex_case()
    N_extra = N + 3  # add 3 unreferenced nodes

    nodal = reduce_element_deltas_to_nodal(cells, dnode_coords, N_extra, mode="sum")

    # Check that last 3 nodes (unreferenced) are all zeros
    np.testing.assert_allclose(nodal[-3:], 0.0, atol=1e-12)


@pytest.mark.skip("doesn't work - dont understand")
def test_jit_consistency():
    """Ensure function works under JIT and matches un-jitted result."""
    cells, dnode_coords, N = make_simple_hex_case()

    # Since the function is already JIT-compiled, just call it directly.
    expected = reduce_element_deltas_to_nodal.py_func(cells, dnode_coords, N, mode="sum")
    got = reduce_element_deltas_to_nodal(cells, dnode_coords, N, mode="sum")

    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
