import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells


def make_unit_hex_mesh():
    """Create a single hexahedron [0,1]^3."""
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=int))]
    mesh = meshio.Mesh(points=points, cells=cells)
    return mesh, points, cells[0][1]


def test_extract_points_and_cells_shapes():
    mesh, ref_points, ref_cells = make_unit_hex_mesh()

    points, cells = extract_points_and_cells(mesh, dtype=jnp.float32, verbose=False)

    # check types
    assert isinstance(points, jnp.ndarray)
    assert isinstance(cells, jnp.ndarray)

    # shape correctness
    assert points.shape == (8, 3)
    assert cells.shape == (1, 8)

    # dtype correctness
    assert points.dtype == jnp.float32
    assert cells.dtype == jnp.int32


def test_extract_points_and_cells_values():
    mesh, ref_points, ref_cells = make_unit_hex_mesh()

    points, cells = extract_points_and_cells(mesh, dtype=jnp.float32, verbose=False)

    np.testing.assert_allclose(np.array(points), ref_points.astype(np.float32))
    np.testing.assert_array_equal(np.array(cells), ref_cells.astype(np.int32))


def test_extract_points_and_cells_no_hex_found():
    """Should raise ValueError when no hex cells present."""
    points = np.random.rand(4, 3)
    cells = [("triangle", np.array([[0, 1, 2]], dtype=int))]
    mesh = meshio.Mesh(points=points, cells=cells)

    with pytest.raises(ValueError):
        extract_points_and_cells(mesh, verbose=False)


if __name__ == "__main__":
    pytest.main([__file__])
