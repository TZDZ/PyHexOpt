import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.obj import expand_disp_from_mask, objective, objective_free


def test_real_mesh():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    fixed_mask = jnp.zeros((N,), dtype=bool).at[fixed_indices].set(True)
    obj = objective(disp, points, cells, fixed_mask)
    assert obj == 0


def test_real_mesh_not_optimal():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    free_mask = ~jnp.zeros((N,), dtype=bool).at[fixed_indices].set(True)
    obj = objective(disp, points, cells, free_mask)
    assert obj > 0


def test_real_mesh_masked():
    msh = meshio.read(r"examples/Square_mesh/quare.msh")
    points, cells = extract_points_and_cells(msh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    free_mask = jnp.ones((N,), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]
    obj = objective_free(free_disp0, points, cells, free_mask)
    assert obj > 0


def make_case():
    """Utility: creates a small fake displacement and mask."""
    N = 6
    # free: 0, 2, 5
    free_mask = jnp.array([True, False, True, False, False, True])
    free_disp = jnp.array(
        [
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
        ]
    )
    return free_disp, free_mask, N


def test_basic_reconstruction():
    free_disp, free_mask, N = make_case()
    full = expand_disp_from_mask(free_disp, free_mask)

    # Expect displacement only where mask == True
    expected = jnp.zeros((N, 3))
    expected = expected.at[jnp.where(free_mask)[0]].set(free_disp)

    np.testing.assert_allclose(np.array(full), np.array(expected), rtol=1e-7, atol=1e-7)


def test_jit_consistency():
    """Ensure JIT version matches non-JIT."""
    free_disp, free_mask, _ = make_case()
    fn_jit = jax.jit(expand_disp_from_mask)

    expected = expand_disp_from_mask(free_disp, free_mask)
    got = fn_jit(free_disp, free_mask)

    np.testing.assert_allclose(np.array(got), np.array(expected), rtol=1e-7, atol=1e-7)


def test_all_free_nodes():
    """If all nodes are free, output should equal free_disp."""
    N = 4
    free_mask = jnp.ones((N,), dtype=bool)
    free_disp = jnp.arange(N * 3, dtype=jnp.float32).reshape((N, 3))
    full = expand_disp_from_mask(free_disp, free_mask)

    np.testing.assert_allclose(np.array(full), np.array(free_disp))


def test_no_free_nodes():
    """If no nodes are free, full displacement should be zero."""
    N = 5
    free_mask = jnp.zeros((N,), dtype=bool)
    free_disp = jnp.zeros((0, 3))
    full = expand_disp_from_mask(free_disp, free_mask)

    np.testing.assert_allclose(np.array(full), np.zeros((N, 3)))


def test_partial_free_nodes():
    """Check that indices match for arbitrary mask."""
    N = 8
    rng = np.random.default_rng(0)
    mask_np = rng.choice([True, False], size=N)
    free_mask = jnp.array(mask_np)
    free_indices = jnp.where(free_mask)[0]

    free_disp = jnp.arange(len(free_indices) * 3).reshape((-1, 3)).astype(jnp.float32)
    full = expand_disp_from_mask(free_disp, free_mask)

    # full should have exactly free_disp rows inserted at free_indices
    assert jnp.allclose(full[free_indices], free_disp)
    assert jnp.allclose(full[~free_mask], 0.0)


def test_expand_disp_from_mask_node_position():
    N = 64
    points = jnp.zeros((N, 3), dtype=jnp.float32)

    disp = jnp.zeros_like(points)
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))

    fixed_indices = jnp.array([0, 3, 4, 7])
    free_mask = jnp.ones((N,), dtype=bool).at[fixed_indices].set(False)

    free_disp0 = disp[free_mask]
    full_disp = expand_disp_from_mask(free_disp0, free_mask)

    assert full_disp.shape == (N, 3)
    np.testing.assert_allclose(np.array(full_disp[fixed_indices]), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.array(full_disp[27]), np.array([0.1, 0.2, 0.3]), atol=1e-8)
    free_indices = np.where(np.array(free_mask))[0]
    others = np.setdiff1d(free_indices, [27])
    np.testing.assert_allclose(np.array(full_disp[others]), 0.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
