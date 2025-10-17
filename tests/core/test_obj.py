import jax
import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.adapters.meshio_ import extract_points_and_cells
from pyhexopt.core.move import uv_to_disp_full
from pyhexopt.core.obj import expand_disp_from_mask, expand_displacements, objective_free, objective_simple
from pyhexopt.core.utils import build_tangent_bases


def test_real_mesh(clean_square_mesh):
    points, cells = extract_points_and_cells(clean_square_mesh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    fixed_mask = jnp.zeros((N,), dtype=bool).at[fixed_indices].set(True)
    obj = objective_simple(disp, points, cells, fixed_mask)
    assert obj == 0


def test_real_mesh_not_optimal(clean_square_mesh):
    points, cells = extract_points_and_cells(clean_square_mesh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    free_mask = ~jnp.zeros((N,), dtype=bool).at[fixed_indices].set(True)
    obj = objective_simple(disp, points, cells, free_mask)
    assert obj > 0


def test_real_mesh_masked(clean_square_mesh):
    points, cells = extract_points_and_cells(clean_square_mesh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    free_mask = jnp.ones((N,), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]
    obj = objective_free(free_disp0, points, cells, free_mask)
    assert obj > 0


def test_real_mesh_masked_grad(clean_square_mesh):
    points, cells = extract_points_and_cells(clean_square_mesh, dtype=jnp.float32)
    disp = jnp.zeros_like(points)  # shape (N,3)
    disp = disp.at[27].set(jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32))
    N = points.shape[0]
    fixed_indices = jnp.array([0, 3, 4, 7])  # e.g., one face is fixed
    free_mask = jnp.ones((N,), dtype=bool).at[fixed_indices].set(False)
    free_disp0 = disp[free_mask]
    grad = jax.grad(objective_free)(free_disp0, points, cells, free_mask)
    assert grad.shape == free_disp0.shape


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


def test_expand_displacements_places_values_correctly():
    """
    Create a toy set of nodes:
      - free_nodes = [0,1] (3D displacements)
      - surface_nodes = [2,3] (2D uv displacements)
      - fixed node = 4
    Verify the expanded displacement has values at the right indices.
    """
    N = 5
    free_nodes = np.array([0, 1], dtype=int)
    surface_nodes = np.array([2, 3], dtype=int)

    # trivial normals (z-up) so tangents are x and y
    normals = np.zeros((N, 3), dtype=float)
    normals[:, 2] = 1.0  # all normals pointing up
    # build tangent bases for only surface nodes
    T1, T2 = build_tangent_bases(np.zeros((N, 3)), normals, surface_nodes)

    free_disp_3d = np.array([[0.5, 0.0, 0.0], [0.0, -0.3, 0.0]], dtype=float)  # move node 0 in x, node1 in -y
    surface_disp_uv = np.array([[1.0, 2.0], [-0.5, 0.25]], dtype=float)  # u->x, v->y

    disp_full = expand_displacements(free_disp_3d, surface_disp_uv, free_nodes, surface_nodes, T1, T2, N)

    # Check shapes
    assert disp_full.shape == (N, 3)

    # Free nodes were set
    assert np.allclose(disp_full[0], free_disp_3d[0])
    assert np.allclose(disp_full[1], free_disp_3d[1])

    # Surface nodes correspond to uv -> tangential displacements (orthogonal to normal)
    for i, node_idx in enumerate(surface_nodes):
        disp = disp_full[node_idx]
        n = np.array([0, 0, 1.0])  # z-up
        # Check orthogonality to normal
        assert abs(np.dot(disp, n)) < 1e-8
        # Check magnitude consistency: |disp| = sqrt(u^2 + v^2)
        uv = surface_disp_uv[i]
        assert np.allclose(np.linalg.norm(disp), np.linalg.norm(uv), atol=1e-8)


def test_surface_displacement_is_tangential():
    """
    Ensure that mapped surface displacements have zero component in the normal direction.
    """
    N = 6
    # Create normals that vary so as to test generality
    normals = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],  # degenerate normal (fallback)
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # normalize nonzero rows
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 0
    normals[nonzero] = normals[nonzero] / norms[nonzero]

    # pick surface nodes (some subset)
    surface_nodes = np.array([1, 2, 3], dtype=int)  # test these
    # build tangent bases for them
    T1, T2 = build_tangent_bases(np.zeros((N, 3)), normals, surface_nodes)

    # uv displacements (random)
    rng = np.random.default_rng(0)
    uv_disp = rng.normal(size=(len(surface_nodes), 2))

    # compute disp_full via numpy expand_displacements (free nodes empty)
    disp_full = expand_displacements(np.zeros((0, 3)), uv_disp, np.array([], dtype=int), surface_nodes, T1, T2, N)

    # Verify tangential: dot(disp, normal) ~ 0
    for i_local, v_idx in enumerate(surface_nodes):
        disp = disp_full[v_idx]
        normal = normals[v_idx]
        if np.linalg.norm(normal) < 1e-12:
            # degenerate normal -> we cannot assert orthogonality, but displacement should be finite
            assert np.isfinite(disp).all()
        else:
            dot = np.dot(disp, normal)
            assert abs(dot) < 1e-8, f"Non-zero normal component {dot} for node {v_idx}"


def test_uv_to_disp_full_jax_and_numpy_agree():
    """
    Verify jitted JAX uv_to_disp_full gives the same result as numpy expand_displacements mapping for surface nodes.
    """
    N = 7
    surface_nodes = np.array([2, 4, 5], dtype=int)
    normals = np.zeros((N, 3))
    normals[:, 2] = 1.0  # all z-up
    T1, T2 = build_tangent_bases(np.zeros((N, 3)), normals, surface_nodes)

    uv = np.array([[0.1, 0.2], [-1.0, 0.3], [0.5, -0.2]], dtype=float)
    # numpy mapping
    disp_np = expand_displacements(np.zeros((0, 3)), uv, np.array([], dtype=int), surface_nodes, T1, T2, N)

    # jax mapping: convert to jnp
    disp_jax = uv_to_disp_full(
        jnp.array(uv), jnp.array(T1), jnp.array(T2), jnp.array(surface_nodes, dtype=jnp.int32), N
    )
    # transfer to numpy
    disp_jax_np = np.array(disp_jax)

    assert np.allclose(disp_np, disp_jax_np, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__])
