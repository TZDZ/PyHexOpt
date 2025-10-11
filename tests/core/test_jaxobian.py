import jax.numpy as jnp
import meshio
import numpy as np
import pytest

from pyhexopt.core.jaxobian import (
    GAUSS_POINTS,
    REF_SIGNS,
    _compute_center_jacobians,
    _compute_jacobians_at_points,
    compute_jacobians,
    compute_scaled_jacobians,
    det3x3,
    dN_trilinear_at_samples,
)


def test_det3x3_single_identity():
    """Determinant of the identity matrix should be 1."""
    identity = jnp.eye(3)
    det = det3x3(identity)
    assert np.isclose(float(det), 1.0, atol=1e-8)


def test_det3x3_batch_known_values():
    """Check batched determinants against numpy.linalg.det."""
    mats = jnp.stack(
        [
            jnp.eye(3),
            jnp.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]),
            jnp.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]),
        ]
    )
    det_ref = np.linalg.det(np.array(mats))
    det_jax = np.array(det3x3(mats))
    assert np.allclose(det_jax, det_ref, atol=1e-8)


def test_det3x3_negative_and_zero():
    """Test negative and zero determinant behavior."""
    mats = jnp.array(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, -1]],  # det = -1
            [[1, 2, 3], [2, 4, 6], [1, 0, 1]],  # det = 0 (rows dependent)
        ],
        dtype=jnp.float32,
    )
    det_jax = np.array(det3x3(mats))
    assert np.isclose(det_jax[0], -1.0, atol=1e-8)
    assert np.isclose(det_jax[1], 0.0, atol=1e-8)


def test_compute_center_jacobians_unit_cube():
    """Jacobian for a unit cube aligned with axes should be diag(0.5,0.5,0.5)."""
    # reference dN/dxi at center = REF_SIGNS * 1/8
    dN_center = (REF_SIGNS * 0.125).astype(jnp.float32)

    # one unit cube: [0,1]^3 -> [-1,1]^3 scaling = 0.5 along each axis
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ],
        dtype=np.float32,
    )
    nodes_xyz = jnp.expand_dims(coords, axis=0)  # (1,8,3)

    J, detJ = _compute_center_jacobians(nodes_xyz, dN_center)
    J_np = np.array(J[0])
    det_np = float(np.array(detJ[0]))

    # expected Jacobian: diag(0.5, 0.5, 0.5)
    assert np.allclose(J_np, np.diag([0.5, 0.5, 0.5]), atol=1e-8)
    assert np.isclose(det_np, 0.125, atol=1e-8)


def test_compute_center_jacobians_random_batch():
    """Compare _compute_center_jacobians results to explicit numpy calculation for random elements."""
    rng = np.random.default_rng(0)
    E = 5
    nodes_xyz = jnp.array(rng.random((E, 8, 3)), dtype=jnp.float32)
    dN_center = (REF_SIGNS * 0.125).astype(jnp.float32)

    J_jax, det_jax = _compute_center_jacobians(nodes_xyz, dN_center)
    J_np_ref = np.einsum("eia,ib->eab", np.array(nodes_xyz), np.array(dN_center))
    det_np_ref = np.linalg.det(J_np_ref)

    assert np.allclose(np.array(J_jax), J_np_ref, atol=1e-6)
    assert np.allclose(np.array(det_jax), det_np_ref, atol=1e-6)


def test_jacobians_at_points_unit_cube():
    """
    For a unit cube [0,1]^3 arranged in MeshIO/VTK node order,
    the mapping from reference [-1,1]^3 -> [0,1]^3 is linear with
    constant Jacobian diag(0.5,0.5,0.5) and determinant 0.125 for all sample points.
    """
    # unit cube nodes (MeshIO / VTK ordering)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [1.0, 1.0, 0.0],  # 2
            [0.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [1.0, 1.0, 1.0],  # 6
            [0.0, 1.0, 1.0],  # 7
        ],
        dtype=np.float32,
    )

    # pack into (E,8,3) with E=1
    nodes_xyz = jnp.expand_dims(jnp.array(coords, dtype=jnp.float32), axis=0)  # (1,8,3)

    # choose a set of sample points (xi,eta,zeta) including center and some corners / midpoints
    sample_points = np.array(
        [
            [0.0, 0.0, 0.0],  # center
            [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],  # gauss point
            [-0.5, 0.3, -0.2],  # arbitrary interior point
        ],
        dtype=np.float32,
    )  # shape (Q,3)

    dN_q = dN_trilinear_at_samples(sample_points, dtype=jnp.float32)  # (Q,8,3)

    # compute
    J, detJ = _compute_jacobians_at_points(nodes_xyz, dN_q)

    J_np = np.array(J)  # shape (1, Q, 3, 3)
    det_np = np.array(detJ)  # shape (1, Q)

    # Expected Jacobian for unit cube is diag(0.5,0.5,0.5) at all points
    expected_J = np.tile(np.diag([0.5, 0.5, 0.5])[None, None, :, :], (1, sample_points.shape[0], 1, 1))

    assert J_np.shape == (1, sample_points.shape[0], 3, 3)
    assert det_np.shape == (1, sample_points.shape[0])

    assert np.allclose(J_np, expected_J, atol=1e-7)
    assert np.allclose(det_np, 0.125, atol=1e-7)


def test_jacobians_at_points_random_batch_against_numpy():
    """
    Random test: compare _compute_jacobians_at_points to an explicit numpy einsum
    reference for multiple elements and multiple sample points.
    """
    rng = np.random.default_rng(42)
    E = 6  # number of elements
    Q = 5  # number of sample points

    # Random node coordinates for each element (E,8,3)
    nodes = rng.random((E, 8, 3)).astype(np.float32)

    # Random sample points inside reference element [-1,1]^3
    sample_pts = (rng.random((Q, 3)).astype(np.float32) * 2.0) - 1.0

    # compute dN for these sample points using the module helper
    dN_q = dN_trilinear_at_samples(sample_pts, dtype=jnp.float32)  # (Q,8,3)

    # run jax function
    J_jax, det_jax = _compute_jacobians_at_points(jnp.array(nodes), dN_q)

    # numpy reference: J[e,q,a,b] = sum_i nodes[e,i,a] * dN_q[q,i,b]
    J_ref = np.einsum("eia,qib->eqab", nodes, np.array(dN_q))
    det_ref = np.linalg.det(J_ref)

    assert np.array(J_jax).shape == J_ref.shape
    assert np.array(det_jax).shape == det_ref.shape

    assert np.allclose(np.array(J_jax), J_ref, atol=1e-6)
    assert np.allclose(np.array(det_jax), det_ref, atol=1e-6)


def test_jacobians_at_points_shape_and_broadcasting():
    """
    Validate that shapes are correct and that the function works when Q=1 and E>1.
    """
    rng = np.random.default_rng(123)
    E = 3
    Q = 1

    nodes = rng.random((E, 8, 3)).astype(np.float32)
    sample_pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)  # single center point

    dN_q = dN_trilinear_at_samples(sample_pts, dtype=jnp.float32)  # (1,8,3)

    J_jax, det_jax = _compute_jacobians_at_points(jnp.array(nodes), dN_q)

    assert np.array(J_jax).shape == (E, Q, 3, 3)
    assert np.array(det_jax).shape == (E, Q)

    # For center point dN simplifies to REF_SIGNS/8; verify with explicit einsum for each element
    J_ref = np.einsum("eia,ib->eab", nodes, np.array(REF_SIGNS * 0.125, dtype=np.float32))
    det_ref = np.linalg.det(J_ref)
    assert np.allclose(np.array(J_jax)[:, 0, :, :], J_ref, atol=1e-6)
    assert np.allclose(np.array(det_jax)[:, 0], det_ref, atol=1e-6)


def test_center_derivatives_match_reference_signs():
    """
    At the element center (ξ=η=ζ=0), the derivative of each trilinear shape function
    should be REF_SIGNS / 8.
    """
    xi_eta_zeta = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    dN = np.array(dN_trilinear_at_samples(xi_eta_zeta))
    expected = np.array(REF_SIGNS * 0.125, dtype=np.float32)  # (8,3)
    assert dN.shape == (1, 8, 3)
    assert np.allclose(dN[0], expected, atol=1e-8)


def test_symmetry_properties():
    """
    Check that flipping a coordinate sign in the node reference position
    flips the corresponding derivative component, but not necessarily leaves others equal.
    """
    xi_eta_zeta = np.array([[0.2, -0.3, 0.1]], dtype=np.float32)
    dN = np.array(dN_trilinear_at_samples(xi_eta_zeta))[0]  # (8,3)

    # Mirror pairs along each local axis (VTK order)
    mirror_x = [(0, 1), (3, 2), (4, 5), (7, 6)]
    mirror_y = [(0, 3), (1, 2), (4, 7), (5, 6)]
    mirror_z = [(0, 4), (1, 5), (2, 6), (3, 7)]

    for a, b in mirror_x:
        assert np.allclose(dN[a, 0], -dN[b, 0], atol=1e-8)

    for a, b in mirror_y:
        assert np.allclose(dN[a, 1], -dN[b, 1], atol=1e-8)

    for a, b in mirror_z:
        assert np.allclose(dN[a, 2], -dN[b, 2], atol=1e-8)


def test_sum_of_shape_function_derivatives_is_zero():
    """
    The sum of dN/dξ, dN/dη, dN/dζ across all 8 nodes should be zero at any point
    because the shape functions form a partition of unity.
    """
    rng = np.random.default_rng(0)
    Q = 10
    xi_eta_zeta = rng.uniform(-1, 1, (Q, 3)).astype(np.float32)
    dN = np.array(dN_trilinear_at_samples(xi_eta_zeta))  # (Q,8,3)
    sums = dN.sum(axis=1)  # (Q,3)
    assert np.allclose(sums, 0.0, atol=1e-7)


def test_finite_difference_verification():
    """
    Verify correctness by finite-difference approximation of shape function derivatives.
    Uses a relaxed relative tolerance suitable for float32 precision.
    """
    rng = np.random.default_rng(123)
    pts = rng.uniform(-0.8, 0.8, (3, 3)).astype(np.float32)
    eps = 3e-4  # slightly larger for stable finite difference in float32
    r = np.array(REF_SIGNS, dtype=np.float32)  # (8,3)

    for xi, eta, zeta in pts:
        xi_eta_zeta = np.array([[xi, eta, zeta]], dtype=np.float32)
        dN_analytic = np.array(dN_trilinear_at_samples(xi_eta_zeta))[0]  # (8,3)

        def N_values(xi_, eta_, zeta_):
            return 0.125 * (1 + r[:, 0] * xi_) * (1 + r[:, 1] * eta_) * (1 + r[:, 2] * zeta_)

        N_xi_plus = N_values(xi + eps, eta, zeta)
        N_xi_minus = N_values(xi - eps, eta, zeta)
        N_eta_plus = N_values(xi, eta + eps, zeta)
        N_eta_minus = N_values(xi, eta - eps, zeta)
        N_zeta_plus = N_values(xi, eta, zeta + eps)
        N_zeta_minus = N_values(xi, eta, zeta - eps)

        dN_fd = np.column_stack(
            [
                (N_xi_plus - N_xi_minus) / (2 * eps),
                (N_eta_plus - N_eta_minus) / (2 * eps),
                (N_zeta_plus - N_zeta_minus) / (2 * eps),
            ]
        )

        # Use both absolute and relative tolerance
        assert np.allclose(dN_analytic, dN_fd, rtol=2e-3, atol=2e-4)


def test_multiple_points_shape_and_dtype():
    xi_eta_zeta = np.array([[0.0, 0.0, 0.0], [0.5, -0.5, 0.3], [-0.8, 0.9, -0.1]], dtype=np.float64)
    dN = dN_trilinear_at_samples(xi_eta_zeta, dtype=jnp.float64)
    assert dN.shape == (3, 8, 3)
    # if x64 not enabled, JAX silently downcasts
    assert dN.dtype in (jnp.float64, jnp.float32)


def make_hex_mesh(points):
    """Helper: create a single-element hexahedral meshio.Mesh."""
    cells = [("hexahedron", np.array([[0, 1, 2, 3, 4, 5, 6, 7]]))]
    return meshio.Mesh(points=points, cells=cells)


def test_compute_jacobians_unit_cube():
    """
    For a unit cube [0,1]^3, reference domain is [-1,1]^3,
    so J = diag(0.5,0.5,0.5), detJ = 0.125.
    """
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

    mesh = make_hex_mesh(points)
    J, detJ = compute_jacobians(mesh, at_center=True, verbose=False)

    expected_J = np.diag([0.5, 0.5, 0.5])
    expected_detJ = 0.125

    np.testing.assert_allclose(J[0], expected_J, atol=1e-6)
    np.testing.assert_allclose(detJ[0], expected_detJ, atol=1e-6)


def test_compute_jacobians_scaled_x():
    """
    Stretch cube along x by factor 2.
    Mapping from [-1,1]^3 → [0,2]×[0,1]×[0,1],
    so J = diag(1.0,0.5,0.5), detJ = 0.25.
    """
    points = np.array(
        [
            [0, 0, 0],
            [2, 0, 0],
            [2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [2, 0, 1],
            [2, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )

    mesh = make_hex_mesh(points)
    J, detJ = compute_jacobians(mesh, at_center=True, verbose=False)

    expected_J = np.diag([1.0, 0.5, 0.5])
    expected_detJ = 0.25

    np.testing.assert_allclose(J[0], expected_J, atol=1e-6)
    np.testing.assert_allclose(detJ[0], expected_detJ, atol=1e-6)


def test_scaled_jacobian_unit_cube():
    """Unit cube [0,1]^3 should give SJ = 1.0 at center."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=True, verbose=False)
    assert SJ.shape == (1,)
    np.testing.assert_allclose(SJ[0], 1.0, atol=1e-6)


def test_scaled_jacobian_stretched_x():
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1.5, 0.8], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=True, verbose=False)
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
    SJ = compute_scaled_jacobians(mesh, at_center=True, verbose=False)

    assert SJ.shape == (1,)
    assert SJ[0] < 0  # inverted element has negative scaled Jacobian


def test_scaled_jacobian_unit_cube():
    """Unit cube [0,1]^3 should give SJ = 1 at all corners."""
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS, verbose=False)
    assert SJ.shape == (1, 8)
    np.testing.assert_allclose(SJ[0], 1.0, atol=1e-6)


def test_scaled_jacobian_stretched_x():
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [0, 0, 1], [2, 0, 1], [2, 1, 1], [0, 1, 1]], dtype=float
    )
    mesh = make_hex_mesh(points)
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS, verbose=False)
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
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS, verbose=False)
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
    SJ = compute_scaled_jacobians(mesh, at_center=False, sample_points=GAUSS_POINTS, verbose=False)
    assert SJ.shape == (1, 8)
    assert np.any(SJ[0] < 1.0)
    assert np.any(SJ[0] < 0.0)  # still not inverted


if __name__ == "__main__":
    pytest.main([__file__])
