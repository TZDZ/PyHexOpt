import jax.numpy as jnp
import numpy as np

from pyhexopt.core.jaxobian import (
    REF_SIGNS,
    _compute_center_jacobians,
    _compute_jacobians_at_points,
    dN_trilinear_at_samples,
)
from tests.core.test_jaxobian import make_hex_mesh


def compute_jacobians(mesh, dtype=jnp.float32, at_center=True, sample_points=None, chunk_size=None, verbose=True):
    """
    High-level function:
      mesh_file: path readable by meshio
      dtype: jnp.float32 or jnp.float64
      at_center: if True compute at element center (0,0,0). If False, sample_points must be provided.
      sample_points: array-like (Q,3) of (xi,eta,zeta) in [-1,1] for which to compute Jacobians
      chunk_size: if provided, processes elements in chunks to save memory
    Returns:
      Js, detJs  (both numpy arrays on host)
        - If at_center:  Js shape (E,3,3), detJs shape (E,)
        - else: Js shape (E,Q,3,3), detJs shape (E,Q)
    """
    points = mesh.points.astype(np.float64)  # numpy array (N,3)
    # prefer 'hexahedron' cell type; meshio provides cells_dict convenience
    if "hexahedron" in mesh.cells_dict:
        cells = mesh.cells_dict["hexahedron"].astype(np.int64)  # (E,8)
    else:
        # try finding a hex type in mesh.cells
        found = None
        for block in mesh.cells:
            if block.type in ("hexahedron", "hex8", "hex"):
                found = block.data
                break
        if found is None:
            msg = "No hexahedron cells found in mesh."
            raise ValueError(msg)
        cells = found.astype(np.int64)

    E = cells.shape[0]
    if verbose:
        print(f"Mesh has {points.shape[0]} points, {E} hexahedral elements.")

    # Gather node coordinates per element -> (E,8,3)
    node_coords = points[cells]  # numpy -> shape (E,8,3)
    node_coords = jnp.asarray(node_coords, dtype=dtype)

    if at_center:
        # dN at center (xi=eta=zeta=0) => simplifies to 1/8 * REF_SIGNS
        dN_center = (REF_SIGNS * (1.0 / 8.0)).astype(dtype)  # (8,3)

        # optionally chunked processing
        if chunk_size is None:
            J, detJ = _compute_center_jacobians(node_coords, dN_center)
            return np.array(J), np.array(detJ)
        Js = []
        dets = []
        for i0 in range(0, E, chunk_size):
            i1 = min(E, i0 + chunk_size)
            Jc, detc = _compute_center_jacobians(node_coords[i0:i1], dN_center)
            Js.append(np.array(Jc))
            dets.append(np.array(detc))
        return np.vstack(Js), np.concatenate(dets)

    # sample_points must be provided
    if sample_points is None:
        msg = "sample_points must be provided when at_center=False."
        raise ValueError(msg)
    dN_q = dN_trilinear_at_samples(sample_points, dtype=dtype)  # (Q,8,3)

    if chunk_size is None:
        J, detJ = _compute_jacobians_at_points(node_coords, dN_q)
        return np.array(J), np.array(detJ)
    Js = []
    dets = []
    for i0 in range(0, E, chunk_size):
        i1 = min(E, i0 + chunk_size)
        Jc, detc = _compute_jacobians_at_points(node_coords[i0:i1], dN_q)
        Js.append(np.array(Jc))
        dets.append(np.array(detc))
    return np.vstack(Js), np.vstack(dets)


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
