# fast_hex_jacobian_jax.py
import jax
import jax.numpy as jnp
import meshio
import numpy as np
from jax import lax

# Reference node signs in MeshIO / VTK order (node 0..7)
# order: (xi, eta, zeta) signs
REF_SIGNS = jnp.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ],
    dtype=jnp.float32,
)
GAUSS_POINTS = np.array(
    [
        [-1, -1, -1],
        [1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
    ],
    dtype=float,
)


def det3x3(M):
    """
    Hand-coded determinant for (...,3,3) arrays.
    Returns shape (...,).
    """
    a00 = M[..., 0, 0]
    a01 = M[..., 0, 1]
    a02 = M[..., 0, 2]
    a10 = M[..., 1, 0]
    a11 = M[..., 1, 1]
    a12 = M[..., 1, 2]
    a20 = M[..., 2, 0]
    a21 = M[..., 2, 1]
    a22 = M[..., 2, 2]
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)


@jax.jit
def _compute_center_jacobians(nodes_xyz, dN_center):
    """
    nodes_xyz: jnp array (E, 8, 3)  -> coordinates of nodes for each element
    dN_center: jnp array (8, 3)     -> dN/d(xi,eta,zeta) at center (same for all elements)

    Returns:
      J: (E, 3, 3)
      detJ: (E,)

    """
    # J_e[a,b] = sum_i X_e[i,a] * dN[i,b]
    # nodes_xyz shape (E,8,3) -> einsum 'e i a, i b -> e a b'
    J = jnp.einsum("eia,ib->eab", nodes_xyz, dN_center)
    detJ = det3x3(J)
    return J, detJ


@jax.jit
def _compute_jacobians_at_points(nodes_xyz, dN_at_q):
    """
    nodes_xyz: (E,8,3)
    dN_at_q: (Q,8,3)  -> derivatives at Q sample points
    Return:
      J: (E, Q, 3, 3)
      detJ: (E, Q)
    """
    # einsum 'e i a, q i b -> e q a b'
    J = jnp.einsum("eia,qib->eqab", nodes_xyz, dN_at_q)
    detJ = det3x3(J)
    return J, detJ


def dN_trilinear_at_samples(xi_eta_zeta, dtype=jnp.float32):
    """
    Compute dN/dxi, dN/deta, dN/dzeta for trilinear hexahedron
    at an array of sample points (xi,eta,zeta).

    xi_eta_zeta: array-like shape (Q,3) with values in [-1,1]
    Returns: jnp array (Q,8,3)
    """
    pts = jnp.asarray(xi_eta_zeta, dtype=dtype)
    if pts.ndim == 1:
        pts = pts[None, :]  # make (1,3)
    xi = pts[:, 0:1]  # (Q,1)
    eta = pts[:, 1:2]
    zeta = pts[:, 2:3]

    r = REF_SIGNS  # (8,3)
    # Broadcast shapes: r[None,:,0] shape (1,8)
    # dN/dxi = 1/8 * r_xi * (1 + r_eta * eta) * (1 + r_zeta * zeta)
    factor = 1.0 / 8.0
    one = 1.0

    dN_xi = factor * (r[None, :, 0] * (one + r[None, :, 1] * eta) * (one + r[None, :, 2] * zeta))
    dN_eta = factor * (r[None, :, 1] * (one + r[None, :, 0] * xi) * (one + r[None, :, 2] * zeta))
    dN_zeta = factor * (r[None, :, 2] * (one + r[None, :, 0] * xi) * (one + r[None, :, 1] * eta))

    # Stack into shape (Q, 8, 3) with order (d/dxi, d/deta, d/dzeta)
    dN = jnp.stack([dN_xi, dN_eta, dN_zeta], axis=-1)  # (Q,8,3)
    return dN.astype(dtype)


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
        else:
            Js = []
            dets = []
            for i0 in range(0, E, chunk_size):
                i1 = min(E, i0 + chunk_size)
                Jc, detc = _compute_center_jacobians(node_coords[i0:i1], dN_center)
                Js.append(np.array(Jc))
                dets.append(np.array(detc))
            return np.vstack(Js), np.concatenate(dets)

    else:
        # sample_points must be provided
        if sample_points is None:
            msg = "sample_points must be provided when at_center=False."
            raise ValueError(msg)
        dN_q = dN_trilinear_at_samples(sample_points, dtype=dtype)  # (Q,8,3)

        if chunk_size is None:
            J, detJ = _compute_jacobians_at_points(node_coords, dN_q)
            return np.array(J), np.array(detJ)
        else:
            Js = []
            dets = []
            for i0 in range(0, E, chunk_size):
                i1 = min(E, i0 + chunk_size)
                Jc, detc = _compute_jacobians_at_points(node_coords[i0:i1], dN_q)
                Js.append(np.array(Jc))
                dets.append(np.array(detc))
            return np.vstack(Js), np.vstack(dets)


@jax.jit
def _scaled_jac_from_center(J, detJ, eps=0.0):
    # Columns of the Jacobian
    Jx, Jy, Jz = J[..., :, 0], J[..., :, 1], J[..., :, 2]
    nx, ny, nz = jnp.linalg.norm(Jx, axis=-1), jnp.linalg.norm(Jy, axis=-1), jnp.linalg.norm(Jz, axis=-1)
    denom = nx * ny * nz

    # Define the normal computation
    def compute_scaled(_):
        return detJ / denom

    # Define the computation with epsilon adjustment if needed
    def compute_with_eps(_):
        return jnp.sign(detJ) * jnp.maximum(jnp.abs(detJ) / denom, eps)

    return jax.lax.cond(eps > 0.0, compute_with_eps, compute_scaled, operand=None)


@jax.jit
def _scaled_jac_from_points(J, detJ, eps=0.0):
    """
    Compute the scaled Jacobian for hexahedral elements at arbitrary points.

    Args:
        J: (E,Q,3,3) array of Jacobians per element and point
        detJ: (E,Q) array of determinants
        eps: small number for regularization (scalar)

    Returns:
        SJ: (E,Q) array of scaled Jacobians
    """
    # Columns of J are the mapped reference axes
    Jx = J[..., :, 0]  # (E,Q,3)
    Jy = J[..., :, 1]
    Jz = J[..., :, 2]

    # Norms of each column vector
    nx = jnp.linalg.norm(Jx, axis=-1)  # (E,Q)
    ny = jnp.linalg.norm(Jy, axis=-1)
    nz = jnp.linalg.norm(Jz, axis=-1)

    denom = nx * ny * nz  # (E,Q)

    # Use lax.cond to avoid Python boolean on traced arrays
    SJ = lax.cond(eps > 0.0, lambda _: detJ / (denom + eps), lambda _: detJ / denom, operand=None)

    return SJ


def compute_scaled_jacobians(
    mesh, dtype=jnp.float32, at_center=True, sample_points=None, chunk_size=None, verbose=True, eps=0.0
):
    """
    Compute scaled Jacobian per element.

    Returns:
      SJ, (numpy array on host)
        - if at_center: shape (E,)
        - else: shape (E,Q)
    Parameters:
      eps: optional small floor for denominator to avoid inf/nan (default 0.0: no floor)
    """
    points = mesh.points.astype(np.float64)
    if "hexahedron" in mesh.cells_dict:
        cells = mesh.cells_dict["hexahedron"].astype(np.int64)
    else:
        found = None
        for block in mesh.cells:
            if block.type in ("hexahedron", "hex8", "hex"):
                found = block.data
                break
        if found is None:
            raise ValueError("No hexahedron cells found in mesh.")
        cells = found.astype(np.int64)

    E = cells.shape[0]
    if verbose:
        print(f"Mesh has {points.shape[0]} points, {E} hexahedral elements.")

    node_coords = points[cells]  # (E,8,3) numpy
    node_coords = jnp.asarray(node_coords, dtype=dtype)

    if at_center:
        dN_center = (REF_SIGNS * (1.0 / 8.0)).astype(dtype)
        if chunk_size is None:
            J, detJ = _compute_center_jacobians(node_coords, dN_center)  # (E,3,3), (E,)
            SJ = _scaled_jac_from_center(J, detJ, eps=eps)
            return np.array(SJ)
        else:
            parts = []
            for i0 in range(0, E, chunk_size):
                i1 = min(E, i0 + chunk_size)
                Jc, detc = _compute_center_jacobians(node_coords[i0:i1], dN_center)
                Sjc = _scaled_jac_from_center(Jc, detc, eps=eps)
                parts.append(np.array(Sjc))
            return np.concatenate(parts)

    else:
        if sample_points is None:
            raise ValueError("sample_points must be provided when at_center=False.")
        dN_q = dN_trilinear_at_samples(sample_points, dtype=dtype)  # (Q,8,3)

        if chunk_size is None:
            J, detJ = _compute_jacobians_at_points(node_coords, dN_q)  # (E,Q,3,3), (E,Q)
            SJ = _scaled_jac_from_points(J, detJ, eps=eps)
            return np.array(SJ)
        else:
            parts = []
            for i0 in range(0, E, chunk_size):
                i1 = min(E, i0 + chunk_size)
                Jc, detc = _compute_jacobians_at_points(node_coords[i0:i1], dN_q)
                Sjc = _scaled_jac_from_points(Jc, detc, eps=eps)
                parts.append(np.array(Sjc))
            return np.vstack(parts)


if __name__ == "__main__":
    # quick demo using a single unit cube (mesh created inline for test)
    # Node ordering follows meshio / VTK convention:
    # bottom: 0,1,2,3 ; top: 4,5,6,7
    unit_cube_nodes = np.array(
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
        dtype=np.float64,
    )
    # single element connectivity
    cells = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    # pack a tiny "mesh" to test compute_jacobians without disk IO:
    # reuse core functions by creating temporary arrays
    node_coords = jnp.asarray(unit_cube_nodes[cells], dtype=jnp.float32)  # (1,8,3)
    dN_center = (REF_SIGNS * 0.125).astype(jnp.float32)

    # warm-up jit compile
    print("Warming up JIT...")
    J_test, det_test = _compute_center_jacobians(node_coords, dN_center)
    print("J (unit cube):\n", np.array(J_test[0]))
    print("detJ (unit cube):", float(np.array(det_test[0])))

    # expected determinant for mapping from [-1,1]^3 -> unit cube [0,1]^3 is 1/8 = 0.125
    print("expected det:", 1.0 / 8.0)
