# fast_hex_jacobian_jax.py
from functools import partial

import jax
import jax.numpy as jnp
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
GAUSS_POINTS = jnp.array(
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
    dtype=jnp.float32,
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


@partial(jax.jit, static_argnames=("dtype"))
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


def compute_scaled_jacobians_from_coords_not_jax(  # noqa: D417, PLR0913
    node_coords,
    dtype=jnp.float32,
    at_center=True,
    sample_points=None,
    chunk_size=None,
    eps=0.0,
):
    """
    Compute scaled Jacobian per element from node coordinates.

    Parameters
    ----------
    node_coords : jax.Array (E,8,3)
        Coordinates of the hexahedron nodes per element
    at_center : bool, default True
        If True, compute at element center (ξ,η,ζ)=(0,0,0)
    sample_points : array-like (Q,3), optional
        Parent coordinates for evaluation (if not at_center)
    chunk_size : int, optional
        Process elements in chunks (for memory)
    eps : float, default 0.0
        Small floor for denominator to avoid inf/nan

    Returns
    -------
    SJ : np.ndarray
        Scaled Jacobian, shape (E,) or (E,Q)

    """
    E = node_coords.shape[0]

    if at_center:
        dN_center = (REF_SIGNS * (1.0 / 8.0)).astype(dtype)
        if chunk_size is None:
            J, detJ = _compute_center_jacobians(node_coords, dN_center)
            SJ = _scaled_jac_from_center(J, detJ, eps=eps)
            return np.array(SJ)
        parts = []
        for i0 in range(0, E, chunk_size):
            i1 = min(E, i0 + chunk_size)
            Jc, detc = _compute_center_jacobians(node_coords[i0:i1], dN_center)
            Sjc = _scaled_jac_from_center(Jc, detc, eps=eps)
            parts.append(np.array(Sjc))
        return np.concatenate(parts)

    # not at center -> use supplied quadrature points
    if sample_points is None:
        msg = "sample_points must be provided when at_center=False."
        raise ValueError(msg)

    dN_q = dN_trilinear_at_samples(sample_points, dtype=dtype)  # (Q,8,3)

    if chunk_size is None:
        J, detJ = _compute_jacobians_at_points(node_coords, dN_q)
        SJ = _scaled_jac_from_points(J, detJ, eps=eps)
        return np.array(SJ)
    parts = []
    for i0 in range(0, E, chunk_size):
        i1 = min(E, i0 + chunk_size)
        Jc, detc = _compute_jacobians_at_points(node_coords[i0:i1], dN_q)
        Sjc = _scaled_jac_from_points(Jc, detc, eps=eps)
        parts.append(np.array(Sjc))
    return np.vstack(parts)


@partial(jax.jit, static_argnames=("at_center", "dtype"))
def compute_scaled_jacobians_from_coords(
    node_coords: jax.Array,
    dtype=jnp.float32,
    at_center: bool = True,
    sample_points: jax.Array | None = None,
    eps: float = 0.0,
) -> jax.Array:
    """Fully JAX version: compute scaled Jacobian per element from node coordinates."""
    node_coords = jnp.asarray(node_coords, dtype=dtype)

    if at_center:
        dN_center = jnp.asarray(REF_SIGNS, dtype=dtype) * (1.0 / 8.0)
        J, detJ = _compute_center_jacobians(node_coords, dN_center)
        SJ = _scaled_jac_from_center(J, detJ, eps=eps)
        return SJ

    if sample_points is None:
        msg = "sample_points must be provided when at_center=False."
        raise ValueError(msg)

    dN_q = dN_trilinear_at_samples(sample_points, dtype=dtype)
    J, detJ = _compute_jacobians_at_points(node_coords, dN_q)
    SJ = _scaled_jac_from_points(J, detJ, eps=eps)
    return SJ
